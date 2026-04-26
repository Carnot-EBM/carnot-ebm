"""Tests for Experiment 843: Governance Pre-flight v14 — .65 RETRO Audit.

Traces to:
    REQ-INFRA-060: The governance pre-flight MUST audit all open RETROs from the prior
        milestone retro and produce a JSON artifact before any new milestone experiments run.
    SCENARIO-INFRA-070: When all three deliverables (retirement_plan.md,
        manifest_enforcement_patch.txt, MILESTONE_PREREQS.md .65 section) are written,
        honest_verdict must be "governance_ready".
"""

import json
import os
from pathlib import Path

import pytest

from scripts.experiment_843_preflight_v14 import (
    DELIVERABLE,
    EXPERIMENT_CAP,
    EXPERIMENT_ID,
    IMMEDIATE_RETIREMENT_EXP_IDS,
    IMMEDIATE_RETIREMENT_SAVINGS_MIN,
    OPEN_RETROS,
    RETROS_CONFIRMED_CLOSED,
    compute_honest_verdict,
    extract_audit_data,
    run_audit,
    update_milestone_prereqs,
    write_manifest_patch,
    write_retirement_plan,
)


# ---------------------------------------------------------------------------
# Tests for OPEN_RETROS / RETROS_CONFIRMED_CLOSED constants
# ---------------------------------------------------------------------------


class TestRetroConstants:
    """REQ-INFRA-060: constants must reflect the authoritative .64 retro state."""

    def test_nine_open_retros(self):
        """OPEN_RETROS must contain exactly 9 items (authoritative .64 retro count)."""
        assert len(OPEN_RETROS) == 9

    def test_two_closed_retros(self):
        """RETROS_CONFIRMED_CLOSED must contain exactly 2 items closed in .64."""
        assert len(RETROS_CONFIRMED_CLOSED) == 2

    def test_symcode_serial_in_closed(self):
        """RETRO-SYMCODE-SERIAL must be in RETROS_CONFIRMED_CLOSED (Exp 841 closure)."""
        assert "RETRO-SYMCODE-SERIAL" in RETROS_CONFIRMED_CLOSED

    def test_tier1_plateau_in_closed(self):
        """RETRO-TIER1-PLATEAU must be in RETROS_CONFIRMED_CLOSED (governance closure)."""
        assert "RETRO-TIER1-PLATEAU" in RETROS_CONFIRMED_CLOSED

    def test_manifest_full_scope_in_open(self):
        """RETRO-MANIFEST-FULL-SCOPE must be open (unguarded dequeue sites still exist)."""
        assert "RETRO-MANIFEST-FULL-SCOPE" in OPEN_RETROS

    def test_svamp_zero_auc_in_open(self):
        """RETRO-SVAMP-ZERO-AUC must be open (new in .64, SVAMP collapsed in Exp 834)."""
        assert "RETRO-SVAMP-ZERO-AUC" in OPEN_RETROS

    def test_ice40_pnr_in_open(self):
        """RETRO-ICE40-PNR-LUT-OVERFLOW must be open (new in .64, Exp 839 pnr_failed)."""
        assert "RETRO-ICE40-PNR-LUT-OVERFLOW" in OPEN_RETROS

    def test_open_and_closed_disjoint(self):
        """No RETRO can appear in both OPEN_RETROS and RETROS_CONFIRMED_CLOSED."""
        overlap = set(OPEN_RETROS) & set(RETROS_CONFIRMED_CLOSED)
        assert len(overlap) == 0, f"Overlap found: {overlap}"

    def test_experiment_cap_is_700(self):
        """EXPERIMENT_CAP must be 700 (the project's declared cap)."""
        assert EXPERIMENT_CAP == 700

    def test_immediate_retirement_ids(self):
        """Immediate retirement list must include Exp 786, 527, 627."""
        assert 786 in IMMEDIATE_RETIREMENT_EXP_IDS
        assert 527 in IMMEDIATE_RETIREMENT_EXP_IDS
        assert 627 in IMMEDIATE_RETIREMENT_EXP_IDS

    def test_immediate_retirement_savings(self):
        """Savings must equal 77+52+51=180 min/milestone from Exp 786+527+627."""
        assert IMMEDIATE_RETIREMENT_SAVINGS_MIN == 180


# ---------------------------------------------------------------------------
# Tests for extract_audit_data()
# ---------------------------------------------------------------------------


class TestExtractAuditData:
    """REQ-INFRA-060: extract_audit_data must correctly derive counts from retro dict."""

    def test_extracts_open_retros_count(self):
        """open_retros_count must equal the length of retros_still_open list."""
        retro = {"retros_still_open": ["R1", "R2", "R3"], "experiments_completed": 700}
        result = extract_audit_data(retro)
        assert result["open_retros_count"] == 3
        assert result["open_retros"] == ["R1", "R2", "R3"]

    def test_extracts_closed_retros(self):
        """retros_confirmed_closed must equal the retros_closed list from the retro dict."""
        retro = {"retros_closed": ["RETRO-A", "RETRO-B"], "experiments_completed": 700}
        result = extract_audit_data(retro)
        assert result["retros_confirmed_closed"] == ["RETRO-A", "RETRO-B"]

    def test_experiments_over_cap_positive(self):
        """experiments_over_cap must be experiments_completed minus EXPERIMENT_CAP when positive."""
        retro = {"experiments_completed": 750, "retros_still_open": []}
        result = extract_audit_data(retro)
        assert result["experiments_completed"] == 750
        assert result["experiments_over_cap"] == 50  # 750 - 700

    def test_experiments_over_cap_zero_when_at_cap(self):
        """experiments_over_cap must be 0 when experiments_completed equals EXPERIMENT_CAP."""
        retro = {"experiments_completed": 700, "retros_still_open": []}
        result = extract_audit_data(retro)
        assert result["experiments_over_cap"] == 0

    def test_experiments_over_cap_zero_when_below_cap(self):
        """experiments_over_cap must be 0 (not negative) when below cap."""
        retro = {"experiments_completed": 650, "retros_still_open": []}
        result = extract_audit_data(retro)
        assert result["experiments_over_cap"] == 0

    def test_missing_fields_default_to_empty(self):
        """extract_audit_data must not raise when retro dict is missing optional fields."""
        result = extract_audit_data({})
        assert result["open_retros"] == []
        assert result["open_retros_count"] == 0
        assert result["retros_confirmed_closed"] == []
        assert result["experiments_completed"] == 0
        assert result["experiments_over_cap"] == 0


# ---------------------------------------------------------------------------
# Tests for compute_honest_verdict()
# ---------------------------------------------------------------------------


class TestComputeHonestVerdict:
    """SCENARIO-INFRA-070: honest_verdict must follow the encoding contract."""

    def test_all_true_gives_governance_ready(self):
        """All three deliverables written -> governance_ready."""
        assert compute_honest_verdict(True, True, True) == "governance_ready"

    def test_prereqs_false_gives_partial(self):
        """Missing prereqs -> governance_partial."""
        assert compute_honest_verdict(False, True, True) == "governance_partial"

    def test_retirement_plan_false_gives_partial(self):
        """Missing retirement plan -> governance_partial."""
        assert compute_honest_verdict(True, False, True) == "governance_partial"

    def test_manifest_patch_false_gives_partial(self):
        """Missing manifest patch -> governance_partial."""
        assert compute_honest_verdict(True, True, False) == "governance_partial"

    def test_all_false_gives_partial(self):
        """All three missing -> governance_partial (not a crash)."""
        assert compute_honest_verdict(False, False, False) == "governance_partial"


# ---------------------------------------------------------------------------
# Tests for write_retirement_plan()
# ---------------------------------------------------------------------------


class TestWriteRetirementPlan:
    """REQ-INFRA-060: retirement_plan.md must be written with required content."""

    def test_writes_file_when_absent(self, tmp_path):
        """write_retirement_plan must create the file and return True when absent."""
        path = tmp_path / "retirement_plan.md"
        result = write_retirement_plan(path)
        assert result is True
        assert path.exists()
        assert path.stat().st_size > 0

    def test_file_contains_exp_786(self, tmp_path):
        """retirement_plan.md must mention Exp 786 (highest-ROI immediate retirement)."""
        path = tmp_path / "retirement_plan.md"
        write_retirement_plan(path)
        content = path.read_text(encoding="utf-8")
        assert "786" in content

    def test_file_contains_exp_527(self, tmp_path):
        """retirement_plan.md must mention Exp 527 (11th+ post-retirement appearance)."""
        path = tmp_path / "retirement_plan.md"
        write_retirement_plan(path)
        content = path.read_text(encoding="utf-8")
        assert "527" in content

    def test_file_contains_exp_627(self, tmp_path):
        """retirement_plan.md must mention Exp 627 (RETRO-SYMCODE-SERIAL proof case)."""
        path = tmp_path / "retirement_plan.md"
        write_retirement_plan(path)
        content = path.read_text(encoding="utf-8")
        assert "627" in content

    def test_idempotent_when_file_exists(self, tmp_path):
        """write_retirement_plan must return True without re-writing if file already exists."""
        path = tmp_path / "retirement_plan.md"
        path.write_text("existing content", encoding="utf-8")
        result = write_retirement_plan(path)
        assert result is True
        # File should not be overwritten — original content preserved.
        assert path.read_text(encoding="utf-8") == "existing content"


# ---------------------------------------------------------------------------
# Tests for write_manifest_patch()
# ---------------------------------------------------------------------------


class TestWriteManifestPatch:
    """REQ-INFRA-060: manifest_enforcement_patch.txt must be written with required content."""

    def test_writes_file_when_absent(self, tmp_path):
        """write_manifest_patch must create the file and return True when absent."""
        path = tmp_path / "manifest_enforcement_patch.txt"
        result = write_manifest_patch(path)
        assert result is True
        assert path.exists()
        assert path.stat().st_size > 0

    def test_file_contains_exp_id_is_excluded(self, tmp_path):
        """Patch must describe the _exp_id_is_excluded helper function."""
        path = tmp_path / "manifest_enforcement_patch.txt"
        write_manifest_patch(path)
        content = path.read_text(encoding="utf-8")
        assert "_exp_id_is_excluded" in content

    def test_file_contains_pick_next_task_reference(self, tmp_path):
        """Patch must explain that pick_next_task() already has the manifest check."""
        path = tmp_path / "manifest_enforcement_patch.txt"
        write_manifest_patch(path)
        content = path.read_text(encoding="utf-8")
        assert "pick_next_task" in content

    def test_file_contains_validation_smoke_test(self, tmp_path):
        """Patch must include a validation command to confirm the patch worked."""
        path = tmp_path / "manifest_enforcement_patch.txt"
        write_manifest_patch(path)
        content = path.read_text(encoding="utf-8")
        # Validation section describes running Python to check exclusion
        assert "excluded=True" in content or "validation" in content.lower()

    def test_idempotent_when_file_exists(self, tmp_path):
        """write_manifest_patch must return True without re-writing if file already exists."""
        path = tmp_path / "manifest_enforcement_patch.txt"
        path.write_text("existing patch content", encoding="utf-8")
        result = write_manifest_patch(path)
        assert result is True
        assert path.read_text(encoding="utf-8") == "existing patch content"


# ---------------------------------------------------------------------------
# Tests for update_milestone_prereqs()
# ---------------------------------------------------------------------------


class TestUpdateMilestonePrereqs:
    """REQ-INFRA-060: MILESTONE_PREREQS.md must gain a .65 section without losing .64 content."""

    def test_appends_section_to_empty_file(self, tmp_path):
        """update_milestone_prereqs must create and write to an empty prereqs file."""
        path = tmp_path / "MILESTONE_PREREQS.md"
        result = update_milestone_prereqs(path)
        assert result is True
        content = path.read_text(encoding="utf-8")
        assert "Milestone 2026.04.65 Pre-flight" in content

    def test_appends_section_without_removing_existing_content(self, tmp_path):
        """update_milestone_prereqs must NOT remove prior milestone sections."""
        path = tmp_path / "MILESTONE_PREREQS.md"
        path.write_text(
            "## Milestone 2026.04.64 Pre-flight\n\nExisting content.\n", encoding="utf-8"
        )
        update_milestone_prereqs(path)
        content = path.read_text(encoding="utf-8")
        assert "Milestone 2026.04.64 Pre-flight" in content
        assert "Existing content." in content
        assert "Milestone 2026.04.65 Pre-flight" in content

    def test_idempotent_when_section_already_present(self, tmp_path):
        """update_milestone_prereqs must return True without re-writing if section exists."""
        path = tmp_path / "MILESTONE_PREREQS.md"
        path.write_text(
            "## Milestone 2026.04.65 Pre-flight\n\nAlready written.\n", encoding="utf-8"
        )
        result = update_milestone_prereqs(path)
        assert result is True
        # Content must not be duplicated.
        content = path.read_text(encoding="utf-8")
        assert content.count("Milestone 2026.04.65 Pre-flight") == 1

    def test_section_contains_open_retros(self, tmp_path):
        """The .65 section must list all 9 open RETROs."""
        path = tmp_path / "MILESTONE_PREREQS.md"
        update_milestone_prereqs(path)
        content = path.read_text(encoding="utf-8")
        for retro in OPEN_RETROS:
            assert retro in content, f"Missing RETRO: {retro}"

    def test_section_contains_closed_retros(self, tmp_path):
        """The .65 section must list the 2 RETROs closed in .64."""
        path = tmp_path / "MILESTONE_PREREQS.md"
        update_milestone_prereqs(path)
        content = path.read_text(encoding="utf-8")
        for retro in RETROS_CONFIRMED_CLOSED:
            assert retro in content, f"Missing closed RETRO: {retro}"

    def test_section_contains_assertions(self, tmp_path):
        """The .65 section must include key assertion requirements."""
        path = tmp_path / "MILESTONE_PREREQS.md"
        update_milestone_prereqs(path)
        content = path.read_text(encoding="utf-8")
        assert "n_svamp_pairs" in content
        assert "n_arc_pairs" in content
        assert "warm_start_sweeps" in content


# ---------------------------------------------------------------------------
# Integration test: run_audit() end-to-end against real results/ directory
# ---------------------------------------------------------------------------


class TestRunAudit:
    """Integration test: run_audit() against the real .64 retro JSON."""

    def test_run_audit_returns_governance_ready(self, tmp_path):
        """run_audit() must return honest_verdict='governance_ready' with real retro data."""
        # Use real results/ dir for the retro JSON, but write governance artifacts to tmp.
        result = run_audit(
            results_dir=Path("results"),
            prereqs_path=tmp_path / "MILESTONE_PREREQS.md",
            retirement_plan_path=tmp_path / "retirement_plan.md",
            manifest_patch_path=tmp_path / "manifest_enforcement_patch.txt",
        )
        assert result["honest_verdict"] == "governance_ready"

    def test_run_audit_open_retros_count(self, tmp_path):
        """run_audit() must report 9 open RETROs from the .64 retro JSON."""
        result = run_audit(
            results_dir=Path("results"),
            prereqs_path=tmp_path / "MILESTONE_PREREQS.md",
            retirement_plan_path=tmp_path / "retirement_plan.md",
            manifest_patch_path=tmp_path / "manifest_enforcement_patch.txt",
        )
        assert result["open_retros_count"] == 9

    def test_run_audit_experiments_over_cap(self, tmp_path):
        """run_audit() must report 50 experiments over cap (750 - 700)."""
        result = run_audit(
            results_dir=Path("results"),
            prereqs_path=tmp_path / "MILESTONE_PREREQS.md",
            retirement_plan_path=tmp_path / "retirement_plan.md",
            manifest_patch_path=tmp_path / "manifest_enforcement_patch.txt",
        )
        assert result["experiments_over_cap"] == 50

    def test_run_audit_all_deliverables_written(self, tmp_path):
        """run_audit() must set all three written flags to True."""
        result = run_audit(
            results_dir=Path("results"),
            prereqs_path=tmp_path / "MILESTONE_PREREQS.md",
            retirement_plan_path=tmp_path / "retirement_plan.md",
            manifest_patch_path=tmp_path / "manifest_enforcement_patch.txt",
        )
        assert result["retirement_plan_written"] is True
        assert result["prereqs_updated"] is True
        assert result["manifest_patch_written"] is True

    def test_run_audit_immediate_retirement_savings(self, tmp_path):
        """run_audit() must report 180 min savings from immediate retirements."""
        result = run_audit(
            results_dir=Path("results"),
            prereqs_path=tmp_path / "MILESTONE_PREREQS.md",
            retirement_plan_path=tmp_path / "retirement_plan.md",
            manifest_patch_path=tmp_path / "manifest_enforcement_patch.txt",
        )
        assert result["immediate_retirement_savings_min"] == 180
        assert result["immediate_retirement_exp_ids"] == [786, 527, 627]

    def test_run_audit_retros_confirmed_closed(self, tmp_path):
        """run_audit() must list RETRO-SYMCODE-SERIAL and RETRO-TIER1-PLATEAU as closed."""
        result = run_audit(
            results_dir=Path("results"),
            prereqs_path=tmp_path / "MILESTONE_PREREQS.md",
            retirement_plan_path=tmp_path / "retirement_plan.md",
            manifest_patch_path=tmp_path / "manifest_enforcement_patch.txt",
        )
        assert "RETRO-SYMCODE-SERIAL" in result["retros_confirmed_closed"]
        assert "RETRO-TIER1-PLATEAU" in result["retros_confirmed_closed"]


# ---------------------------------------------------------------------------
# Tests for the written deliverable JSON
# ---------------------------------------------------------------------------


class TestDeliverableJson:
    """Verify the experiment deliverable JSON has all required fields."""

    def test_deliverable_exists(self):
        """results/experiment_843_preflight_v14.json must exist on disk."""
        assert Path(DELIVERABLE).exists(), f"Deliverable not found: {DELIVERABLE}"

    def test_deliverable_is_valid_json(self):
        """Deliverable must parse as valid JSON."""
        with open(DELIVERABLE, encoding="utf-8") as fh:
            artifact = json.load(fh)
        assert isinstance(artifact, dict)

    def test_deliverable_experiment_id(self):
        """Deliverable experiment field must be 843."""
        with open(DELIVERABLE, encoding="utf-8") as fh:
            artifact = json.load(fh)
        assert artifact["experiment"] == EXPERIMENT_ID

    def test_deliverable_honest_verdict(self):
        """Deliverable honest_verdict must be 'governance_ready'."""
        with open(DELIVERABLE, encoding="utf-8") as fh:
            artifact = json.load(fh)
        assert artifact["honest_verdict"] == "governance_ready"

    def test_deliverable_open_retros_count(self):
        """Deliverable open_retros_count must be 9."""
        with open(DELIVERABLE, encoding="utf-8") as fh:
            artifact = json.load(fh)
        assert artifact["open_retros_count"] == 9

    def test_deliverable_retirement_plan_written(self):
        """Deliverable retirement_plan_written must be True."""
        with open(DELIVERABLE, encoding="utf-8") as fh:
            artifact = json.load(fh)
        assert artifact["retirement_plan_written"] is True

    def test_deliverable_prereqs_updated(self):
        """Deliverable prereqs_updated must be True."""
        with open(DELIVERABLE, encoding="utf-8") as fh:
            artifact = json.load(fh)
        assert artifact["prereqs_updated"] is True

    def test_deliverable_manifest_patch_written(self):
        """Deliverable manifest_patch_written must be True."""
        with open(DELIVERABLE, encoding="utf-8") as fh:
            artifact = json.load(fh)
        assert artifact["manifest_patch_written"] is True
