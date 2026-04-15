"""Tests for python/carnot/pipeline/conductor_env.py — RETRO-012/014 fix.

Coverage targets (100% required)
---------------------------------
- ConductorEnvFix dataclass: fields present and correctly typed
- build_conductor_env_fix: creates scripts/conductor_gpu_env.sh, returns correct dataclass
- verify_env_script_exports: True when export present, False when absent, False when path missing
- RetroJSONEnforcer.check_result_json_exists: True/False based on glob match
- RetroJSONEnforcer.audit_missing_jsons: returns only missing experiment IDs

Spec: REQ-INFRA-015, REQ-INFRA-016,
      SCENARIO-INFRA-016, SCENARIO-INFRA-017, SCENARIO-INFRA-018
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from python.carnot.pipeline.conductor_env import (
    ConductorEnvFix,
    RetroJSONEnforcer,
    build_conductor_env_fix,
    verify_env_script_exports,
)


# ---------------------------------------------------------------------------
# ConductorEnvFix dataclass
# ---------------------------------------------------------------------------


class TestConductorEnvFix:
    """REQ-INFRA-015: ConductorEnvFix dataclass fields."""

    def test_fields_present(self, tmp_path: Path) -> None:
        """All four required fields exist and are accessible."""
        fix = ConductorEnvFix(
            env_script_path=tmp_path / "conductor_gpu_env.sh",
            exports={"CARNOT_FORCE_LIVE": "1"},
            apply_cmd="source scripts/conductor_gpu_env.sh",
            is_documented=True,
        )
        assert fix.env_script_path == tmp_path / "conductor_gpu_env.sh"
        assert fix.exports == {"CARNOT_FORCE_LIVE": "1"}
        assert fix.apply_cmd == "source scripts/conductor_gpu_env.sh"
        assert fix.is_documented is True

    def test_is_documented_false(self, tmp_path: Path) -> None:
        """is_documented can be False."""
        fix = ConductorEnvFix(
            env_script_path=tmp_path / "x.sh",
            exports={},
            apply_cmd="source x.sh",
            is_documented=False,
        )
        assert fix.is_documented is False


# ---------------------------------------------------------------------------
# build_conductor_env_fix
# ---------------------------------------------------------------------------


class TestBuildConductorEnvFix:
    """SCENARIO-INFRA-016: build_conductor_env_fix creates script + returns dataclass."""

    def test_creates_script_file(self, tmp_path: Path) -> None:
        """The function writes scripts/conductor_gpu_env.sh under project_root."""
        scripts_dir = tmp_path / "scripts"
        scripts_dir.mkdir()
        build_conductor_env_fix(tmp_path)
        assert (tmp_path / "scripts" / "conductor_gpu_env.sh").exists()

    def test_script_contains_export(self, tmp_path: Path) -> None:
        """The script contains export CARNOT_FORCE_LIVE=1."""
        (tmp_path / "scripts").mkdir()
        build_conductor_env_fix(tmp_path)
        content = (tmp_path / "scripts" / "conductor_gpu_env.sh").read_text()
        assert "export CARNOT_FORCE_LIVE=1" in content

    def test_script_contains_retro_comment(self, tmp_path: Path) -> None:
        """The script contains a comment identifying RETRO-012."""
        (tmp_path / "scripts").mkdir()
        build_conductor_env_fix(tmp_path)
        content = (tmp_path / "scripts" / "conductor_gpu_env.sh").read_text()
        assert "RETRO-012" in content

    def test_script_contains_shebang(self, tmp_path: Path) -> None:
        """The script starts with a bash shebang."""
        (tmp_path / "scripts").mkdir()
        build_conductor_env_fix(tmp_path)
        content = (tmp_path / "scripts" / "conductor_gpu_env.sh").read_text()
        assert content.startswith("#!/usr/bin/env bash")

    def test_returns_conductor_env_fix(self, tmp_path: Path) -> None:
        """The return value is a ConductorEnvFix instance."""
        (tmp_path / "scripts").mkdir()
        result = build_conductor_env_fix(tmp_path)
        assert isinstance(result, ConductorEnvFix)

    def test_exports_dict_correct(self, tmp_path: Path) -> None:
        """exports dict maps CARNOT_FORCE_LIVE to '1'."""
        (tmp_path / "scripts").mkdir()
        result = build_conductor_env_fix(tmp_path)
        assert result.exports.get("CARNOT_FORCE_LIVE") == "1"

    def test_apply_cmd_is_source(self, tmp_path: Path) -> None:
        """apply_cmd is 'source scripts/conductor_gpu_env.sh'."""
        (tmp_path / "scripts").mkdir()
        result = build_conductor_env_fix(tmp_path)
        assert result.apply_cmd == "source scripts/conductor_gpu_env.sh"

    def test_is_documented_true(self, tmp_path: Path) -> None:
        """is_documented is True for a freshly created script."""
        (tmp_path / "scripts").mkdir()
        result = build_conductor_env_fix(tmp_path)
        assert result.is_documented is True

    def test_env_script_path_correct(self, tmp_path: Path) -> None:
        """env_script_path points to the created file."""
        (tmp_path / "scripts").mkdir()
        result = build_conductor_env_fix(tmp_path)
        assert result.env_script_path == tmp_path / "scripts" / "conductor_gpu_env.sh"
        assert result.env_script_path.exists()

    def test_idempotent_second_call(self, tmp_path: Path) -> None:
        """Calling twice does not raise — script is overwritten cleanly."""
        (tmp_path / "scripts").mkdir()
        build_conductor_env_fix(tmp_path)
        result = build_conductor_env_fix(tmp_path)
        assert result.exports.get("CARNOT_FORCE_LIVE") == "1"

    def test_creates_scripts_dir_if_missing(self, tmp_path: Path) -> None:
        """scripts/ directory is created if it does not exist."""
        # Do NOT pre-create scripts/
        result = build_conductor_env_fix(tmp_path)
        assert result.env_script_path.exists()


# ---------------------------------------------------------------------------
# verify_env_script_exports
# ---------------------------------------------------------------------------


class TestVerifyEnvScriptExports:
    """SCENARIO-INFRA-016/017: verify_env_script_exports."""

    def test_returns_true_when_export_present(self, tmp_path: Path) -> None:
        """Returns True when export CARNOT_FORCE_LIVE=1 is in the script."""
        f = tmp_path / "env.sh"
        f.write_text("#!/usr/bin/env bash\nexport CARNOT_FORCE_LIVE=1\n")
        assert verify_env_script_exports(f) is True

    def test_returns_false_when_export_absent(self, tmp_path: Path) -> None:
        """Returns False when the variable is not exported."""
        f = tmp_path / "env.sh"
        f.write_text("#!/usr/bin/env bash\n# no exports\n")
        assert verify_env_script_exports(f) is False

    def test_returns_false_when_path_missing(self, tmp_path: Path) -> None:
        """Returns False when the path does not exist."""
        assert verify_env_script_exports(tmp_path / "nonexistent.sh") is False

    def test_returns_false_for_partial_match(self, tmp_path: Path) -> None:
        """Returns False when only CARNOT_FORCE_LIVE without =1 is present."""
        f = tmp_path / "env.sh"
        f.write_text("#!/usr/bin/env bash\nexport CARNOT_FORCE_LIVE=0\n")
        assert verify_env_script_exports(f) is False

    def test_returns_true_for_build_output(self, tmp_path: Path) -> None:
        """Returns True for a script produced by build_conductor_env_fix."""
        (tmp_path / "scripts").mkdir()
        fix = build_conductor_env_fix(tmp_path)
        assert verify_env_script_exports(fix.env_script_path) is True


# ---------------------------------------------------------------------------
# RetroJSONEnforcer.check_result_json_exists
# ---------------------------------------------------------------------------


class TestCheckResultJsonExists:
    """REQ-INFRA-016, SCENARIO-INFRA-018."""

    def test_returns_true_when_file_exists(self, tmp_path: Path) -> None:
        """Returns True when experiment_357_*.json exists."""
        (tmp_path / "experiment_357_llm_z3_formalizer.json").write_text("{}")
        enforcer = RetroJSONEnforcer()
        assert enforcer.check_result_json_exists(357, tmp_path) is True

    def test_returns_false_when_file_missing(self, tmp_path: Path) -> None:
        """Returns False when no experiment_358_*.json exists."""
        enforcer = RetroJSONEnforcer()
        assert enforcer.check_result_json_exists(358, tmp_path) is False

    def test_returns_false_when_results_dir_empty(self, tmp_path: Path) -> None:
        """Returns False when results directory is empty."""
        enforcer = RetroJSONEnforcer()
        assert enforcer.check_result_json_exists(362, tmp_path) is False

    def test_returns_true_any_slug(self, tmp_path: Path) -> None:
        """Any filename matching experiment_NNN_*.json counts."""
        (tmp_path / "experiment_362_any_slug_here.json").write_text("{}")
        enforcer = RetroJSONEnforcer()
        assert enforcer.check_result_json_exists(362, tmp_path) is True

    def test_ignores_wrong_exp_id(self, tmp_path: Path) -> None:
        """experiment_363_*.json does not satisfy check for exp_id=362."""
        (tmp_path / "experiment_363_other.json").write_text("{}")
        enforcer = RetroJSONEnforcer()
        assert enforcer.check_result_json_exists(362, tmp_path) is False


# ---------------------------------------------------------------------------
# RetroJSONEnforcer.audit_missing_jsons
# ---------------------------------------------------------------------------


class TestAuditMissingJsons:
    """SCENARIO-INFRA-018: audit_missing_jsons returns only missing IDs."""

    def test_returns_missing_ids(self, tmp_path: Path) -> None:
        """When 357 exists but 358 and 362 do not, returns [358, 362]."""
        (tmp_path / "experiment_357_llm_z3_formalizer.json").write_text("{}")
        enforcer = RetroJSONEnforcer()
        missing = enforcer.audit_missing_jsons([357, 358, 362], tmp_path)
        assert missing == [358, 362]

    def test_returns_empty_when_all_present(self, tmp_path: Path) -> None:
        """Returns [] when every experiment has a result JSON."""
        for eid in [357, 358, 362]:
            (tmp_path / f"experiment_{eid}_results.json").write_text("{}")
        enforcer = RetroJSONEnforcer()
        assert enforcer.audit_missing_jsons([357, 358, 362], tmp_path) == []

    def test_returns_all_when_none_present(self, tmp_path: Path) -> None:
        """Returns all IDs when directory is empty."""
        enforcer = RetroJSONEnforcer()
        assert enforcer.audit_missing_jsons([357, 358, 362], tmp_path) == [357, 358, 362]

    def test_order_preserved(self, tmp_path: Path) -> None:
        """Missing IDs are returned in the same order as input."""
        enforcer = RetroJSONEnforcer()
        missing = enforcer.audit_missing_jsons([362, 358, 357], tmp_path)
        assert missing == [362, 358, 357]

    def test_empty_list(self, tmp_path: Path) -> None:
        """Empty input list returns empty output."""
        enforcer = RetroJSONEnforcer()
        assert enforcer.audit_missing_jsons([], tmp_path) == []
