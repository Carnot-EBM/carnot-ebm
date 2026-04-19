"""Tests for Exp 518 — Batching Migration Sprint.

Covers all helper functions in scripts/experiment_518_batching_migration_sprint.py:
- extract_exp_id: parses exp ID from script path
- estimate_savings_minutes: heuristic savings estimate
- group_violations_by_script: groups flat violation list
- rank_scripts_by_savings: sorts by estimated savings
- find_simple_loop: detects auto-migratable sequential loop
- build_bir_replacement: builds the BatchedInferenceRunner replacement text
- ensure_bir_import: injects import when missing, idempotent when present
- attempt_script_migration: end-to-end migration attempt on temp files

Spec: REQ-INFRA-047, REQ-INFRA-048,
      SCENARIO-INFRA-055, SCENARIO-INFRA-056
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "scripts"))

import experiment_518_batching_migration_sprint as exp518  # noqa: E402


# ---------------------------------------------------------------------------
# extract_exp_id
# ---------------------------------------------------------------------------


class TestExtractExpId:
    """extract_exp_id parses the integer experiment ID from a script filename."""

    def test_standard_path(self):
        # Normal experiment script: extracts the number between 'experiment_' and '_'
        result = exp518.extract_exp_id(
            "/repo/scripts/experiment_123_some_title.py"
        )
        assert result == 123

    def test_non_experiment_script(self):
        # generate_qa_dataset.py has no 'experiment_NNN_' pattern → returns None
        result = exp518.extract_exp_id(
            "/repo/scripts/generate_qa_dataset.py"
        )
        assert result is None

    def test_high_number(self):
        # Large exp IDs (>400) exist and must parse cleanly
        result = exp518.extract_exp_id("experiment_479_gsm_symbolic_adversarial_live.py")
        assert result == 479

    def test_filename_only(self):
        # Works with just the filename, no directory prefix
        result = exp518.extract_exp_id("experiment_42_foo.py")
        assert result == 42


# ---------------------------------------------------------------------------
# estimate_savings_minutes
# ---------------------------------------------------------------------------


class TestEstimateSavingsMinutes:
    """estimate_savings_minutes returns sensible wall-time estimates."""

    def test_known_exp_uses_fraction(self):
        # Exp 308 has known wall time of 105 min; 20% = 21.0
        result = exp518.estimate_savings_minutes(308, n_violations=1)
        assert result == pytest.approx(105.0 * 0.20)

    def test_unknown_exp_low_number_no_recency(self):
        # exp_id below recency threshold (400): base rate only
        result = exp518.estimate_savings_minutes(100, n_violations=2)
        assert result == pytest.approx(3.0 * 2 * 1.0)

    def test_unknown_exp_high_number_with_recency(self):
        # exp_id >= 400 gets 1.5x recency factor
        result = exp518.estimate_savings_minutes(450, n_violations=2)
        assert result == pytest.approx(3.0 * 2 * 1.5)

    def test_none_exp_id_no_recency(self):
        # Non-experiment scripts (exp_id=None) get base rate, no recency
        result = exp518.estimate_savings_minutes(None, n_violations=1)
        assert result == pytest.approx(3.0 * 1 * 1.0)

    def test_boundary_exp_id_400(self):
        # exp_id == 400 should trigger recency factor
        result = exp518.estimate_savings_minutes(400, n_violations=1)
        assert result == pytest.approx(3.0 * 1 * 1.5)


# ---------------------------------------------------------------------------
# group_violations_by_script
# ---------------------------------------------------------------------------


class TestGroupViolationsByScript:
    """group_violations_by_script consolidates the flat violation list."""

    def test_single_script_single_violation(self):
        violations = [{"script_path": "/a/exp_1.py", "line_no": 10}]
        grouped = exp518.group_violations_by_script(violations)
        assert list(grouped.keys()) == ["/a/exp_1.py"]
        assert len(grouped["/a/exp_1.py"]) == 1

    def test_single_script_multiple_violations(self):
        # Same script appearing twice should produce one key with two entries
        violations = [
            {"script_path": "/a/exp_1.py", "line_no": 10},
            {"script_path": "/a/exp_1.py", "line_no": 20},
        ]
        grouped = exp518.group_violations_by_script(violations)
        assert len(grouped) == 1
        assert len(grouped["/a/exp_1.py"]) == 2

    def test_multiple_scripts(self):
        violations = [
            {"script_path": "/a/exp_1.py", "line_no": 10},
            {"script_path": "/a/exp_2.py", "line_no": 5},
        ]
        grouped = exp518.group_violations_by_script(violations)
        assert len(grouped) == 2

    def test_empty_list(self):
        assert exp518.group_violations_by_script([]) == {}


# ---------------------------------------------------------------------------
# rank_scripts_by_savings
# ---------------------------------------------------------------------------


class TestRankScriptsBySavings:
    """rank_scripts_by_savings sorts scripts by estimated_savings_min descending."""

    def test_known_exp_ranks_higher_than_unknown(self):
        # Exp 308 known 105 min → 21 min savings; unknown exp 50 → 3 min savings
        grouped = {
            "/s/experiment_308_foo.py": [{"script_path": "/s/experiment_308_foo.py"}],
            "/s/experiment_50_bar.py": [{"script_path": "/s/experiment_50_bar.py"}],
        }
        ranked = exp518.rank_scripts_by_savings(grouped)
        assert ranked[0]["exp_id"] == 308

    def test_result_has_required_keys(self):
        grouped = {
            "/s/experiment_100_x.py": [{"script_path": "/s/experiment_100_x.py"}],
        }
        ranked = exp518.rank_scripts_by_savings(grouped)
        row = ranked[0]
        assert "script_path" in row
        assert "exp_id" in row
        assert "n_violations" in row
        assert "estimated_savings_min" in row
        assert "violations" in row

    def test_more_violations_ranks_higher(self):
        # Two unknown scripts: 3 violations vs 1 violation (same exp range)
        grouped = {
            "/s/experiment_100_x.py": [{"script_path": "/s/experiment_100_x.py"}] * 3,
            "/s/experiment_101_y.py": [{"script_path": "/s/experiment_101_y.py"}],
        }
        ranked = exp518.rank_scripts_by_savings(grouped)
        assert ranked[0]["exp_id"] == 100  # 3 violations > 1 violation

    def test_empty_grouped(self):
        assert exp518.rank_scripts_by_savings({}) == []


# ---------------------------------------------------------------------------
# find_simple_loop
# ---------------------------------------------------------------------------


class TestFindSimpleLoop:
    """find_simple_loop detects the exact 3-line auto-migratable pattern."""

    def test_detects_simple_pattern(self):
        content = (
            "results = []\n"
            "for q in questions:\n"
            "    ans = infer(q)\n"
            "    results.append(ans)\n"
            "print(results)\n"
        )
        m = exp518.find_simple_loop(content)
        assert m is not None
        assert m.group("var") == "q"
        assert m.group("items") == "questions"
        assert m.group("fn") == "infer"
        assert m.group("results") == "results"

    def test_indented_pattern(self):
        # Inside a function — should still match with correct indent
        content = (
            "def run():\n"
            "    out = []\n"
            "    for item in samples:\n"
            "        r = model(item)\n"
            "        out.append(r)\n"
            "    return out\n"
        )
        m = exp518.find_simple_loop(content)
        assert m is not None
        assert m.group("indent") == "    "

    def test_complex_body_not_matched(self):
        # Multi-statement body (3 lines) should NOT match
        content = (
            "for q in questions:\n"
            "    ans = infer(q)\n"
            "    results.append(ans)\n"
            "    count += 1\n"
        )
        m = exp518.find_simple_loop(content)
        assert m is None

    def test_no_loop_in_content(self):
        content = "x = 1\ny = 2\n"
        assert exp518.find_simple_loop(content) is None

    def test_different_var_names(self):
        content = (
            "for prob in problems:\n"
            "    result = run_model(prob)\n"
            "    output.append(result)\n"
        )
        m = exp518.find_simple_loop(content)
        assert m is not None
        assert m.group("var") == "prob"
        assert m.group("fn") == "run_model"


# ---------------------------------------------------------------------------
# build_bir_replacement
# ---------------------------------------------------------------------------


class TestBuildBirReplacement:
    """build_bir_replacement produces correct BatchedInferenceRunner code."""

    def _make_match(self, content: str) -> "re.Match":
        m = exp518.find_simple_loop(content)
        assert m is not None, "Test setup: simple loop not found in content"
        return m

    def test_replacement_contains_bir(self):
        content = (
            "for q in questions:\n"
            "    r = infer(q)\n"
            "    res.append(r)\n"
        )
        m = self._make_match(content)
        rep = exp518.build_bir_replacement(m)
        assert "BatchedInferenceRunner" in rep
        assert "run_batch" in rep
        assert "infer" in rep
        assert "questions" in rep
        assert "res" in rep

    def test_replacement_three_lines(self):
        content = (
            "for q in questions:\n"
            "    r = fn(q)\n"
            "    out.append(r)\n"
        )
        m = self._make_match(content)
        rep = exp518.build_bir_replacement(m)
        # The replacement is exactly 3 lines
        lines = rep.splitlines()
        assert len(lines) == 3

    def test_indentation_preserved(self):
        content = (
            "    for q in questions:\n"
            "        r = fn(q)\n"
            "        out.append(r)\n"
        )
        m = self._make_match(content)
        rep = exp518.build_bir_replacement(m)
        # Each replacement line should start with the same 4-space indent
        for line in rep.splitlines():
            assert line.startswith("    ")


# ---------------------------------------------------------------------------
# ensure_bir_import
# ---------------------------------------------------------------------------


class TestEnsureBirImport:
    """ensure_bir_import injects the import exactly once."""

    def test_already_imported_noop(self):
        content = (
            "from experiment_template import BatchedInferenceRunner\n"
            "x = 1\n"
        )
        result = exp518.ensure_bir_import(content)
        # Should be unchanged when BatchedInferenceRunner is already present
        assert result == content
        assert result.count("BatchedInferenceRunner") == 1

    def test_inserts_after_experiment_template_import(self):
        content = (
            "from experiment_template import ExperimentTemplate\n"
            "x = 1\n"
        )
        result = exp518.ensure_bir_import(content)
        # BatchedInferenceRunner import should appear right after the existing import
        lines = result.splitlines()
        et_idx = next(i for i, l in enumerate(lines) if "ExperimentTemplate" in l)
        bir_idx = next(i for i, l in enumerate(lines) if "BatchedInferenceRunner" in l)
        assert bir_idx == et_idx + 1

    def test_inserts_before_first_import_when_no_template_import(self):
        content = "import json\nx = 1\n"
        result = exp518.ensure_bir_import(content)
        assert "BatchedInferenceRunner" in result
        lines = result.splitlines()
        bir_idx = next(i for i, l in enumerate(lines) if "BatchedInferenceRunner" in l)
        json_idx = next(i for i, l in enumerate(lines) if l.strip() == "import json")
        assert bir_idx < json_idx

    def test_prepend_when_no_imports(self):
        # File with no import statements at all — falls through to prepend
        content = "x = 1\ny = 2\n"
        result = exp518.ensure_bir_import(content)
        assert result.startswith(exp518._BIR_IMPORT_LINE)

    def test_idempotent(self):
        # Calling twice should not double-inject
        content = "import json\nx = 1\n"
        once = exp518.ensure_bir_import(content)
        twice = exp518.ensure_bir_import(once)
        assert once == twice


# ---------------------------------------------------------------------------
# attempt_script_migration
# ---------------------------------------------------------------------------


class TestAttemptScriptMigration:
    """attempt_script_migration end-to-end: reads, patches, verifies, writes."""

    def test_file_not_found(self):
        result = exp518.attempt_script_migration("/nonexistent/path/script.py")
        assert result["success"] is False
        assert result["reason"] == "file_not_found"
        assert result["lines_changed"] == 0

    def test_no_simple_loop(self, tmp_path):
        # Script with a complex loop (no match) → no_simple_loop_found
        script = tmp_path / "experiment_999_complex.py"
        script.write_text(
            "for q in questions:\n"
            "    ans = infer(q)\n"
            "    log.info(ans)\n"
            "    results.append(ans)\n",
            encoding="utf-8",
        )
        result = exp518.attempt_script_migration(str(script))
        assert result["success"] is False
        assert result["reason"] == "no_simple_loop_found"

    def test_successful_migration(self, tmp_path):
        # Script with exactly the simple 3-line pattern → migrated
        script = tmp_path / "experiment_777_simple.py"
        script.write_text(
            "import json\n"
            "from experiment_template import ExperimentTemplate\n"
            "questions = ['a', 'b']\n"
            "results = []\n"
            "for q in questions:\n"
            "    r = my_fn(q)\n"
            "    results.append(r)\n"
            "print(results)\n",
            encoding="utf-8",
        )
        result = exp518.attempt_script_migration(str(script))
        assert result["success"] is True
        assert result["reason"] == "simple_loop_replaced"
        assert result["lines_changed"] >= 0

        # Verify the file now contains BatchedInferenceRunner
        patched = script.read_text(encoding="utf-8")
        assert "BatchedInferenceRunner" in patched
        assert "run_batch" in patched
        # Original for-loop should be gone
        assert "for q in questions:" not in patched

    def test_successful_migration_valid_python(self, tmp_path):
        # The patched file must parse as valid Python
        import ast as _ast

        script = tmp_path / "experiment_778_valid.py"
        script.write_text(
            "import json\n"
            "questions = ['x']\n"
            "results = []\n"
            "for q in questions:\n"
            "    r = fn(q)\n"
            "    results.append(r)\n",
            encoding="utf-8",
        )
        exp518.attempt_script_migration(str(script))
        patched = script.read_text(encoding="utf-8")
        # Should not raise
        _ast.parse(patched)

    def test_returns_dict_keys(self, tmp_path):
        # Return value always has required keys regardless of outcome
        script = tmp_path / "experiment_780_keys.py"
        script.write_text("x = 1\n", encoding="utf-8")
        result = exp518.attempt_script_migration(str(script))
        assert "script_path" in result
        assert "success" in result
        assert "reason" in result
        assert "lines_changed" in result
