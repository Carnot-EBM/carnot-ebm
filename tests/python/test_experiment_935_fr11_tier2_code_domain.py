"""Tests for Experiment 935: FR-11 Tier 2 Code Domain Memory.

Every test in this file traces to REQ-LEARN-060 or SCENARIO-LEARN-104.

The tests are deliberately self-contained: they do NOT read the real
experiment_905 JSON from disk (that would make them fragile to missing
fixtures).  Instead, they build minimal synthetic fixtures that exercise
the same code paths.

Spec: REQ-LEARN-060, SCENARIO-LEARN-104
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Path wiring — tests run from the repo root or from pytest.
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.experiment_935_fr11_tier2_code_domain import (  # noqa: E402
    _categorise_by_retries,
    _retry_strategy_for_category,
    _make_code_repair_template,
    load_exp905_patterns,
    session1_populate_library,
    session2_replay_templates,
    SOURCE_MODEL_ID,
    REPLAY_PROBLEMS,
)
from carnot.pipeline.constraint_template_library import ConstraintTemplateLibrary  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_exp905_fixture(repaired: list[dict], failed: list[dict], baseline: list[dict]) -> dict:
    """Build a minimal Exp 905-style JSON dict for testing.

    Args:
        repaired:  Problems with baseline_passed=False, repair_passed=True.
        failed:    Problems with both passed=False.
        baseline:  Problems with baseline_passed=True.
    """
    return {
        "experiment": 905,
        "results_per_problem": repaired + failed + baseline,
    }


def _repaired_problem(task_id: str, n_retries: int) -> dict:
    return {
        "task_id": task_id,
        "baseline_passed": False,
        "repair_passed": True,
        "n_retries": n_retries,
        "energy_score_best": -1.0,
    }


def _failed_problem(task_id: str) -> dict:
    return {
        "task_id": task_id,
        "baseline_passed": False,
        "repair_passed": False,
        "n_retries": 3,
        "energy_score_best": -5.0,
    }


def _baseline_problem(task_id: str) -> dict:
    return {
        "task_id": task_id,
        "baseline_passed": True,
        "repair_passed": True,
        "n_retries": 0,
        "energy_score_best": -2.0,
    }


# ---------------------------------------------------------------------------
# Tests for _categorise_by_retries — REQ-LEARN-060-2
# ---------------------------------------------------------------------------


class TestCategoriseByRetries:
    """Spec: REQ-LEARN-060-2"""

    def test_zero_retries_is_easy(self):
        # REQ-LEARN-060-2
        assert _categorise_by_retries(0) == "easy_fix"

    def test_one_retry_is_easy(self):
        # REQ-LEARN-060-2
        assert _categorise_by_retries(1) == "easy_fix"

    def test_two_retries_is_medium(self):
        # REQ-LEARN-060-2
        assert _categorise_by_retries(2) == "medium_fix"

    def test_three_retries_is_hard(self):
        # REQ-LEARN-060-2
        assert _categorise_by_retries(3) == "hard_fix"

    def test_high_retries_is_hard(self):
        # REQ-LEARN-060-2: any n >= 3 → hard_fix
        assert _categorise_by_retries(10) == "hard_fix"


# ---------------------------------------------------------------------------
# Tests for _retry_strategy_for_category — REQ-LEARN-060-2
# ---------------------------------------------------------------------------


class TestRetryStrategyForCategory:
    """Spec: REQ-LEARN-060-2"""

    def test_easy_strategy_is_string(self):
        # REQ-LEARN-060-2
        s = _retry_strategy_for_category("easy_fix")
        assert isinstance(s, str) and len(s) > 0

    def test_medium_strategy_is_string(self):
        # REQ-LEARN-060-2
        s = _retry_strategy_for_category("medium_fix")
        assert isinstance(s, str) and len(s) > 0

    def test_hard_strategy_is_string(self):
        # REQ-LEARN-060-2
        s = _retry_strategy_for_category("hard_fix")
        assert isinstance(s, str) and len(s) > 0

    def test_unknown_category_returns_fallback(self):
        # REQ-LEARN-060-2: unknown categories don't raise, return fallback string
        s = _retry_strategy_for_category("unknown_category")
        assert isinstance(s, str) and len(s) > 0


# ---------------------------------------------------------------------------
# Tests for _make_code_repair_template — REQ-LEARN-060-2
# ---------------------------------------------------------------------------


class TestMakeCodeRepairTemplate:
    """Spec: REQ-LEARN-060-2"""

    def test_pattern_key_matches_error_type(self):
        # REQ-LEARN-060-2
        t = _make_code_repair_template("easy_fix", "one-pass repair")
        assert t.pattern_key == "code_repair_easy_fix"

    def test_template_fn_returns_constraint_result(self):
        # REQ-LEARN-060-2: template_fn always emits one hint
        t = _make_code_repair_template("easy_fix", "one-pass repair")
        results = t.template_fn("any response text")
        assert len(results) == 1

    def test_template_fn_result_has_correct_type(self):
        # REQ-LEARN-060-2
        t = _make_code_repair_template("medium_fix", "two-pass repair")
        results = t.template_fn("")
        assert results[0].constraint_type == "code_repair_medium_fix"

    def test_template_fn_metadata_satisfied_true(self):
        # REQ-LEARN-060-2: advisory templates are always satisfied=True
        t = _make_code_repair_template("hard_fix", "multi-pass repair")
        results = t.template_fn("def foo(): ...")
        assert results[0].metadata["satisfied"] is True


# ---------------------------------------------------------------------------
# Tests for load_exp905_patterns — REQ-LEARN-060-1
# ---------------------------------------------------------------------------


class TestLoadExp905Patterns:
    """Spec: REQ-LEARN-060-1"""

    def test_only_repaired_problems_returned(self, tmp_path):
        # REQ-LEARN-060-1: only baseline_passed=False AND repair_passed=True
        fixture = _make_exp905_fixture(
            repaired=[_repaired_problem("HumanEval/10", 1)],
            failed=[_failed_problem("HumanEval/0")],
            baseline=[_baseline_problem("HumanEval/1")],
        )
        fp = tmp_path / "exp905.json"
        fp.write_text(json.dumps(fixture))
        patterns = load_exp905_patterns(fp)
        assert len(patterns) == 1
        assert patterns[0]["task_id"] == "HumanEval/10"

    def test_baseline_pass_excluded(self, tmp_path):
        # REQ-LEARN-060-1: baseline-passing problems are excluded
        fixture = _make_exp905_fixture(
            repaired=[],
            failed=[],
            baseline=[_baseline_problem("HumanEval/1")],
        )
        fp = tmp_path / "exp905.json"
        fp.write_text(json.dumps(fixture))
        patterns = load_exp905_patterns(fp)
        assert len(patterns) == 0

    def test_failed_both_excluded(self, tmp_path):
        # REQ-LEARN-060-1: failed-both problems are excluded
        fixture = _make_exp905_fixture(
            repaired=[],
            failed=[_failed_problem("HumanEval/0")],
            baseline=[],
        )
        fp = tmp_path / "exp905.json"
        fp.write_text(json.dumps(fixture))
        patterns = load_exp905_patterns(fp)
        assert len(patterns) == 0

    def test_pattern_has_required_keys(self, tmp_path):
        # REQ-LEARN-060-1
        fixture = _make_exp905_fixture(
            repaired=[_repaired_problem("HumanEval/10", 2)],
            failed=[],
            baseline=[],
        )
        fp = tmp_path / "exp905.json"
        fp.write_text(json.dumps(fixture))
        patterns = load_exp905_patterns(fp)
        p = patterns[0]
        for key in (
            "task_id",
            "n_retries",
            "energy_score_best",
            "error_category",
            "retry_strategy",
        ):
            assert key in p

    def test_multiple_repaired_problems(self, tmp_path):
        # REQ-LEARN-060-1: all 3 difficulty categories present in 17 Exp 905 problems
        repaired = (
            [_repaired_problem(f"HumanEval/{i}", 1) for i in range(8)]  # easy_fix
            + [_repaired_problem(f"HumanEval/{i + 100}", 2) for i in range(5)]  # medium_fix
            + [_repaired_problem(f"HumanEval/{i + 200}", 3) for i in range(4)]  # hard_fix
        )
        fixture = _make_exp905_fixture(repaired=repaired, failed=[], baseline=[])
        fp = tmp_path / "exp905.json"
        fp.write_text(json.dumps(fixture))
        patterns = load_exp905_patterns(fp)
        assert len(patterns) == 17
        categories = {p["error_category"] for p in patterns}
        assert categories == {"easy_fix", "medium_fix", "hard_fix"}


# ---------------------------------------------------------------------------
# Tests for session1_populate_library — REQ-LEARN-060-2, REQ-LEARN-060-3
# ---------------------------------------------------------------------------


class TestSession1PopulateLibrary:
    """Spec: REQ-LEARN-060-2, REQ-LEARN-060-3"""

    def _make_patterns(self) -> list[dict]:
        from scripts.experiment_935_fr11_tier2_code_domain import (
            _categorise_by_retries,
            _retry_strategy_for_category,
        )

        raw = (
            [_repaired_problem(f"HumanEval/{i}", 1) for i in range(8)]
            + [_repaired_problem(f"HumanEval/{i + 100}", 2) for i in range(5)]
            + [_repaired_problem(f"HumanEval/{i + 200}", 3) for i in range(4)]
        )
        patterns = []
        for r in raw:
            cat = _categorise_by_retries(r["n_retries"])
            patterns.append(
                {
                    "task_id": r["task_id"],
                    "n_retries": r["n_retries"],
                    "energy_score_best": r["energy_score_best"],
                    "error_category": cat,
                    "retry_strategy": _retry_strategy_for_category(cat),
                }
            )
        return patterns

    def test_three_templates_added(self):
        # REQ-LEARN-060-2: one template per distinct category
        patterns = self._make_patterns()
        library, n_added = session1_populate_library(patterns)
        assert n_added == 3

    def test_all_templates_active_after_session1(self):
        # REQ-LEARN-060-3: every added template must be active immediately
        patterns = self._make_patterns()
        library, _ = session1_populate_library(patterns)
        active = library.get_active_templates(SOURCE_MODEL_ID)
        assert len(active) == 3

    def test_observation_counts_positive(self):
        # REQ-LEARN-060-2: observe_pattern called once per problem
        patterns = self._make_patterns()
        library, _ = session1_populate_library(patterns)
        obs_dict = library.to_dict()
        for entry in obs_dict["observations"]:
            assert entry["count"] >= 1

    def test_empty_patterns_returns_zero_templates(self):
        # REQ-LEARN-060-2: edge case — no repaired problems → no templates
        library, n_added = session1_populate_library([])
        assert n_added == 0
        assert library.get_active_templates(SOURCE_MODEL_ID) == []


# ---------------------------------------------------------------------------
# Tests for cross-session persistence — REQ-LEARN-060-4
# ---------------------------------------------------------------------------


class TestCrossSessionPersistence:
    """Spec: REQ-LEARN-060-4"""

    def _populate_library(self) -> ConstraintTemplateLibrary:
        from scripts.experiment_935_fr11_tier2_code_domain import (
            _categorise_by_retries,
            _retry_strategy_for_category,
        )

        raw = (
            [_repaired_problem(f"HumanEval/{i}", 1) for i in range(3)]
            + [_repaired_problem(f"HumanEval/{i + 100}", 2) for i in range(2)]
            + [_repaired_problem(f"HumanEval/{i + 200}", 3) for i in range(2)]
        )
        patterns = [
            {
                "task_id": r["task_id"],
                "n_retries": r["n_retries"],
                "energy_score_best": r["energy_score_best"],
                "error_category": _categorise_by_retries(r["n_retries"]),
                "retry_strategy": _retry_strategy_for_category(
                    _categorise_by_retries(r["n_retries"])
                ),
            }
            for r in raw
        ]
        library, _ = session1_populate_library(patterns)
        return library

    def test_to_dict_has_observations(self):
        # REQ-LEARN-060-4
        library = self._populate_library()
        d = library.to_dict()
        assert "observations" in d
        assert len(d["observations"]) == 3

    def test_from_dict_restores_counts(self):
        # REQ-LEARN-060-4: round-trip preserves observation counts
        library = self._populate_library()
        d = library.to_dict()
        restored = ConstraintTemplateLibrary.from_dict(d)
        # Re-register templates (callables not serializable).
        from scripts.experiment_935_fr11_tier2_code_domain import (
            _make_code_repair_template,
            _retry_strategy_for_category,
        )

        for category in ("easy_fix", "medium_fix", "hard_fix"):
            restored.add_template(
                _make_code_repair_template(category, _retry_strategy_for_category(category))
            )
        # All three should be active after round-trip.
        active = restored.get_active_templates(SOURCE_MODEL_ID)
        assert len(active) == 3

    def test_json_round_trip_via_string(self, tmp_path):
        # REQ-LEARN-060-4: full JSON serialization → file → deserialization
        library = self._populate_library()
        d = library.to_dict()
        p = tmp_path / "lib.json"
        p.write_text(json.dumps(d, indent=2))
        reloaded = json.loads(p.read_text())
        assert reloaded["observations"] == d["observations"]


# ---------------------------------------------------------------------------
# Tests for session2_replay_templates — REQ-LEARN-060-5
# ---------------------------------------------------------------------------


class TestSession2ReplayTemplates:
    """Spec: REQ-LEARN-060-5, SCENARIO-LEARN-104"""

    def _library_dict_with_active_templates(self) -> dict:
        from scripts.experiment_935_fr11_tier2_code_domain import (
            _categorise_by_retries,
            _retry_strategy_for_category,
        )

        raw = (
            [_repaired_problem(f"HumanEval/{i}", 1) for i in range(3)]
            + [_repaired_problem(f"HumanEval/{i + 100}", 2) for i in range(2)]
            + [_repaired_problem(f"HumanEval/{i + 200}", 3) for i in range(2)]
        )
        patterns = [
            {
                "task_id": r["task_id"],
                "n_retries": r["n_retries"],
                "energy_score_best": r["energy_score_best"],
                "error_category": _categorise_by_retries(r["n_retries"]),
                "retry_strategy": _retry_strategy_for_category(
                    _categorise_by_retries(r["n_retries"])
                ),
            }
            for r in raw
        ]
        library, _ = session1_populate_library(patterns)
        return library.to_dict()

    def test_templates_replayed_in_s2_positive(self):
        # REQ-LEARN-060-5: at least 1 problem matches a template
        lib_dict = self._library_dict_with_active_templates()
        result = session2_replay_templates(lib_dict, ["HumanEval/25"])
        assert result["templates_replayed_in_s2"] >= 1

    def test_replay_improvement_positive(self):
        # REQ-LEARN-060-5: replay_improvement > 0 when templates active
        lib_dict = self._library_dict_with_active_templates()
        result = session2_replay_templates(lib_dict, REPLAY_PROBLEMS)
        assert result["replay_improvement"] > 0

    def test_template_match_rate_one(self):
        # SCENARIO-LEARN-104: all 10 problems match since templates are advisory
        lib_dict = self._library_dict_with_active_templates()
        result = session2_replay_templates(lib_dict, REPLAY_PROBLEMS)
        assert result["template_match_rate"] == 1.0

    def test_problem_results_length_matches_input(self):
        # REQ-LEARN-060-5: one result entry per input problem
        lib_dict = self._library_dict_with_active_templates()
        result = session2_replay_templates(lib_dict, REPLAY_PROBLEMS)
        assert len(result["problem_results"]) == len(REPLAY_PROBLEMS)

    def test_empty_library_gives_zero_replay(self):
        # REQ-LEARN-060-5: edge case — empty library → no matches
        empty_dict: dict = {"observations": []}
        result = session2_replay_templates(empty_dict, REPLAY_PROBLEMS)
        assert result["templates_replayed_in_s2"] == 0
        assert result["replay_improvement"] == 0.0

    def test_problem_results_have_task_id(self):
        # SCENARIO-LEARN-104
        lib_dict = self._library_dict_with_active_templates()
        result = session2_replay_templates(lib_dict, ["HumanEval/25", "HumanEval/26"])
        for pr in result["problem_results"]:
            assert "task_id" in pr
            assert "template_matched" in pr

    def test_empty_problem_list(self):
        # REQ-LEARN-060-5: edge case — no problems → zero rates
        lib_dict = self._library_dict_with_active_templates()
        result = session2_replay_templates(lib_dict, [])
        assert result["templates_replayed_in_s2"] == 0
        assert result["template_match_rate"] == 0.0


# ---------------------------------------------------------------------------
# Integration test: end-to-end Session 1 → persist → Session 2
# ---------------------------------------------------------------------------


class TestEndToEnd:
    """Integration test covering the full Exp 935 pipeline.

    Spec: REQ-LEARN-060, SCENARIO-LEARN-104
    """

    def test_full_pipeline_tier2_code_memory_works(self, tmp_path):
        """Full pipeline: load patterns → populate → persist → reload → replay.

        Spec: REQ-LEARN-060, SCENARIO-LEARN-104
        """
        # Build fixture matching Exp 905's 17-problem structure.
        from scripts.experiment_935_fr11_tier2_code_domain import (
            _categorise_by_retries,
            _retry_strategy_for_category,
        )

        raw = (
            [_repaired_problem(f"HumanEval/{i}", 1) for i in range(8)]
            + [_repaired_problem(f"HumanEval/{i + 100}", 2) for i in range(5)]
            + [_repaired_problem(f"HumanEval/{i + 200}", 3) for i in range(4)]
        )
        fixture_path = tmp_path / "exp905.json"
        fixture_path.write_text(json.dumps(_make_exp905_fixture(raw, [], [])))

        # Step 1: load patterns.
        patterns = load_exp905_patterns(fixture_path)
        assert len(patterns) == 17

        # Step 2: Session 1 populate.
        library, n_added = session1_populate_library(patterns)
        assert n_added >= 3  # REQ-LEARN-060

        # Step 3: persist.
        lib_path = tmp_path / "lib.json"
        lib_path.write_text(json.dumps(library.to_dict(), indent=2))

        # Step 4: reload.
        reloaded_dict = json.loads(lib_path.read_text())

        # Step 5: Session 2 replay.
        s2 = session2_replay_templates(reloaded_dict, REPLAY_PROBLEMS)
        assert s2["templates_replayed_in_s2"] >= 1  # REQ-LEARN-060-5
        assert s2["replay_improvement"] > 0  # SCENARIO-LEARN-104

        # Step 6: determine honest_verdict.
        if n_added == 0:
            verdict = "tier2_code_memory_plateau"
        elif s2["templates_replayed_in_s2"] >= 1 and s2["replay_improvement"] > 0:
            verdict = "tier2_code_memory_works"
        else:
            verdict = "tier2_code_memory_partial"
        assert verdict == "tier2_code_memory_works"  # SCENARIO-LEARN-104
