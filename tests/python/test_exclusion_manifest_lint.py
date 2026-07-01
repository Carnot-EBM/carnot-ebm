"""Tests for scripts/exclusion_manifest_lint.py's BLOCKED_PATTERN_MATCHED class.

Origin: 2026-07-01 outer-loop incident. `.469`'s planner ran 8 minutes AFTER a
same-session known-issues.md retraction landed (the FoVer in-domain
candidate-selection-pool premise, proven a construction artifact) but still
emitted 3 tasks asserting the retracted premise as fact. Neither pre-existing
violation class caught it: EXP_ID_RETIRED only matches task ids that reuse a
retired exp_id (these were brand-new ids), and SCOPE_MATCHED_PRIOR_FAILURE
only matches PAST ARTIFACT scope-signatures (these ids had no prior artifact
to match against). An outer-loop session had to hand-patch the live roadmap
after the fact.

BLOCKED_PATTERN_MATCHED closes that gap: it checks every draft task's
title+prompt against `ops/exclusion_manifest.yaml`'s `retired_extras[].
blocked_patterns` entries, regardless of the task's own id or prior artifact
history. These tests pin the contract using a synthetic manifest (isolated
from the real, evolving ops/exclusion_manifest.yaml via PROJECT_ROOT
monkeypatching) so they don't churn as real retirement entries are added:

  - a task whose id/title looks unrelated but whose PROMPT contains a
    blocked_pattern string: HARD-blocked
  - the same task with a valid operator_override: WARNING (not blocked)
  - the same task with a valid prior_failures: block: WARNING (not blocked)
  - a task with no matching content: clean (no risk emitted)
  - matching is case-insensitive
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import yaml


def _load():
    repo_root = Path(__file__).resolve().parents[2]
    module_path = repo_root / "scripts" / "exclusion_manifest_lint.py"
    spec = importlib.util.spec_from_file_location("exclusion_manifest_lint", module_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["exclusion_manifest_lint"] = mod
    spec.loader.exec_module(mod)
    return mod


_MOD = _load()

_SYNTHETIC_MANIFEST = {
    "retired": [],
    "retired_experiments": [],
    "retired_extras": [
        {
            "id": "synthetic_retired_scope_vTEST",
            "experiment_scope": "synthetic retired scope for testing",
            "reason": "retire_if_same_verdict: synthetic test retirement",
            "experiment_ids": ["exp1", "exp2"],
            "retired_milestone": "2026.07.999",
            "retired_by_artifact": "results/experiment_1_synthetic.json",
            "recorded_by_artifact": "results/experiment_2_synthetic.json",
            "operator_reopen_required": True,
            "retire_if_same_verdict": True,
            "blocked_patterns": [
                "synthetic blocked phrase",
                "another blocked pattern here",
            ],
        }
    ],
}


@pytest.fixture
def isolated_project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point the loaded module at a temp PROJECT_ROOT with a synthetic manifest
    and an empty results/ dir (so FailureLedger has nothing to scope-match on,
    isolating BLOCKED_PATTERN_MATCHED from CLASS 2's artifact-history check)."""
    ops_dir = tmp_path / "ops"
    ops_dir.mkdir()
    (ops_dir / "exclusion_manifest.yaml").write_text(yaml.safe_dump(_SYNTHETIC_MANIFEST))
    (tmp_path / "results").mkdir()
    (tmp_path / "scripts").mkdir()
    monkeypatch.setattr(_MOD, "PROJECT_ROOT", tmp_path)
    return tmp_path


def _write_roadmap(tmp_path: Path, tasks: list[dict]) -> Path:
    path = tmp_path / "research-roadmap-next.yaml"
    path.write_text(yaml.safe_dump({"milestone": "2026.07.999", "tasks": tasks}))
    return path


class TestBlockedPatternMatched:
    def test_prompt_match_on_unrelated_id_is_hard_blocked(self, isolated_project: Path) -> None:
        """SCENARIO: a brand-new task id (no prior artifact, doesn't reuse a
        retired id) whose PROMPT contains a blocked_pattern string must still
        be caught -- this is the exact `.469` incident shape."""
        roadmap = _write_roadmap(
            isolated_project,
            [
                {
                    "id": "exp9999-totally-unrelated-scope",
                    "title": "PHASE Q unrelated-sounding title",
                    "prompt": "CONTEXT: this task will exercise the synthetic blocked phrase here.",
                    "agent_type": "codex",
                }
            ],
        )
        risks = _MOD.lint(roadmap)
        matched = [r for r in risks if r.violation_class == "BLOCKED_PATTERN_MATCHED"]
        assert len(matched) == 1
        assert matched[0].severity == "HARD"
        assert "synthetic_retired_scope_vTEST" in matched[0].detail

    def test_no_match_is_clean(self, isolated_project: Path) -> None:
        roadmap = _write_roadmap(
            isolated_project,
            [
                {
                    "id": "exp9998-clean-task",
                    "title": "PHASE Q something completely different",
                    "prompt": "CONTEXT: no overlap with any retired scope at all.",
                    "agent_type": "codex",
                }
            ],
        )
        risks = _MOD.lint(roadmap)
        assert [r for r in risks if r.violation_class == "BLOCKED_PATTERN_MATCHED"] == []

    def test_operator_override_downgrades_to_warning(self, isolated_project: Path) -> None:
        roadmap = _write_roadmap(
            isolated_project,
            [
                {
                    "id": "exp9997-with-override",
                    "title": "PHASE Q unrelated title",
                    "prompt": "CONTEXT: this task will exercise the synthetic blocked phrase here.",
                    "operator_override": "2026-07-01 operator directive: reopening with a new technique.",
                    "agent_type": "codex",
                }
            ],
        )
        risks = _MOD.lint(roadmap)
        matched = [r for r in risks if r.violation_class == "BLOCKED_PATTERN_MATCHED"]
        assert len(matched) == 1
        assert matched[0].severity == "WARNING"
        assert matched[0].has_operator_override is True

    def test_valid_prior_failures_downgrades_to_warning(self, isolated_project: Path) -> None:
        roadmap = _write_roadmap(
            isolated_project,
            [
                {
                    "id": "exp9996-with-prior-failures",
                    "title": "PHASE Q unrelated title",
                    "prompt": "CONTEXT: this task will exercise the synthetic blocked phrase here.",
                    "prior_failures": [
                        {
                            "experiment_id": "exp1",
                            "verdict": "synthetic_null",
                            "addressed_by": "a genuinely new technique",
                            "retire_if_same_verdict": True,
                        }
                    ],
                    "agent_type": "codex",
                }
            ],
        )
        risks = _MOD.lint(roadmap)
        matched = [r for r in risks if r.violation_class == "BLOCKED_PATTERN_MATCHED"]
        assert len(matched) == 1
        assert matched[0].severity == "WARNING"

    def test_matching_is_case_insensitive(self, isolated_project: Path) -> None:
        roadmap = _write_roadmap(
            isolated_project,
            [
                {
                    "id": "exp9995-case-test",
                    "title": "PHASE Q Title Case",
                    "prompt": "CONTEXT: this exercises SYNTHETIC BLOCKED PHRASE in caps.",
                    "agent_type": "codex",
                }
            ],
        )
        risks = _MOD.lint(roadmap)
        assert len([r for r in risks if r.violation_class == "BLOCKED_PATTERN_MATCHED"]) == 1

    def test_main_exits_nonzero_on_hard_violation(
        self, isolated_project: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        roadmap = _write_roadmap(
            isolated_project,
            [
                {
                    "id": "exp9994-cli-test",
                    "title": "PHASE Q cli test",
                    "prompt": "CONTEXT: another blocked pattern here triggers this.",
                    "agent_type": "codex",
                }
            ],
        )
        old_argv = sys.argv
        sys.argv = ["exclusion_manifest_lint.py", str(roadmap)]
        try:
            exit_code = _MOD.main()
        finally:
            sys.argv = old_argv
        assert exit_code == 1
        out = capsys.readouterr().out
        assert "BLOCKED_PATTERN_MATCHED" in out
