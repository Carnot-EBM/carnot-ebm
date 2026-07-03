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
import json
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
    # Pre-cache the REAL failure_ledger module in sys.modules before any test
    # monkeypatches PROJECT_ROOT: lint()'s lazy `from failure_ledger import ...`
    # does `sys.path.insert(0, str(PROJECT_ROOT / "scripts"))` using whatever
    # PROJECT_ROOT is AT CALL TIME -- with PROJECT_ROOT monkeypatched to an
    # isolated tmp_path (empty scripts/ dir), that import would silently fail
    # (caught by lint()'s broad except Exception), leaving validate_prior_failures
    # None and every match forced to the not-valid-priors HARD path regardless
    # of a task's actual prior_failures content. Importing it here, from the
    # real scripts/ dir, caches it in sys.modules so the later `from
    # failure_ledger import ...` inside lint() resolves from cache regardless
    # of sys.path at call time (failure_ledger itself doesn't read PROJECT_ROOT
    # at import time -- only FailureLedger.load_from_artifacts(PROJECT_ROOT),
    # called explicitly with the correct, monkeypatched value, does).
    sys.path.insert(0, str(repo_root / "scripts"))
    import failure_ledger  # noqa: F401

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

    def test_valid_prior_failures_clears_the_check_entirely(self, isolated_project: Path) -> None:
        """A well-formed prior_failures: block means the task properly addressed
        the retirement (per Failed-Experiment Rerun Discipline) -- matching
        CLASS 2 (SCOPE_MATCHED_PRIOR_FAILURE)'s convention, this clears the
        check entirely (no risk at all), not a downgrade to WARNING. Only
        operator_override downgrades to WARNING; a valid prior_failures block
        is a full pass."""
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
        assert [r for r in risks if r.violation_class == "BLOCKED_PATTERN_MATCHED"] == []

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

    def test_source_scope_audit_task_is_exempt(self, isolated_project: Path) -> None:
        """SCENARIO: a recurring per-milestone hygiene task whose job is to CHECK for
        retired-scope reruns must not be blocked for MENTIONING a retired scope while
        auditing for it -- caught as a real false positive on `.471`'s
        exp5135-v471-source-scope-audit, whose prompt literally says 'verify... that no
        task reruns retired FoVer or other exclusion-manifest scopes' and got
        HARD-blocked for containing the word 'fover'. Confirmed recurring via
        python/carnot/experiment_5123_v470_source_scope_audit.py (the .470 predecessor)."""
        roadmap = _write_roadmap(
            isolated_project,
            [
                {
                    "id": "exp5135-v471-source-scope-audit",
                    "title": "PHASE 0B source/scope audit",
                    "prompt": (
                        "CONTEXT: verify that no task reruns retired scopes. This "
                        "exercises the synthetic blocked phrase here as an example of "
                        "what to check for, not what to do."
                    ),
                    "agent_type": "codex",
                }
            ],
        )
        risks = _MOD.lint(roadmap)
        assert [r for r in risks if r.violation_class == "BLOCKED_PATTERN_MATCHED"] == []

    def test_non_audit_task_with_same_phrase_still_blocked(self, isolated_project: Path) -> None:
        """Regression guard: the scope-audit exemption must be narrow -- a task that is
        NOT a source-scope-audit but happens to contain the same blocked phrase is still
        caught."""
        roadmap = _write_roadmap(
            isolated_project,
            [
                {
                    "id": "exp5136-totally-different-task",
                    "title": "PHASE B1 something else entirely",
                    "prompt": "CONTEXT: this task will exercise the synthetic blocked phrase here.",
                    "agent_type": "codex",
                }
            ],
        )
        risks = _MOD.lint(roadmap)
        assert len([r for r in risks if r.violation_class == "BLOCKED_PATTERN_MATCHED"]) == 1


class TestWrongMechanismNegationAware:
    """2026-07-01: CLASS 4 (WRONG_MECHANISM_PRECONDITION) got the same class of false
    positive as BLOCKED_PATTERN_MATCHED -- caught on `.471`'s exp5144, whose prompt says
    "Do not touch host /dev/mmcblk* for KV260; use SSH to the board." (textbook-correct
    per CLAUDE.md "KV260 SSH-Not-SD-Card Discipline") but got HARD-blocked anyway: the
    joint board+path regex has no negation awareness. Fixed via _is_negated_context, a
    tight character-window check for a negation marker just before the match."""

    def test_correctly_negated_kv260_mmcblk_is_not_blocked(self, isolated_project: Path) -> None:
        roadmap = _write_roadmap(
            isolated_project,
            [
                {
                    "id": "exp5144-authenticated-board-workload-v471",
                    "title": "PHASE C2 hardware continuity",
                    "prompt": (
                        "CONTEXT: convert board reachability into authenticated workload "
                        "transcripts. Do not touch host /dev/mmcblk* for KV260; use SSH "
                        "to the board."
                    ),
                    "agent_type": "codex",
                }
            ],
        )
        risks = _MOD.lint(roadmap)
        assert [r for r in risks if r.violation_class == "WRONG_MECHANISM_PRECONDITION"] == []

    def test_actual_wrong_mechanism_usage_still_blocked(self, isolated_project: Path) -> None:
        """Regression guard: a task that ACTUALLY instructs the agent to use the
        retired /dev/mmcblk precondition (no negation) must still be caught."""
        roadmap = _write_roadmap(
            isolated_project,
            [
                {
                    "id": "exp5199-genuinely-wrong-mechanism",
                    "title": "PHASE Q KV260 precondition check",
                    "prompt": "CONTEXT: check board readiness via ls /dev/mmcblk* on the host before proceeding.",
                    "agent_type": "codex",
                }
            ],
        )
        risks = _MOD.lint(roadmap)
        assert len([r for r in risks if r.violation_class == "WRONG_MECHANISM_PRECONDITION"]) == 1

    def test_negation_far_from_match_does_not_suppress_violation(
        self, isolated_project: Path
    ) -> None:
        """Regression guard: a negation word elsewhere in a LONG prompt, well outside
        the tight character window before the match, must not suppress a genuine
        violation -- the exemption is narrow by design."""
        filler = "x" * 200
        roadmap = _write_roadmap(
            isolated_project,
            [
                {
                    "id": "exp5198-unrelated-negation-far-away",
                    "title": "PHASE Q KV260 precondition check",
                    "prompt": (
                        f"CONTEXT: do not skip any steps. {filler} "
                        "Check board readiness via ls /dev/mmcblk* on the host."
                    ),
                    "agent_type": "codex",
                }
            ],
        )
        risks = _MOD.lint(roadmap)
        assert len([r for r in risks if r.violation_class == "WRONG_MECHANISM_PRECONDITION"]) == 1


class TestScopeMatchedPriorFailureProseAutoDowngrade:
    """2026-07-03: `.474` REFUSED activation for ~45 minutes -- 7 tasks scope-matched
    real prior artifacts but had no structured `prior_failures:` block, even though
    EVERY task's own prompt already explained why it wasn't a doomed rerun (an
    outer-loop session had to read each prompt and hand-add the structured field).
    This is a mechanical safety net for that exact recurrence: when a task's prompt
    names a matched prior's id verbatim with nearby differentiation language, CLASS 3
    auto-downgrades HARD -> WARNING (never a full clear) instead of blocking activation
    for a reasoning gap that was already visible in the prose."""

    @pytest.fixture
    def project_with_prior_failure(self, isolated_project: Path) -> tuple[Path, str]:
        """Seeds results/ with one BLOCKED prior artifact whose scope
        ("widget-repair-pipeline") will overlap a same-scope draft task."""
        results_dir = isolated_project / "results"
        artifact = {
            "experiment": "experiment_9001_widget_repair_pipeline_v1",
            "title": "Widget Repair Pipeline v1",
            "honest_verdict": "blocked_widget_repair_pipeline_precondition_missing",
        }
        (results_dir / "experiment_9001_widget_repair_pipeline_v1.json").write_text(
            json.dumps(artifact)
        )
        return isolated_project, "exp9001-widget-repair-pipeline-v1"

    def test_no_prose_and_no_structured_block_is_still_hard_blocked(
        self, project_with_prior_failure: tuple[Path, str]
    ) -> None:
        """Baseline: without the fix, this shape is exactly what makes `.474`
        HARD-block -- scope-matched, no prior_failures:, prompt doesn't even
        mention the prior. Confirms the fixture actually triggers CLASS 3."""
        tmp_path, prior_id = project_with_prior_failure
        roadmap = _write_roadmap(
            tmp_path,
            [
                {
                    "id": "exp9002-widget-repair-pipeline-v2",
                    "title": "PHASE Q widget repair pipeline v2",
                    "prompt": "CONTEXT: scale the widget repair pipeline to n=30.",
                    "agent_type": "codex",
                }
            ],
        )
        risks = _MOD.lint(roadmap)
        matched = [r for r in risks if r.violation_class == "SCOPE_MATCHED_PRIOR_FAILURE"]
        assert len(matched) == 1
        assert matched[0].severity == "HARD"

    def test_prose_naming_prior_with_differentiation_language_auto_downgrades(
        self, project_with_prior_failure: tuple[Path, str]
    ) -> None:
        """The fix: prompt names the matched prior id verbatim with nearby
        differentiation language ('hardens' / 'was blocked') -- downgrades to
        WARNING, and the detail carries a visible AUTO-DOWNGRADED audit marker."""
        tmp_path, prior_id = project_with_prior_failure
        roadmap = _write_roadmap(
            tmp_path,
            [
                {
                    "id": "exp9002-widget-repair-pipeline-v2",
                    "title": "PHASE Q widget repair pipeline v2",
                    "prompt": (
                        f"CONTEXT: {prior_id} was blocked on a missing precondition, not a "
                        "methodology failure. This task hardens that same pipeline now that "
                        "the precondition is fixed, scaling to n=30."
                    ),
                    "agent_type": "codex",
                }
            ],
        )
        risks = _MOD.lint(roadmap)
        matched = [r for r in risks if r.violation_class == "SCOPE_MATCHED_PRIOR_FAILURE"]
        assert len(matched) == 1
        assert matched[0].severity == "WARNING"
        assert "AUTO-DOWNGRADED" in matched[0].detail

    def test_prose_naming_prior_without_differentiation_language_stays_hard(
        self, project_with_prior_failure: tuple[Path, str]
    ) -> None:
        """Regression guard: merely MENTIONING the prior id (e.g. citing it as
        related background) without any differentiation language must NOT
        auto-downgrade -- the marker phrase is load-bearing, not just the id."""
        tmp_path, prior_id = project_with_prior_failure
        roadmap = _write_roadmap(
            tmp_path,
            [
                {
                    "id": "exp9002-widget-repair-pipeline-v2",
                    "title": "PHASE Q widget repair pipeline v2",
                    "prompt": f"CONTEXT: see also {prior_id} for related background. Scale to n=30.",
                    "agent_type": "codex",
                }
            ],
        )
        risks = _MOD.lint(roadmap)
        matched = [r for r in risks if r.violation_class == "SCOPE_MATCHED_PRIOR_FAILURE"]
        assert len(matched) == 1
        assert matched[0].severity == "HARD"

    def test_valid_structured_prior_failures_block_still_takes_priority(
        self, project_with_prior_failure: tuple[Path, str]
    ) -> None:
        """A genuine, valid structured prior_failures: block must still fully clear
        the check as before -- the auto-downgrade is a fallback, not a replacement."""
        tmp_path, prior_id = project_with_prior_failure
        roadmap = _write_roadmap(
            tmp_path,
            [
                {
                    "id": "exp9002-widget-repair-pipeline-v2",
                    "title": "PHASE Q widget repair pipeline v2",
                    "prompt": "CONTEXT: scale the widget repair pipeline to n=30.",
                    "prior_failures": [
                        {
                            "experiment_id": prior_id,
                            "verdict": "blocked_widget_repair_pipeline_precondition_missing",
                            "addressed_by": "Precondition fixed; this scales the same pipeline.",
                            "retire_if_same_verdict": False,
                        }
                    ],
                    "agent_type": "codex",
                }
            ],
        )
        risks = _MOD.lint(roadmap)
        assert [r for r in risks if r.violation_class == "SCOPE_MATCHED_PRIOR_FAILURE"] == []
