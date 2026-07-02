"""Tests for the in-process doc reconciler.

The reconciler replaces the Haiku doc-reconciliation Claude Code call
in the conductor with a mechanical Python pass that reads the artifact
JSON, maps the honest_verdict to a status label, and appends to
ops/changelog.md (always), _bmad/traceability.md (when new
REQ-*/SCENARIO-* are added in the most recent commit), and
ops/status.md (only when the experiment is a clear win with new REQ-*).

These tests cover the mapping table verbatim against the Haiku prompt
rules, the artifact-finding logic, and a full end-to-end reconcile()
invocation against a temporary repo skeleton — there is no LLM call
to mock, only file IO and a git-show subprocess that we mock at the
boundary.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest import mock

import pytest

# The reconciler lives under scripts/, not python/carnot/, because it is
# tightly coupled to the conductor and not a reusable library function.
SCRIPTS_DIR = Path(__file__).resolve().parent.parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from in_process_doc_reconcile import (  # noqa: E402
    build_changelog_entry,
    extract_key_metric,
    extract_new_req_scenario_ids,
    find_artifact,
    map_status_label,
    reconcile,
)


# ---------------------------------------------------------------------------
# map_status_label — verbatim against the Haiku prompt's mapping table
# (scripts/research_conductor.py:2280-2295)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "verdict, expected",
    [
        # Wins — single token from the win list
        ("retro_028_closed", "✅ Complete"),
        ("injection_field_fixed", "✅ Complete"),
        ("vg_search_effective", "✅ Complete"),
        ("hf_models_published", "✅ Complete"),
        ("retrieval_auc_exceeds_target", "✅ Complete"),
        # Partial / research-finding tokens
        ("retro_061_partial", "⚠️ Research Finding"),
        ("constraint_addition_no_delta_live", "⚠️ Research Finding"),
        ("arbiter_still_wrong", "⚠️ Research Finding"),
        ("jepa_v22_below_random", "⚠️ Research Finding"),
        ("loss_redesign_partial", "⚠️ Research Finding"),
        ("tier1_plateau_persists", "⚠️ Research Finding"),
        ("injection_negative_delta", "⚠️ Research Finding"),
        # Blocked — gate / prereq / tool unavailable
        ("blocked_no_delta", "⚠️ Blocked"),
        ("blocked_gate", "⚠️ Blocked"),
        ("tools_not_installed", "⚠️ Blocked"),
        ("tools_unavailable", "⚠️ Blocked"),
        ("blocked_model_load_failed", "⚠️ Blocked"),
        # Failed — exception or timeout
        ("timed_out", "❌ Failed"),
        ("exception", "❌ Failed"),
        ("test_suite_failed", "❌ Failed"),
        # Default — unrecognised verdict falls to "Research Finding", not "Complete"
        ("frobnicated_the_widget", "⚠️ Research Finding"),
        ("", "⚠️ Research Finding"),
    ],
)
def test_map_status_label_matches_haiku_table(verdict, expected):
    """Each verdict maps to the same label the Haiku prompt would assign.

    Tracks scripts/research_conductor.py:2280-2295. If the Haiku prompt's
    mapping table changes, this test must be updated in lockstep so the
    in-process path doesn't drift from the LLM-based path.
    """
    assert map_status_label(verdict) == expected


def test_map_status_label_blocked_beats_failed_when_both_match():
    """A verdict like 'blocked_model_load_failed' is Blocked, not Failed.

    Both tokens appear in the string, but structurally the verdict says
    "blocked, because the model failed to load" — the block is the
    determining state. Haiku's actual classification on Exp 811 (.62)
    confirmed this; the in-process path must match.
    """
    assert map_status_label("blocked_model_load_failed") == "⚠️ Blocked"


def test_map_status_label_failed_when_no_block_token():
    """A pure failure verdict with no block token maps to ❌ Failed."""
    assert map_status_label("gpu_load_failed") == "❌ Failed"


def test_map_status_label_no_false_positive_on_required():
    """The bare token 'required' must not flip 'fr11_required_pass' to Blocked.

    Documents the carve-out in _BLOCKED_TOKENS where 'required' is
    deliberately omitted. A verdict like 'fr11_required_pass' contains
    'required' as a substring but is in fact a win, so the mapping must
    reach the win category via the 'pass'... actually 'pass' isn't in
    the win list either, so this falls to ⚠️ Research Finding (the safe
    default). The point this test makes is the *negative* one: the token
    'required' alone is not enough to classify as Blocked.
    """
    assert map_status_label("fr11_required_pass") != "⚠️ Blocked"


# ---------------------------------------------------------------------------
# find_artifact
# ---------------------------------------------------------------------------


def test_find_artifact_extracts_experiment_number(tmp_path):
    """Task ids of the shape 'expNNN-...' resolve to results/experiment_NNN_*.json."""
    results = tmp_path / "results"
    results.mkdir()
    (results / "experiment_819_injection_field_fix.json").write_text("{}")
    (results / "experiment_820_some_other.json").write_text("{}")

    result = find_artifact("exp819-injection-field-fix", results)
    assert result is not None
    assert result.name == "experiment_819_injection_field_fix.json"


def test_find_artifact_returns_none_for_unknown_task(tmp_path):
    """A task id that doesn't match the expNNN pattern returns None.

    Defensive: don't guess; let the caller log and fall back to Haiku.
    """
    (tmp_path / "results").mkdir()
    assert find_artifact("not-an-experiment-id", tmp_path / "results") is None


def test_find_artifact_picks_most_recent_when_multiple_match(tmp_path):
    """When multiple files match, return the most recently modified.

    This handles retries where a partially-written artifact from an
    earlier attempt sits next to the final one.
    """
    results = tmp_path / "results"
    results.mkdir()
    older = results / "experiment_819_v1.json"
    older.write_text("{}")
    import os
    import time

    time.sleep(0.01)  # ensure mtime difference
    newer = results / "experiment_819_v2.json"
    newer.write_text("{}")

    result = find_artifact("exp819", results)
    assert result == newer
    assert result is not None  # for type checker


# ---------------------------------------------------------------------------
# extract_key_metric — picks first matching scalar from a fixed priority list
# ---------------------------------------------------------------------------


def test_extract_key_metric_prefers_auc_over_other_metrics():
    """When several metrics are present, AUC wins (it appears first in the list)."""
    artifact = {"auc": 0.95, "accuracy": 0.7, "n_repaired": 14}
    assert extract_key_metric(artifact) == "auc=0.95"


def test_extract_key_metric_skips_none_and_collections():
    """Nones and lists/dicts don't fit on a one-line changelog entry."""
    artifact = {
        "auc": None,
        "accuracy": [0.5, 0.6],  # list — skipped
        "details": {"k": "v"},  # dict — skipped
        "delta_overall": 0.0,  # scalar — kept
    }
    assert extract_key_metric(artifact) == "delta_overall=0.0"


def test_extract_key_metric_returns_empty_when_no_known_field():
    """A metric-free artifact yields the empty string, not a default value."""
    assert extract_key_metric({"some_unrelated_field": 42}) == ""


# ---------------------------------------------------------------------------
# build_changelog_entry — format matches project convention
# ---------------------------------------------------------------------------


def test_build_changelog_entry_format():
    """The one-line entry has the expected shape and contains all the parts."""
    artifact = {"honest_verdict": "injection_field_fixed", "discrimination_rate": 1.0}
    line = build_changelog_entry(
        artifact=artifact,
        task_title="Exp 819: IsingEBM External Field Fix",
        status_label="✅ Complete",
        artifact_relpath="results/experiment_819_injection_field_fix.json",
        today_iso="2026-04-24",
    )
    # Begins with date and bullet
    assert line.startswith("- 2026-04-24:")
    # Contains the task title and status emoji
    assert "Exp 819: IsingEBM External Field Fix" in line
    assert "✅ Complete" in line
    # Contains the verdict verbatim
    assert "honest_verdict=injection_field_fixed" in line
    # Contains the metric and the artifact path
    assert "discrimination_rate=1.0" in line
    assert "results/experiment_819_injection_field_fix.json" in line
    # Ends with newline so successive appends don't run together
    assert line.endswith("\n")


# ---------------------------------------------------------------------------
# extract_new_req_scenario_ids — parse git show output for new spec rows
# ---------------------------------------------------------------------------


def test_extract_new_req_scenario_ids_parses_added_lines(tmp_path, monkeypatch):
    """REQ and SCENARIO ids added to spec.md files are extracted.

    Mocks the git subprocess so the test doesn't depend on an actual
    repo state.
    """
    fake_diff = (
        "diff --git a/openspec/capabilities/foo/spec.md b/openspec/capabilities/foo/spec.md\n"
        "@@ -10,0 +11,3 @@\n"
        "+- **REQ-VERIFY-095**: a new requirement\n"
        "+- **REQ-VERIFY-096**: another requirement\n"
        "+- **SCENARIO-VERIFY-129**: a new scenario\n"
        "diff --git a/python/carnot/foo.py b/python/carnot/foo.py\n"  # not a spec
        "@@ -0,0 +1 @@\n"
        "+- **REQ-NOT-INCLUDED**: this is not in a spec.md\n"
    )
    completed = mock.MagicMock(stdout=fake_diff)
    with mock.patch("in_process_doc_reconcile.subprocess.run", return_value=completed):
        reqs, scenarios = extract_new_req_scenario_ids(tmp_path)
    assert reqs == ["REQ-VERIFY-095", "REQ-VERIFY-096"]
    assert scenarios == ["SCENARIO-VERIFY-129"]


def test_extract_new_req_scenario_ids_dedupes_within_commit(monkeypatch, tmp_path):
    """If the same id appears twice in the diff, it's reported once."""
    fake_diff = (
        "diff --git a/openspec/capabilities/foo/spec.md b/openspec/capabilities/foo/spec.md\n"
        "+- **REQ-DUP-001**: first mention\n"
        "+- **REQ-DUP-001**: same id repeated\n"
    )
    completed = mock.MagicMock(stdout=fake_diff)
    with mock.patch("in_process_doc_reconcile.subprocess.run", return_value=completed):
        reqs, _ = extract_new_req_scenario_ids(tmp_path)
    assert reqs == ["REQ-DUP-001"]


def test_extract_new_req_scenario_ids_returns_empty_when_git_fails(tmp_path):
    """A git failure yields an empty list rather than raising.

    Conductor will then skip the optional traceability/status.md updates
    and rely solely on the always-on changelog append. Defensive.
    """
    import subprocess

    with mock.patch(
        "in_process_doc_reconcile.subprocess.run",
        side_effect=subprocess.SubprocessError("git not available"),
    ):
        reqs, scenarios = extract_new_req_scenario_ids(tmp_path)
    assert reqs == []
    assert scenarios == []


# ---------------------------------------------------------------------------
# reconcile — end-to-end against a temporary repo skeleton
# ---------------------------------------------------------------------------


def _make_repo_skeleton(tmp_path: Path) -> Path:
    """Build a minimal repo layout the reconciler can write into."""
    (tmp_path / "results").mkdir()
    (tmp_path / "ops").mkdir()
    (tmp_path / "_bmad").mkdir()
    (tmp_path / "ops" / "changelog.md").write_text("# Changelog\n\nExisting line.\n")
    (tmp_path / "ops" / "status.md").write_text(
        "# Status\n\n| Date | Task | Status | Verdict | Artifact |\n"
        "|------|------|--------|---------|----------|\n"
        "| 2026-04-24 | seed | ✅ Complete | seed | n/a |\n"
    )
    (tmp_path / "_bmad" / "traceability.md").write_text(
        "# Traceability\n\n| ID | Source | Status | Artifact |\n"
        "|----|--------|--------|----------|\n"
    )
    return tmp_path


def test_reconcile_appends_changelog_for_clear_win(tmp_path):
    """A win verdict with no new REQ/SCENARIO: changelog only."""
    repo = _make_repo_skeleton(tmp_path)
    artifact_path = repo / "results" / "experiment_900_demo.json"
    artifact_path.write_text(
        json.dumps(
            {
                "honest_verdict": "demo_complete",
                "auc": 0.92,
            }
        )
    )
    task = {"id": "exp900-demo", "title": "Exp 900: Demo Win"}
    with mock.patch(
        "in_process_doc_reconcile.subprocess.run",
        return_value=mock.MagicMock(stdout=""),
    ):
        result = reconcile(task, repo_root=repo, today="2026-04-24")
    assert result.changelog_appended is True
    assert result.status_appended is False
    assert result.traceability_rows_added == 0
    assert result.status_label == "✅ Complete"
    changelog = (repo / "ops" / "changelog.md").read_text()
    assert "Exp 900: Demo Win" in changelog
    assert "honest_verdict=demo_complete" in changelog
    assert "auc=0.92" in changelog


def test_reconcile_appends_status_and_traceability_when_new_reqs_added(tmp_path):
    """A win with new REQ-* gets all three updates: changelog + status + traceability."""
    repo = _make_repo_skeleton(tmp_path)
    artifact_path = repo / "results" / "experiment_901_capability.json"
    artifact_path.write_text(
        json.dumps(
            {
                "honest_verdict": "capability_ready",
            }
        )
    )
    task = {"id": "exp901-capability", "title": "Exp 901: New Capability"}
    fake_diff = (
        "diff --git a/openspec/capabilities/foo/spec.md b/openspec/capabilities/foo/spec.md\n"
        "+- **REQ-CAP-001**: new req\n"
        "+- **SCENARIO-CAP-001**: new scenario\n"
    )
    with mock.patch(
        "in_process_doc_reconcile.subprocess.run",
        return_value=mock.MagicMock(stdout=fake_diff),
    ):
        result = reconcile(task, repo_root=repo, today="2026-04-24")
    assert result.changelog_appended is True
    assert result.status_appended is True
    assert result.traceability_rows_added == 2
    trace = (repo / "_bmad" / "traceability.md").read_text()
    assert "REQ-CAP-001" in trace
    assert "SCENARIO-CAP-001" in trace
    assert "Implemented" in trace.split("REQ-CAP-001")[1].splitlines()[0]
    status = (repo / "ops" / "status.md").read_text()
    assert "Exp 901: New Capability" in status


def test_reconcile_does_not_promote_partial_to_implemented(tmp_path):
    """A partial verdict with new REQ-*: traceability rows are 'Implemented-Partial'."""
    repo = _make_repo_skeleton(tmp_path)
    artifact_path = repo / "results" / "experiment_902_partial.json"
    artifact_path.write_text(
        json.dumps(
            {
                "honest_verdict": "constraint_addition_no_delta_live",
            }
        )
    )
    task = {"id": "exp902-partial", "title": "Exp 902: Partial"}
    fake_diff = (
        "diff --git a/openspec/capabilities/foo/spec.md b/openspec/capabilities/foo/spec.md\n"
        "+- **REQ-PART-001**: new req\n"
    )
    with mock.patch(
        "in_process_doc_reconcile.subprocess.run",
        return_value=mock.MagicMock(stdout=fake_diff),
    ):
        result = reconcile(task, repo_root=repo, today="2026-04-24")
    assert result.status_label == "⚠️ Research Finding"
    # status.md is NOT appended — partial experiments are not delivered capabilities
    assert result.status_appended is False
    trace = (repo / "_bmad" / "traceability.md").read_text()
    # Traceability row is Implemented-Partial, never Implemented
    assert "Implemented-Partial" in trace
    assert "| REQ-PART-001 | Exp 902: Partial | Implemented |" not in trace


def test_reconcile_skips_when_artifact_missing(tmp_path):
    """No artifact = early return with skipped_reason set, no file changes."""
    repo = _make_repo_skeleton(tmp_path)
    task = {"id": "exp999-not-found", "title": "Exp 999: Missing"}
    result = reconcile(task, repo_root=repo, today="2026-04-24")
    assert result.changelog_appended is False
    assert result.skipped_reason is not None
    assert "no artifact" in result.skipped_reason
    # Nothing was written
    changelog = (repo / "ops" / "changelog.md").read_text()
    assert "Exp 999" not in changelog


def test_reconcile_skips_when_artifact_unreadable(tmp_path):
    """Corrupt JSON in the artifact is reported as unreadable, no crash."""
    repo = _make_repo_skeleton(tmp_path)
    (repo / "results" / "experiment_998_corrupt.json").write_text("{not json")
    task = {"id": "exp998-corrupt", "title": "Exp 998"}
    result = reconcile(task, repo_root=repo, today="2026-04-24")
    assert result.changelog_appended is False
    assert result.skipped_reason is not None
    assert "unreadable" in result.skipped_reason


# ---------------------------------------------------------------------------
# Newly added (.68) — win-token additions and classify_artifact
# ---------------------------------------------------------------------------


from in_process_doc_reconcile import classify_artifact  # noqa: E402


@pytest.mark.parametrize(
    "verdict",
    [
        "tier1_relay_works_live",  # .65 Exp 848
        "gguf_cache_implemented",  # .65 Exp 849
        "deployed",  # .66 Exp 856
        "streaming_cot_wired",  # .67 Exp 874
        "fr11_self_learning_confirmed",  # .66 Exp 862 (was already a win — sanity)
    ],
)
def test_added_win_tokens_now_classify_as_complete(verdict):
    """Verdicts that were under-counted in .65/.66/.67 now map to ✅ Complete.

    These are the specific historical cases where the in-process reconciler
    labeled real wins as ⚠️ Research Finding because the relevant token
    was missing from `_WIN_TOKENS`. The fix added `wired`, `implemented`,
    `works`, `deployed` to the list. This test pins those classifications.
    """
    assert map_status_label(verdict) == "✅ Complete"


@pytest.mark.parametrize(
    "verdict",
    [
        # exp1791 (2026-05): a clean, unambiguous win that fell all the way through
        # to the ⚠️ Research Finding default because "success" was never in
        # _WIN_TOKENS -- despite correctly following CLAUDE.md's Verdict
        # Terminal-Prefix Discipline. This misclassification became a phantom
        # "prior failure" in FailureLedger, HARD-blocking .471's exp5140
        # (a legitimate continuation of a genuine v470 success) as a doomed rerun.
        "success: RTL structural logic validated theoretically and mapped to Verilog/TCL specification for Symbolic-KAN.",
        "success_verified_symbolic_gating",  # exp2071, same bug, underscore form
        "shipped_minimal_repair_pipeline_v5",  # CLAUDE.md's own documented example verdict
        "passed_qwen3.6_logprob_telemetry_topk_available",  # CLAUDE.md's own documented example verdict
    ],
)
def test_terminal_prefix_win_tokens_now_classify_as_complete(verdict):
    """2026-07-01: `success`, `succeeded`, `shipped`, `passed` were missing from
    `_WIN_TOKENS` even though CLAUDE.md's Verdict Terminal-Prefix Discipline
    mandates exactly these four prefix families (complete:/success:/passed:/
    shipped:). Corpus-wide scan at fix time found 352 "success:"-prefixed and 13
    "shipped:"-prefixed artifacts (of 4160 total) misclassified this way -- all
    verified to have ZERO genuine blocked/failed/partial content (pure oversight,
    not a real ambiguity). Verified additive-safe: 0 artifacts flipped AWAY from
    Complete; the blocked/failed/partial checks still run before the win-token
    check, so a genuinely mixed verdict like
    "complete_..._weak_fit_..._hardware_leg_blocked_..." (a real Carnot artifact,
    exp_kv260_residual_energy_decay_exponent.py) is correctly UNCHANGED by this
    fix -- see test_mixed_terminal_verdict_still_not_complete below.
    """
    assert map_status_label(verdict) == "✅ Complete"


def test_mixed_terminal_verdict_still_not_complete():
    """Regression guard: a verdict with a correct terminal prefix but genuinely
    mixed/negative content must NOT be promoted to Complete by the
    success/shipped/passed addition -- blocked/failed/partial tokens in the
    descriptive tail still win, exactly as before. Real Carnot verdict shape
    (kv260_residual_energy_decay_exponent, a genuine weak-fit + hardware-blocked
    result)."""
    verdict = (
        "complete_kv260_residual_energy_methodology_validated_cpu_weak_fit_r2_0.39_"
        "hardware_leg_blocked_no_runtime_sweep_control"
    )
    assert map_status_label(verdict) == "⚠️ Blocked"


def test_classify_artifact_promotes_on_retro_closed_field():
    """An artifact with retro_*_closed populated is promoted to ✅ Complete
    even when the verdict text alone wouldn't carry a win-token.

    This is the classify_artifact extension over plain map_status_label —
    a closed retro is a concrete deliverable; the experiment is not just
    a "research finding" if it actually closed something.
    """
    artifact = {
        "honest_verdict": "some_neutral_phrase_with_no_tokens",
        "retro_constraint_zero_delta_closed": True,
    }
    assert classify_artifact(artifact) == "✅ Complete"


def test_classify_artifact_does_not_promote_on_failed_verdict():
    """A failed verdict beats any retro_*_closed signal.

    Failed-with-also-some-retro-closed is a contradictory shape —
    discipline-conservative interpretation: the failure wins.
    """
    artifact = {
        "honest_verdict": "timed_out",
        "retro_some_thing_closed": "RETRO-CLAIMED-CLOSURE",
    }
    assert classify_artifact(artifact) == "❌ Failed"


def test_classify_artifact_does_not_promote_when_retro_value_is_falsy():
    """retro_*_closed: False / "" / "false" should not trigger a promotion."""
    for falsy in (False, "", "false", 0, None):
        artifact = {
            "honest_verdict": "neutral_text",
            "retro_x_closed": falsy,
        }
        assert classify_artifact(artifact) == "⚠️ Research Finding"


def test_classify_artifact_promotes_on_retro_value_with_retro_tag_string():
    """A value like 'RETRO-GGUF-CACHE-IMPORT' counts as a real closure.

    Mirrors the .65 Exp 849 artifact where retro_closed had a string
    naming the retro that was closed. Common pattern.
    """
    artifact = {
        "honest_verdict": "gguf_cache_implemented",
        "retro_closed": "RETRO-GGUF-CACHE-IMPORT",
    }
    assert classify_artifact(artifact) == "✅ Complete"


def test_classify_artifact_unchanged_when_no_retro_field():
    """Without any retro_*_closed field, behaviour matches map_status_label."""
    artifact = {"honest_verdict": "some_neutral_phrase"}
    assert classify_artifact(artifact) == "⚠️ Research Finding"
    artifact = {"honest_verdict": "complete"}
    assert classify_artifact(artifact) == "✅ Complete"


def test_classify_artifact_ignores_non_retro_closed_fields():
    """A field like 'gate_closed' or 'window_closed' should not trigger
    the promotion — only `retro_*_closed` is the structural signal.
    """
    artifact = {
        "honest_verdict": "neutral_phrase",
        "gate_closed": True,  # not a retro
        "window_closed": "yes",  # not a retro
    }
    assert classify_artifact(artifact) == "⚠️ Research Finding"
