"""Tests for the failed-experiment rerun-discipline mechanical enforcement.

The CLAUDE.md "Failed-Experiment Rerun Discipline" rule binds the
planner at policy level. The .66/.67/.68 planner runs confirmed the
policy alone doesn't transfer to YAML structure (0 of 12 tasks ever
populate `prior_failures:` despite the rule). This module is the
mechanical safety net — the conductor calls into it before each Sonnet
spawn and refuses to launch tasks whose scope matches a prior failure
without an adequate `prior_failures:` entry.

Tests cover:
  - Scope-signature extraction (version-suffix stripping, slug
    canonicalisation)
  - Scope-overlap matching (conservative — false positives are more
    expensive than false negatives)
  - LedgerEntry construction from artifacts (skips win-verdict
    artifacts, handles malformed JSON robustly)
  - Validation of `prior_failures:` field (four-part discipline)
  - End-to-end is_doomed_rerun on synthetic seed data that mirrors
    .65–.68 patterns
Spec: REQ-INFRA-067
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parent.parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from failure_ledger import (  # noqa: E402
    FailureLedger,
    LedgerEntry,
    _scope_signature,
    _scopes_overlap,
    validate_prior_failures,
)

# ---------------------------------------------------------------------------
# Scope signature extraction
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "task_id, expected",
    [
        # Bare slug — version stripped
        ("exp870-sota-code-repair-v7", "sota-code-repair"),
        ("exp881-code-repair-v8-gemma4", "code-repair-v8-gemma4"),  # mid-version stripped
        ("exp857-sota-code-repair-v6", "sota-code-repair"),
        # No version suffix
        ("exp819-injection-field-fix", "injection-field-fix"),
        # Version with trailing tokens
        ("exp872-jepa-v25-dg-prm", "jepa-v25-dg-prm"),  # not a clean v\d+$ at end
    ],
)
def test_scope_signature_extraction(task_id, expected):
    """Scope signatures strip trailing version suffixes for stable comparison."""
    assert _scope_signature(task_id) == expected


def test_scope_signature_handles_title_format():
    """Title-style strings (`Exp 870: foo bar`) also work."""
    assert _scope_signature("Exp 870: SOTA Code Repair v7") == "sota-code-repair"


# ---------------------------------------------------------------------------
# Scope overlap (the matching primitive)
# ---------------------------------------------------------------------------


def test_scopes_overlap_long_substring_matches():
    """`code-repair` (11 chars) is long enough to declare two scopes equivalent."""
    assert _scopes_overlap("sota-code-repair", "code-repair-v8-gemma4")


def test_scopes_overlap_too_short_substring_does_not_match():
    """A single token like `ising` (5 chars) is NOT enough on its own —
    the conservative bias means we'd rather miss a match than create
    a false positive."""
    assert not _scopes_overlap("ising", "ising")  # both <8 chars


def test_scopes_overlap_unrelated_scopes_dont_match():
    """`code-repair` and `live-benchmark` should not be flagged as
    related — they share no >=8-char substring.
    """
    assert not _scopes_overlap("sota-code-repair", "live-benchmark")
    assert not _scopes_overlap("jepa-ood", "ice40-bitstream")


def test_scopes_overlap_handles_empty_inputs():
    """Empty or None scopes should never match anything."""
    assert not _scopes_overlap("", "code-repair")
    assert not _scopes_overlap("code-repair", "")


# ---------------------------------------------------------------------------
# Loading a ledger from artifacts
# ---------------------------------------------------------------------------


def _write_artifact(
    results_dir: Path, exp_num: int, slug: str, verdict: str, extra: dict | None = None
) -> Path:
    """Helper: write a minimal artifact file and return its path."""
    results_dir.mkdir(parents=True, exist_ok=True)
    target = results_dir / f"experiment_{exp_num}_{slug}.json"
    data = {
        "experiment": exp_num,
        "title": f"Exp {exp_num}: {slug}",
        "honest_verdict": verdict,
    }
    if extra:
        data.update(extra)
    target.write_text(json.dumps(data))
    return target


def test_load_skips_winning_artifacts(tmp_path):
    """Artifacts with ✅-Complete verdicts are NOT in the ledger."""
    _write_artifact(tmp_path / "results", 819, "ising_field", "injection_field_fixed")
    _write_artifact(tmp_path / "results", 820, "gguf", "import_fixed_repair_positive")
    ledger = FailureLedger.load_from_artifacts(tmp_path)
    assert len(ledger.entries) == 0


def test_load_includes_failed_blocked_partial_verdicts(tmp_path):
    """All non-win verdicts become ledger entries.

    Verdict tokens used here are picked to actually classify into the
    three different categories per the in-process reconciler's mapping
    (scripts/in_process_doc_reconcile.py). Earlier draft incorrectly
    expected `model_not_cached` to map to ⚠️ Blocked — that token
    isn't in `_BLOCKED_TOKENS`, so it actually defaults to
    ⚠️ Research Finding. Fixed by using `blocked_model_load_failed`
    here (matches `blocked` in the blocked-tokens list).
    """
    _write_artifact(
        tmp_path / "results", 850, "code_repair_v5", "blocked_model_load_failed"
    )  # ⚠ Blocked
    _write_artifact(
        tmp_path / "results", 858, "live_benchmark_v5", "simulation_fallback"
    )  # ⚠ Research Finding (default)
    _write_artifact(tmp_path / "results", 869, "gguf_predownload", "download_failed")  # ❌ Failed
    ledger = FailureLedger.load_from_artifacts(tmp_path)
    assert len(ledger.entries) == 3
    labels = sorted(e.status_label for e in ledger.entries)
    assert "❌ Failed" in labels
    assert "⚠️ Blocked" in labels
    assert "⚠️ Research Finding" in labels


def test_load_handles_malformed_artifacts(tmp_path):
    """Corrupt JSON or missing fields don't crash the loader."""
    results = tmp_path / "results"
    results.mkdir()
    (results / "experiment_999_corrupt.json").write_text("{not json")
    _write_artifact(results, 998, "no_verdict", "")  # empty verdict
    _write_artifact(results, 850, "valid_failure", "model_not_cached")
    ledger = FailureLedger.load_from_artifacts(tmp_path)
    # Only the well-formed failure should be in the ledger
    assert len(ledger.entries) == 1
    assert ledger.entries[0].experiment_id == "exp850-valid-failure"


def test_load_returns_empty_ledger_when_no_results_dir(tmp_path):
    """Missing results/ directory → empty ledger, no error."""
    ledger = FailureLedger.load_from_artifacts(tmp_path)
    assert ledger.entries == []


def test_load_skips_top_level_list_artifacts(tmp_path):
    """Some early artifacts (e.g., experiment_798_cpmi_pairs_triples.json) are
    top-level JSON lists. The ledger must skip these without crashing."""
    results = tmp_path / "results"
    results.mkdir()
    (results / "experiment_798_pairs.json").write_text(json.dumps([{"x": 1}, {"x": 2}]))
    _write_artifact(results, 850, "valid_failure", "model_not_cached")
    ledger = FailureLedger.load_from_artifacts(tmp_path)
    assert len(ledger.entries) == 1
    assert ledger.entries[0].experiment_id == "exp850-valid-failure"


def test_load_coerces_dict_verdict_via_status_field(tmp_path):
    """Early artifacts (Exps 256/257/259/292/293/304/317) have a dict-shaped
    honest_verdict like {"status": "blocked", "explanation": "..."}.
    The ledger must coerce via the inner `status` field, not crash."""
    results = tmp_path / "results"
    results.mkdir()
    target = results / "experiment_293_legacy_dict.json"
    target.write_text(
        json.dumps(
            {
                "experiment": 293,
                "title": "Exp 293: legacy dict verdict",
                "honest_verdict": {"status": "blocked", "explanation": "no creds"},
            }
        )
    )
    ledger = FailureLedger.load_from_artifacts(tmp_path)
    assert len(ledger.entries) == 1
    assert ledger.entries[0].verdict == "blocked"
    assert ledger.entries[0].status_label == "⚠️ Blocked"


def test_load_skips_dict_verdict_with_no_status(tmp_path):
    """A dict-shaped honest_verdict with no `status` key has no failure
    signal we can reduce to a label — skip rather than guess."""
    results = tmp_path / "results"
    results.mkdir()
    target = results / "experiment_999_no_status.json"
    target.write_text(
        json.dumps(
            {
                "experiment": 999,
                "honest_verdict": {"explanation": "structured but no status"},
            }
        )
    )
    ledger = FailureLedger.load_from_artifacts(tmp_path)
    assert ledger.entries == []


def test_matching_priors_shortcircuits_on_preflight_scope():
    """Preflight is a structurally-recurring per-milestone scaffolding task.
    Even when the ledger has 5 prior preflight entries with non-✅ verdicts,
    a new preflight is NOT a doomed rerun — it is scheduled audit work.

    Regression test for the .71 first-fire false-positive on Exp 917
    (preflight v20) which blocked legitimate audit work because it scope-
    matched prior preflights with `preflight_v*_clean_manifest_pending`-
    style verdicts.
    """
    from failure_ledger import FailureLedger, LedgerEntry

    ledger = FailureLedger()
    for i, slug in enumerate(
        [
            "preflight-v2",
            "preflight-v9",
            "preflight-v10",
            "preflight-v11",
            "zombie-kill-preflight-v8",
        ]
    ):
        ledger.entries.append(
            LedgerEntry(
                experiment_id=f"exp{400 + i}-{slug}",
                title=f"Exp {400 + i}: {slug}",
                verdict="env_not_propagating",  # any non-✅ verdict
                status_label="⚠️ Research Finding",
                scope="preflight" if "zombie" not in slug else "zombie-kill-preflight",
            )
        )
    new_task = {"id": "exp917-preflight-v20", "title": "Exp 917: Pre-flight v20"}
    assert ledger.matching_priors(new_task) == []
    check = ledger.is_doomed_rerun(new_task)
    assert not check.blocked
    assert "no scope-matching priors" in check.reason


def test_matching_priors_shortcircuits_on_milestone_retro_scope():
    """Milestone retros are also structurally-recurring scaffolding."""
    from failure_ledger import FailureLedger, LedgerEntry

    ledger = FailureLedger()
    ledger.entries.append(
        LedgerEntry(
            experiment_id="exp891-milestone-retro",
            title="Exp 891: Milestone Retro",
            verdict="research_finding",
            status_label="⚠️ Research Finding",
            scope="milestone-retro",
        )
    )
    new_task = {"id": "exp928-milestone-retro-71", "title": "Exp 928: Milestone Retro 71"}
    assert ledger.matching_priors(new_task) == []


def test_recurring_scaffolding_does_not_break_legitimate_match():
    """The scaffolding short-circuit only applies when the *target* task is
    scaffolding. A non-scaffolding task should still match prior failures
    normally."""
    from failure_ledger import FailureLedger, LedgerEntry

    ledger = FailureLedger()
    ledger.entries.append(
        LedgerEntry(
            experiment_id="exp881-code-repair-v8",
            title="Exp 881: Code Repair v8",
            verdict="zero_constraints",
            status_label="❌ Failed",
            scope="code-repair",
        )
    )
    new_task = {"id": "exp895-code-repair-50q-scaleup", "title": "Exp 895: Code Repair 50q"}
    matches = ledger.matching_priors(new_task)
    assert len(matches) == 1
    assert matches[0].experiment_id == "exp881-code-repair-v8"


# ---------------------------------------------------------------------------
# matching_priors
# ---------------------------------------------------------------------------


def _ledger_with(*entries) -> FailureLedger:
    """Helper: construct a ledger from explicit LedgerEntry objects."""
    ledger = FailureLedger()
    ledger.entries.extend(entries)
    return ledger


def test_matching_priors_finds_scope_matches():
    """A new task whose scope substring-matches a prior failure's scope
    is identified as a doomed-rerun candidate."""
    ledger = _ledger_with(
        LedgerEntry(
            experiment_id="exp870-sota-code-repair-v7",
            title="SOTA Code Repair v7",
            verdict="blocked",
            status_label="⚠️ Blocked",
            scope="sota-code-repair",
        )
    )
    new_task = {
        "id": "exp881-code-repair-v8-gemma4",
        "title": "Exp 881: Code Repair v8",
    }
    priors = ledger.matching_priors(new_task)
    assert len(priors) == 1
    assert priors[0].experiment_id == "exp870-sota-code-repair-v7"


def test_matching_priors_excludes_self():
    """A task can't be its own prior failure (id-equality check)."""
    ledger = _ledger_with(
        LedgerEntry(
            experiment_id="exp870-sota-code-repair-v7",
            title="x",
            verdict="blocked",
            status_label="⚠️ Blocked",
            scope="sota-code-repair",
        )
    )
    same_task = {"id": "exp870-sota-code-repair-v7", "title": "x"}
    assert ledger.matching_priors(same_task) == []


def test_matching_priors_returns_empty_when_no_overlap():
    """An unrelated new task doesn't match prior failures."""
    ledger = _ledger_with(
        LedgerEntry(
            experiment_id="exp840-live-benchmark",
            title="x",
            verdict="simulation_fallback",
            status_label="⚠️ Research Finding",
            scope="live-benchmark",
        )
    )
    new_task = {"id": "exp819-ising-field-fix", "title": "Ising fix"}
    assert ledger.matching_priors(new_task) == []


# ---------------------------------------------------------------------------
# validate_prior_failures (the four-part discipline check)
# ---------------------------------------------------------------------------


def test_validate_missing_field_rejects():
    """No prior_failures field → invalid."""
    result = validate_prior_failures({})
    assert result.valid is False
    assert "prior_failures" in result.missing_fields[0]


def test_validate_empty_list_rejects():
    """An empty prior_failures: [] → invalid."""
    result = validate_prior_failures({"prior_failures": []})
    assert result.valid is False


def test_validate_complete_entry_accepts():
    """All four required fields populated → valid."""
    result = validate_prior_failures(
        {
            "prior_failures": [
                {
                    "experiment_id": "exp870-sota-code-repair-v7",
                    "verdict": "blocked",
                    "addressed_by": "Pivot to transformers loader (Exp 881)",
                    "retire_if_same_verdict": True,
                }
            ]
        }
    )
    assert result.valid is True


@pytest.mark.parametrize(
    "missing",
    [
        "experiment_id",
        "verdict",
        "addressed_by",
        "retire_if_same_verdict",
    ],
)
def test_validate_rejects_each_missing_required_field(missing):
    """Each of the four required fields is independently required."""
    entry = {
        "experiment_id": "expXXX",
        "verdict": "blocked",
        "addressed_by": "...",
        "retire_if_same_verdict": True,
    }
    del entry[missing]
    result = validate_prior_failures({"prior_failures": [entry]})
    assert result.valid is False
    assert missing in result.missing_fields


def test_validate_rejects_empty_string_field():
    """An empty string in any required field is treated as missing."""
    result = validate_prior_failures(
        {
            "prior_failures": [
                {
                    "experiment_id": "expXXX",
                    "verdict": "",  # empty
                    "addressed_by": "fix",
                    "retire_if_same_verdict": True,
                }
            ]
        }
    )
    assert result.valid is False
    assert "verdict" in result.missing_fields


def test_validate_rejects_non_dict_entry():
    """An entry that isn't a dict is rejected with a clear reason."""
    result = validate_prior_failures({"prior_failures": ["not a dict"]})
    assert result.valid is False
    assert "not a dict" in result.reason


# ---------------------------------------------------------------------------
# End-to-end: is_doomed_rerun against the .65-.68 patterns
# ---------------------------------------------------------------------------


def test_is_doomed_rerun_blocks_when_no_prior_failures(tmp_path):
    """A task whose scope matches a prior failure but lacks
    `prior_failures:` field is blocked.

    Reproduces the .68 case where Exp 881 (code-repair-v8) was proposed
    after Exp 870 (sota-code-repair-v7) failed, with no `prior_failures:`
    field on the new task. The ledger should refuse it.
    """
    # Seed: Exp 870 failed with model_not_cached
    _write_artifact(tmp_path / "results", 870, "sota_code_repair_v7", "blocked_model_not_cached")
    ledger = FailureLedger.load_from_artifacts(tmp_path)

    new_task = {
        "id": "exp881-code-repair-v8-gemma4",
        "title": "Exp 881: Code Repair v8",
        # NO prior_failures: field
    }
    check = ledger.is_doomed_rerun(new_task)
    assert check.blocked is True
    assert "prior_failures" in check.reason
    assert len(check.matched_priors) == 1


def test_is_doomed_rerun_allows_when_prior_failures_complete(tmp_path):
    """A task with a complete prior_failures: entry passes the check."""
    _write_artifact(tmp_path / "results", 870, "sota_code_repair_v7", "blocked_model_not_cached")
    ledger = FailureLedger.load_from_artifacts(tmp_path)

    new_task = {
        "id": "exp881-code-repair-v8-gemma4",
        "title": "Exp 881: Code Repair v8 — Gemma4 transformers",
        "prior_failures": [
            {
                "experiment_id": "exp870-sota-code-repair-v7",
                "verdict": "blocked_model_not_cached",
                "addressed_by": (
                    "Pivot away from GGUF download path entirely; use "
                    "Gemma4-E4B-it via transformers loader (already cached)"
                ),
                "retire_if_same_verdict": True,
            }
        ],
    }
    check = ledger.is_doomed_rerun(new_task)
    assert check.blocked is False
    assert "satisfies discipline" in check.reason


def test_is_doomed_rerun_allows_unrelated_task(tmp_path):
    """A task whose scope doesn't match any prior is allowed."""
    _write_artifact(tmp_path / "results", 840, "live_benchmark", "simulation_fallback")
    ledger = FailureLedger.load_from_artifacts(tmp_path)
    unrelated = {
        "id": "exp819-ising-field-fix",
        "title": "Exp 819: Ising External Field Fix",
    }
    check = ledger.is_doomed_rerun(unrelated)
    assert check.blocked is False
    assert "no scope-matching priors" in check.reason


def test_is_doomed_rerun_conservative_matcher_does_not_match_unrelated_stems(tmp_path):
    """`live-benchmark` and `live-cascade` are different scopes despite
    sharing the prefix `live-`.

    Demonstrates the conservative matcher's by-design behavior: a
    short shared prefix (5 chars `live-`) is below the 8-char minimum
    overlap threshold, so the matcher does NOT flag this as a doomed
    rerun. False negatives (missed scope matches) are accepted as the
    cost of avoiding false positives (incorrectly blocking legitimate
    iterations).
    """
    _write_artifact(tmp_path / "results", 840, "live_benchmark_v3", "simulation_fallback")
    _write_artifact(tmp_path / "results", 853, "live_benchmark_v4", "simulation_fallback")
    _write_artifact(tmp_path / "results", 858, "live_benchmark_v5", "simulation_fallback")
    _write_artifact(tmp_path / "results", 871, "live_benchmark_v6", "simulation_fallback")
    ledger = FailureLedger.load_from_artifacts(tmp_path)

    new_task_no_prior = {
        "id": "exp882-live-cascade-v7",
        "title": "Exp 882: Live Cascade v7",
    }
    check = ledger.is_doomed_rerun(new_task_no_prior)
    assert check.blocked is False
    assert len(check.matched_priors) == 0
    assert "no scope-matching priors" in check.reason


def test_is_doomed_rerun_blocks_recurring_live_benchmark_chain(tmp_path):
    """When the new task DOES share a long substring with priors,
    the matcher correctly blocks. `live-benchmark-v6` shares the
    11-char `live-benchmark` substring with `live-benchmark-v3`.
    """
    _write_artifact(tmp_path / "results", 840, "live_benchmark_v3", "simulation_fallback")
    _write_artifact(tmp_path / "results", 853, "live_benchmark_v4", "simulation_fallback")
    _write_artifact(tmp_path / "results", 858, "live_benchmark_v5", "simulation_fallback")
    ledger = FailureLedger.load_from_artifacts(tmp_path)

    new_task = {
        "id": "exp871-live-benchmark-v6",
        "title": "Exp 871: Live Benchmark v6",
        # no prior_failures — should be blocked
    }
    check = ledger.is_doomed_rerun(new_task)
    assert check.blocked is True
    assert len(check.matched_priors) == 3  # 840, 853, 858 all match


def test_is_doomed_rerun_blocks_when_prior_failures_field_incomplete(tmp_path):
    """A task that has prior_failures: but missing fields is still blocked."""
    _write_artifact(tmp_path / "results", 870, "sota_code_repair_v7", "blocked")
    ledger = FailureLedger.load_from_artifacts(tmp_path)
    new_task = {
        "id": "exp881-code-repair-v8",
        "title": "Exp 881: Code Repair v8",
        "prior_failures": [
            {
                "experiment_id": "exp870-sota-code-repair-v7",
                "verdict": "blocked",
                # MISSING addressed_by + retire_if_same_verdict
            }
        ],
    }
    check = ledger.is_doomed_rerun(new_task)
    assert check.blocked is True
    assert "prior_failures" in check.reason or "missing" in check.reason
