"""Tests for exp1117 infrastructure hardening v3 (4 fixes carried over from .86 retro).

Spec: REQ-INFRA-095 — dispatch manifest, doc async, bootstrap grace, fast-eval flag.

Each fix targets one of the bottlenecks that recurred across the .83-.86
milestones and together cost ~98 min/milestone of zero-research wall time.
The four assertions below are the regression tripwires that prevent the
fixes from silently regressing in a future conductor edit.

Fix 1 (Bottleneck 1): dispatch-time YAML manifest enforcement.  The
        ``failure_ledger_v2.is_excluded_by_manifest`` helper must report a
        retired experiment id as excluded BEFORE the conductor spawns its
        subagent; otherwise the queue keeps re-launching retired tasks
        (.83-.86 carryover for exp906).

Fix 2 (Bottleneck 2): ``CARNOT_BATCH_DOC_RECONCILE`` defaults to "1".  The
        conductor's main() function setdefault's the env var so async
        post-experiment doc reconciliation is the new baseline; explicit
        unset (CARNOT_BATCH_DOC_RECONCILE=0) is required to opt back into
        inline blocking reconciliation.

Fix 3 (Bottleneck 3): ``grace_period_s`` task-schema field.  Long-running
        GPU tasks declare ``grace_period_s: 1800`` to suppress the
        bootstrap-stable-deliverable kill until 30 min in; everyone else
        gets the 600 s default.  The conductor reads it via
        ``int(task.get("grace_period_s", 600))`` and threads it into
        ``run_agent``.

Fix 4 (Bonus): ``CARNOT_FAST_EVAL=1`` corpus subsampling.  Architecture-
        sweep experiments call ``maybe_subsample_corpus`` to cap the
        FoVer corpus at 500 random pairs (deterministic seed) when the
        flag is set.  Default off to preserve headline-result
        reproducibility.

Spec: openspec/change-proposals/conductor-supervisor.md (operator-attention
reduction).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
PYTHON_DIR = PROJECT_ROOT / "python"
for _p in (str(SCRIPTS_DIR), str(PYTHON_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def test_retired_experiment_blocked_at_dispatch(tmp_path):
    """Fix 1: a task whose extracted experiment_id is in the YAML manifest

    must be reported excluded by the dispatch-time helper.  Reproduces the
    exp1104 dispatch-time enforcement test against a synthetic manifest so
    this check survives even if the production manifest is reorganised.
    """
    from failure_ledger_v2 import is_excluded_by_manifest  # type: ignore[import-not-found]

    manifest_yaml = tmp_path / "exclusion_manifest.yaml"
    manifest_yaml.write_text(
        "retired_experiments:\n"
        "  - experiment_id: 906\n"
        "    completed_milestone: '2026.04.79'\n"
        "    reason: 'no-progress 3 milestones'\n"
    )
    excluded, reason = is_excluded_by_manifest(
        {"id": "exp906-something", "title": "Anything"},
        yaml_manifest_path=manifest_yaml,
    )
    assert excluded, f"exp906 must be blocked at dispatch; reason={reason}"
    assert "906" in reason

    not_excluded, _ = is_excluded_by_manifest(
        {"id": "exp1118-rlvr-ssd-dualgpu", "title": "RLVR SSD"},
        yaml_manifest_path=manifest_yaml,
    )
    assert not not_excluded, "fresh exp must NOT be blocked"


def test_doc_reconcile_batch_mode_default():
    """Fix 2: ``main()`` in research_conductor.py must call

    ``os.environ.setdefault('CARNOT_BATCH_DOC_RECONCILE', '1')``.  We assert
    on the source rather than running main() because main() spawns
    subprocesses; a textual check is the simplest stable contract.
    """
    src = (SCRIPTS_DIR / "research_conductor.py").read_text()
    assert 'os.environ.setdefault("CARNOT_BATCH_DOC_RECONCILE", "1")' in src, (
        "conductor must default CARNOT_BATCH_DOC_RECONCILE=1 in main()"
    )
    assert 'os.environ.get("CARNOT_BATCH_DOC_RECONCILE", "1") == "1"' in src, (
        "conductor must consult the default in args.async_doc_recon promotion"
    )


def test_grace_period_applied_before_bootstrap_guard():
    """Fix 3: ``run_agent`` accepts a ``grace_period_s`` parameter (default

    600) AND the deliverable-stable kill respects it.  We exercise both
    contracts: the function signature and the gate condition in source.
    """
    import inspect

    import research_conductor  # type: ignore[import-not-found]

    sig = inspect.signature(research_conductor.run_agent)
    assert "grace_period_s" in sig.parameters, "run_agent must accept a grace_period_s kwarg"
    default = sig.parameters["grace_period_s"].default
    assert default == 600, f"default grace must be 600s, got {default}"

    src = (SCRIPTS_DIR / "research_conductor.py").read_text()
    assert "(now - start_time) >= grace_period_s" in src, (
        "deliverable-stable kill must be gated on grace_period_s"
    )
    assert 'task.get("grace_period_s", 600)' in src, (
        "research_step must read grace_period_s from the task YAML"
    )


def test_fast_eval_samples_500_pairs(monkeypatch):
    """Fix 4: ``maybe_subsample_corpus`` must return 500 random pairs

    when ``CARNOT_FAST_EVAL=1`` and pass items through unchanged otherwise.
    The seed is fixed so a re-run of the same experiment lands the same
    subset — preserving the only reproducibility guarantee fast-eval can
    offer.
    """
    from carnot.pipeline.verify_repair import maybe_subsample_corpus

    full_corpus = list(range(6548))

    monkeypatch.delenv("CARNOT_FAST_EVAL", raising=False)
    untouched = maybe_subsample_corpus(full_corpus)
    assert untouched == full_corpus, "flag off → must return the corpus unchanged"

    monkeypatch.setenv("CARNOT_FAST_EVAL", "1")
    sampled = maybe_subsample_corpus(full_corpus)
    assert len(sampled) == 500, f"expected 500 sampled pairs, got {len(sampled)}"
    assert all(item in full_corpus for item in sampled), "sampled items must come from corpus"
    assert len(set(sampled)) == 500, "sampled items must be unique (no replacement)"

    sampled_again = maybe_subsample_corpus(full_corpus)
    assert sampled == sampled_again, "fixed seed → same subset across calls"

    small_corpus = list(range(100))
    small_sampled = maybe_subsample_corpus(small_corpus)
    assert small_sampled == small_corpus, "len(items) <= sample_size → must pass through unchanged"
