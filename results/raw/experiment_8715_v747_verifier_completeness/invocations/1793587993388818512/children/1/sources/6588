#!/usr/bin/env python3
"""Exp 1105 — Failure-Ledger v2: Issues 2+3+4 + doc-reconcile batch flag.

Three further conductor regressions surfaced during milestone .85 in
addition to the Issues 1/5/manifest set already shipped by exp1104.
This experiment ships the surgical fixes for all three plus an
operator-opt-in doc-reconcile batching flag.

The fixes themselves live entirely inside
``scripts/research_conductor.py`` — no new module is needed because
each fix is a small change at an existing call site.  This script's
job is to verify that the wire-in markers are present in the
conductor source, run the regression test file that locks the
behaviour in, and emit the deployment artifact.

Failure modes being closed
---------------------------
- Issue 2 (cap-resets race operator patches): when the operator
  commits a fix between the conductor's third failure and the next
  pick_next_task call, the patch should not lose to the cap that
  fires ~30 minutes after the first failure.  Fix: at pick_next_task
  time, check git log since the last recorded failure for commits
  that touch any path the task cares about (deliverable, roadmap
  entry, upstream gated_on artifact); if anything matched, reset the
  task's failure counter to 0.
- Issue 3 (stable-deliverable-detection uses 60s unchanged mtime):
  the conductor's in-loop "kill the agent early when its deliverable
  has been stable for 60s" path was firing on stale artifacts left
  over from earlier iterations.  Fix: require ``mtime > start_time``
  before allowing the early kill — anything older than the current
  agent's start timestamp is stale by construction.
- Issue 4 (pre-test fingerprint cache saves START state): the cache
  was missing on subsequent iterations whenever a commit landed
  during the pre-test, because the cache was saved with the
  START-of-pretest fingerprint.  Fix: recompute the fingerprint
  AFTER the pre-test run completes and persist that, so the next
  iteration's fingerprint (which reflects the post-commit file
  state) hits the cache.
- Doc-reconcile batching: the per-experiment Haiku doc reconciler
  was eating ~28 min of inline blocking time across the milestone.
  This experiment adds the ``CARNOT_BATCH_DOC_RECONCILE`` env flag
  (default off): when enabled, doc-reconcile work is queued and
  flushed at end-of-batch instead of running inline.

Verification strategy
---------------------
We verify the conductor source contains the wire-in markers rather
than monkey-patching the running conductor.  The conductor module
has heavy import-time side effects (logging setup, lock files, GPU
reaper) we don't want to trigger during a small audit experiment.
The wire-in markers are stable, easy-to-grep substrings that the
fix code introduced.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _verify_conductor_wire_in() -> dict[str, bool]:
    """Confirm the conductor source contains the wire-in markers for each fix.

    Each marker is a short, stable substring introduced specifically by
    the fix.  We deliberately match on the comment+code together so a
    refactor that changes only the comment without touching the code
    (or vice versa) is caught.
    """
    src_path = PROJECT_ROOT / "scripts" / "research_conductor.py"
    src = src_path.read_text()
    return {
        # Issue 2 — _cap_reset_applies + caller in pick_next_task
        "issue2_cap_reset_helper_present": "def _cap_reset_applies(" in src,
        "issue2_cap_reset_invoked_in_pick_next_task": "Issue-2 cap-reset:" in src,
        "issue2_changed_files_since_helper": "def _changed_files_since(" in src,
        # Issue 3 — mtime > start_time guard inside the run_agent loop
        "issue3_mtime_guard_present": "if st.st_mtime <= start_time:" in src,
        "issue3_mtime_guard_doc": "fix (Issue 3): require mtime > start_time" in src,
        # Issue 4 — END-of-pretest fingerprint persistence
        "issue4_end_fingerprint_persist": "fix (Issue 4): persist the END-of-pretest fingerprint"
        in src,
        "issue4_end_fingerprint_recompute": "end_fp = _compute_pretest_fingerprint()" in src,
        # Doc-reconcile batching flag and queue plumbing
        "doc_reconcile_flag_present": 'os.environ.get("CARNOT_BATCH_DOC_RECONCILE"' in src,
        "doc_reconcile_enqueue_present": "def _enqueue_doc_reconcile(" in src,
        "doc_reconcile_flush_present": "def _flush_doc_reconcile_batch(" in src,
        "doc_reconcile_inline_path_branches_on_flag": "if _doc_reconcile_batch_enabled():" in src,
    }


def _run_tests() -> tuple[int, int]:
    """Run the regression test module and return (n_written, n_passing).

    We deliberately scope the test run to the test file this
    experiment exercises.  Project-wide tests are run by the
    conductor's own pre/post-test gates and are not the responsibility
    of an audit experiment.
    """
    test_file = PROJECT_ROOT / "tests" / "python" / "test_failure_ledger_v2_cap.py"
    pytest_bin = PROJECT_ROOT / ".venv" / "bin" / "pytest"
    n_written = _count_test_functions(test_file)
    if not pytest_bin.exists():
        return n_written, 0  # cannot run; report the count we wrote
    result = subprocess.run(
        [str(pytest_bin), str(test_file), "-v", "--no-cov", "-p", "no:cacheprovider"],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        env={"JAX_PLATFORMS": "cpu", "PATH": "/usr/bin:/bin"},
    )
    output = result.stdout + result.stderr
    n_passed = len(re.findall(r"\bPASSED\b", output))
    return n_written, n_passed


def _count_test_functions(test_file: Path) -> int:
    """Count test functions in the regression file.

    Used so the artifact's ``tests_written`` field stays in sync with
    the actual test file as it grows.  We grep for ``def test_`` lines
    rather than parsing the AST — the file is small and the regex is
    sufficient.
    """
    if not test_file.exists():
        return 0
    return len(re.findall(r"^\s*def test_", test_file.read_text(), flags=re.MULTILINE))


def main() -> int:
    started = datetime.now(UTC)

    wire_in = _verify_conductor_wire_in()
    n_written, n_passing = _run_tests()

    issue2_deployed = (
        wire_in["issue2_cap_reset_helper_present"]
        and wire_in["issue2_cap_reset_invoked_in_pick_next_task"]
        and wire_in["issue2_changed_files_since_helper"]
    )
    issue3_deployed = wire_in["issue3_mtime_guard_present"] and wire_in["issue3_mtime_guard_doc"]
    issue4_deployed = (
        wire_in["issue4_end_fingerprint_persist"] and wire_in["issue4_end_fingerprint_recompute"]
    )
    doc_reconcile_flag_added = (
        wire_in["doc_reconcile_flag_present"]
        and wire_in["doc_reconcile_enqueue_present"]
        and wire_in["doc_reconcile_flush_present"]
        and wire_in["doc_reconcile_inline_path_branches_on_flag"]
    )

    deployed_count = sum(
        [issue2_deployed, issue3_deployed, issue4_deployed, doc_reconcile_flag_added]
    )
    tests_pass = n_passing == n_written and n_written > 0
    if deployed_count == 4 and tests_pass:
        verdict = "all_four_fixes_deployed"
    elif deployed_count == 3:
        verdict = "three_of_four_deployed"
    elif deployed_count >= 1:
        verdict = "partial"
    else:
        verdict = "failed"

    finished = datetime.now(UTC)
    artifact = {
        "experiment": "exp1105-failure-ledger-v2-cap-mtime-fingerprint",
        "title": "Failure-Ledger v2 (Issues 2+3+4 + doc-reconcile batch flag)",
        "run_date": finished.strftime("%Y-%m-%d"),
        "duration_s": (finished - started).total_seconds(),
        "schema_version": 1,
        "status": "success" if verdict == "all_four_fixes_deployed" else "partial",
        "honest_verdict": verdict,
        # Required schema fields per task instructions:
        "failure_ledger_cap_reset_deployed": issue2_deployed,
        "stable_deliverable_mtime_fix_deployed": issue3_deployed,
        "end_fingerprint_cache_deployed": issue4_deployed,
        "doc_reconcile_batch_flag_added": doc_reconcile_flag_added,
        "tests_written": n_written,
        "tests_passing": n_passing,
        "wire_in_checks": wire_in,
        "deliverable": ("results/experiment_1105_failure_ledger_v2_cap_mtime_fingerprint.json"),
        "summary": (
            f"Issue 2 (cap-reset on fix-commit): {issue2_deployed}; "
            f"Issue 3 (mtime>start_time staleness guard): {issue3_deployed}; "
            f"Issue 4 (end-of-pretest fingerprint cache): {issue4_deployed}; "
            f"doc-reconcile batch flag: {doc_reconcile_flag_added}; "
            f"tests {n_passing}/{n_written}"
        ),
    }
    out_path = (
        PROJECT_ROOT / "results" / "experiment_1105_failure_ledger_v2_cap_mtime_fingerprint.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
    print(json.dumps(artifact, indent=2, sort_keys=True))
    return 0 if verdict == "all_four_fixes_deployed" else 1


if __name__ == "__main__":
    sys.exit(main())
