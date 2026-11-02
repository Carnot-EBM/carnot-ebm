#!/usr/bin/env python3
"""Experiment 1067 — Verify the two cascade-failure fixes shipped in milestone .80.

Why this experiment exists
--------------------------
Two conductor bugs were diagnosed and fixed in milestone .80 but never
replay-verified end-to-end because the prior milestones kept failing pre-tests
or running into other wedges before the verification step landed:

  Bug 1 — Bootstrap-stub fast-path
      ``_deliverable_exists()`` originally returned True for any file at the
      task's deliverable path. Sonnet's "write artifact FIRST" defensive
      pattern produced bootstrap stubs containing ``{"status": "running",
      ...}`` and the fast-path then short-circuited every retry. Downstream
      gates read the ``False`` placeholder fields forever.

  Bug 2 — YAML-bool / Python-bool gate mismatch
      ``_eval_op()`` compared ``actual == expected`` as raw Python objects.
      When YAML deserialised a value as the *string* ``"True"`` but the
      gate's RHS was Python ``True`` (or vice-versa), the comparison
      returned ``False`` even though both sides represented the same
      Boolean intent. Several downstream tasks gate-blocked indefinitely.

Both fixes are already on disk:
  - ``scripts/research_conductor.py:_deliverable_exists`` reads the JSON
    status field and refuses to fast-path bootstrap statuses.
  - ``scripts/conductor_gates.py:_coerce_gate_value`` normalises common
    truthy/falsy strings (and 0/1 numbers) onto Python bool when at least
    one side of an ``==``/``!=`` is a Python bool.

This experiment closes them out by:
  1. Asserting the symbols are present in their respective modules.
  2. Running ``tests/python/test_conductor_deliverable_status.py`` and
     recording how many tests pass (the original change proposal called
     for 12 tests; the file has since grown to include parametrised
     regressions, so we record the exact number rather than asserting
     == 12).
  3. Replaying the bootstrap-stub wedge: a JSON file with
     ``{"status": "running", ...}`` MUST NOT be reported as a finished
     deliverable.
  4. Replaying the gate-coercion wedge: ``_coerce_gate_value("True")``,
     ``_coerce_gate_value("true")``, ``_coerce_gate_value("1")`` MUST all
     return Python ``True``; ``_coerce_gate_value(False)`` MUST stay
     ``False``.

The artifact's ``honest_verdict`` is ``both_fixes_deployed_verified`` only
when every check above succeeds. Any failure downgrades the verdict so the
operational retrospective can pick up the regression.

Note on the prompt's suggested ``rc._coerce_gate_value`` invocation: the
function lives at module scope in ``scripts/conductor_gates``, not on the
``ResearchConductor`` class, so this script imports it directly. Same
spirit, correct location.
"""

from __future__ import annotations

import datetime as _dt
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

EXPERIMENT_ID = 1067
EXPERIMENT_SLUG = "exp1067-gate-coercion-v3"
TITLE = "Conductor fastpath + gate-coercion replay verification (v3)"
DELIVERABLE = PROJECT_ROOT / "results" / "experiment_1067_gate_coercion_v3.json"


def _check_fastpath_fix() -> tuple[bool, dict]:
    """Confirm `_deliverable_exists` rejects status='running' bootstrap stubs.

    The check creates a temporary JSON file with ``{"status": "running"}``,
    hands it to ``_deliverable_exists`` via a fake task dict, and verifies
    the function returns False (i.e. the file is NOT treated as a finished
    deliverable). This is the core property the .80 fix added.
    """
    from scripts.research_conductor import _deliverable_exists  # noqa: PLC0415

    with tempfile.NamedTemporaryFile(
        "w", suffix=".json", delete=False, dir=str(PROJECT_ROOT / "results")
    ) as fh:
        fh.write(json.dumps({"status": "running", "honest_verdict": "n/a"}))
        wedge_path = Path(fh.name)
    try:
        rel = wedge_path.relative_to(PROJECT_ROOT)
        result_running = _deliverable_exists({"deliverable": str(rel)})
    finally:
        wedge_path.unlink(missing_ok=True)

    # Also verify a `success` artifact IS still treated as finished, so we
    # haven't over-corrected and started skipping good deliverables.
    with tempfile.NamedTemporaryFile(
        "w", suffix=".json", delete=False, dir=str(PROJECT_ROOT / "results")
    ) as fh:
        fh.write(json.dumps({"status": "success", "honest_verdict": "ok"}))
        ok_path = Path(fh.name)
    try:
        rel = ok_path.relative_to(PROJECT_ROOT)
        result_success = _deliverable_exists({"deliverable": str(rel)})
    finally:
        ok_path.unlink(missing_ok=True)

    passed = (result_running is False) and (result_success is True)
    return passed, {
        "running_treated_as_unfinished": result_running is False,
        "success_treated_as_finished": result_success is True,
    }


def _check_gate_coercion_fix() -> tuple[bool, dict]:
    """Confirm `_coerce_gate_value` normalises bool-ish strings to bool.

    We exercise the same shapes the .80 wedge actually saw in the wild:
    string "True"/"true"/"1" must coerce to Python True; string
    "False"/"false"/"0" must coerce to Python False; native booleans must
    pass through unchanged. Strings that are NOT bool-ish (e.g.
    "preflight_complete") must round-trip unchanged so we don't accidentally
    rewrite string-equality gates.
    """
    from scripts.conductor_gates import _coerce_gate_value  # noqa: PLC0415

    cases = {
        "True_str": _coerce_gate_value("True") is True,
        "true_str": _coerce_gate_value("true") is True,
        "TRUE_str": _coerce_gate_value("TRUE") is True,
        "one_str": _coerce_gate_value("1") is True,
        "yes_str": _coerce_gate_value("yes") is True,
        "False_str": _coerce_gate_value("False") is False,
        "false_str": _coerce_gate_value("false") is False,
        "zero_str": _coerce_gate_value("0") is False,
        "True_native": _coerce_gate_value(True) is True,
        "False_native": _coerce_gate_value(False) is False,
        "int_one": _coerce_gate_value(1) is True,
        "int_zero": _coerce_gate_value(0) is False,
        "non_bool_string_passthrough": (
            _coerce_gate_value("preflight_complete") == "preflight_complete"
        ),
    }
    return all(cases.values()), cases


def _run_deliverable_status_tests() -> tuple[int, int, str]:
    """Run the deliverable-status test file and return (passed, total, raw_output_tail).

    Coverage is disabled (``--no-cov``) because the global pytest config sets
    a 99% project-wide minimum that this single test file cannot satisfy on
    its own — that minimum is enforced by the broader CI run, not here.
    """
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "tests/python/test_conductor_deliverable_status.py",
            "-v",
            "--no-cov",
            "-p",
            "no:cacheprovider",
        ],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        timeout=180,
    )
    out = proc.stdout + proc.stderr
    passed = out.count(" PASSED")
    failed = out.count(" FAILED")
    total = passed + failed
    tail = "\n".join(out.strip().splitlines()[-5:])
    return passed, total, tail


def main() -> int:
    """Run all four checks and write the artifact."""
    started_at = _dt.datetime.now(_dt.UTC)
    t0 = time.time()

    fastpath_ok, fastpath_detail = _check_fastpath_fix()
    gate_ok, gate_detail = _check_gate_coercion_fix()
    tests_passed, tests_total, tests_tail = _run_deliverable_status_tests()
    tests_all_green = tests_total > 0 and tests_passed == tests_total

    if fastpath_ok and gate_ok and tests_all_green:
        verdict = "both_fixes_deployed_verified"
        status = "success"
    elif fastpath_ok and not gate_ok:
        verdict = "fastpath_only"
        status = "partial"
    elif gate_ok and not fastpath_ok:
        verdict = "gate_only"
        status = "partial"
    elif not fastpath_ok and not gate_ok:
        verdict = "neither_fixed"
        status = "failed"
    else:
        verdict = "failed"
        status = "failed"

    finished_at = _dt.datetime.now(_dt.UTC)
    artifact = {
        "schema": "carnot.experiment.v1",
        "experiment": EXPERIMENT_SLUG,
        "experiment_id": EXPERIMENT_ID,
        "title": TITLE,
        "run_date": started_at.isoformat(),
        "started_at": started_at.isoformat(),
        "finished_at": finished_at.isoformat(),
        "duration_s": round(time.time() - t0, 3),
        "status": status,
        "honest_verdict": verdict,
        "fastpath_fix_present": fastpath_ok,
        "fastpath_detail": fastpath_detail,
        "gate_coercion_fixed": gate_ok,
        "gate_coercion_detail": gate_detail,
        "deliverable_status_tests_passing": tests_passed,
        "deliverable_status_tests_total": tests_total,
        "deliverable_status_tests_tail": tests_tail,
        "wedge_replay_clean": fastpath_ok,
        "gate_coercion_replay_clean": gate_ok,
        "predecessor_experiments": [
            "exp1039-conductor-fastpath-gate-coercion",
        ],
        "fix_locations": {
            "fastpath": "scripts/research_conductor.py:_deliverable_exists",
            "gate_coercion": "scripts/conductor_gates.py:_coerce_gate_value",
        },
        "change_proposal": ("openspec/change-proposals/conductor-fastpath-bootstrap-skip.md"),
        "decision_class": "verify",
    }

    DELIVERABLE.parent.mkdir(parents=True, exist_ok=True)
    with DELIVERABLE.open("w", encoding="utf-8") as fh:
        json.dump(artifact, fh, indent=2, sort_keys=False)
        fh.write("\n")

    print(
        json.dumps(
            {
                "verdict": verdict,
                "fastpath_ok": fastpath_ok,
                "gate_ok": gate_ok,
                "tests_passed": tests_passed,
                "tests_total": tests_total,
                "deliverable": str(DELIVERABLE.relative_to(PROJECT_ROOT)),
            },
            indent=2,
        )
    )
    return 0 if status == "success" else 1


if __name__ == "__main__":
    sys.exit(main())
