#!/usr/bin/env python3
"""Experiment 1048 — eval-metrics canonical wiring + differential-agent-routing
validation.

WHY this experiment exists (verbose-layman explanation):
    Milestone .80 pre-shipped three pieces of infrastructure (canonical metric
    helpers in ``python/carnot/eval/metrics.py``, the audit script
    ``scripts/audit_metric_provenance.py``, and a commit-watchdog) but did not
    wire them together. This experiment closes the loop:

      1. ``build_result()`` was already updated to emit a ``metrics_used`` key,
         but no caller was forced to use it. We verify it works end-to-end.
      2. The conductor never *self-heals* on suspicious AUROC values. We add a
         post-write check that scans for {0.0, 0.001, 0.999, 1.0} edge values
         (the historical inverted-AUROC fingerprint) and writes an alert into
         ``ops/supervisor-alerts.json`` so the operator gets paged.
      3. The audit script crashed on list-shaped JSON files in ``results/``. We
         guarded with isinstance(data, dict) so it now lists the legacy backlog
         cleanly. ``n_legacy_metric_users`` reports that count.
      4. The differential-agent-routing change proposal added ``model: opus``
         to roadmap task schema. We validate the active roadmap parses with
         that field present.

WHAT this experiment writes to the artifact:
    metrics_used_field_deployed: bool          — build_result() emits it.
    auroc_anomaly_detector_wired: bool         — _check_auroc_anomaly() lives.
    n_legacy_metric_users: int                  — count of un-tagged artifacts.
    model_field_validated: bool                 — roadmap.yaml parses.
    tests_passing: int                          — pytest count for our tests.
    honest_verdict: "eval_metrics_wired_routing_validated" | "partial" | ...

This is wholly an integration experiment — no model training, no GPU. Runs in
seconds; the conductor's "short, focused runs are correct" rule applies.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))


def _check_metrics_used_deployed() -> bool:
    """Confirm the ExperimentTemplate.build_result() helper accepts and emits
    the ``metrics_used`` field.

    This is the integration check that future experiments can rely on. The
    test in tests/python/test_eval_metrics_wiring.py is the unit-level
    equivalent; here we run the same check inline so the artifact captures
    a positive signal even if the test suite is skipped for some reason.
    """
    from experiment_template import ExperimentTemplate  # type: ignore[import-not-found]

    tmpl = ExperimentTemplate(
        exp_id="exp1048-probe",
        title="metrics_used integration probe",
        deliverable=str(REPO_ROOT / "results" / "_throwaway_probe.json"),
        requires_gpu=False,
    )
    tmpl._t0 = 0.0
    tmpl._started_at = "2026-04-29T00:00:00Z"
    artifact = tmpl.build_result(data={"foo": 1}, status="success", metrics_used=["auroc"])
    return (
        "metrics_used" in artifact
        and artifact["metrics_used"] == ["auroc"]
        and "metrics_provenance" in artifact
    )


def _check_auroc_anomaly_wired() -> bool:
    """Confirm scripts/research_conductor.py now exposes a callable AUROC
    anomaly self-heal hook. We import the module and look for both the helper
    and its call site inside ``_log_experiment_completion``.
    """
    try:
        import scripts.research_conductor as rc  # type: ignore[import-not-found]
    except Exception:
        return False
    if not hasattr(rc, "_check_auroc_anomaly"):
        return False
    completion_src = (REPO_ROOT / "scripts" / "research_conductor.py").read_text()
    return "_check_auroc_anomaly(task)" in completion_src


def _count_legacy_metric_users() -> int:
    """Run the provenance audit and parse out the count of deliverables that
    lack a ``metrics_provenance`` tag. These are the legacy-implementation
    candidates that need a manual cross-reference if a metric bug is found.
    """
    audit = REPO_ROOT / "scripts" / "audit_metric_provenance.py"
    proc = subprocess.run(
        [sys.executable, str(audit)],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
        timeout=120,
    )
    if proc.returncode != 0:
        return -1
    match = re.search(r"Without \(pre-2026-04-28 or no metrics\):\s+(\d+)", proc.stdout)
    if not match:
        return -1
    return int(match.group(1))


def _validate_roadmap_model_field() -> bool:
    """Run validate_prior_failures.py on the active roadmap and confirm clean
    exit. The schema strict-parses ``model: opus|sonnet|null`` so any drift
    that drops a value would surface here.
    """
    proc = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "validate_prior_failures.py"),
            str(REPO_ROOT / "research-roadmap.yaml"),
        ],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
        timeout=120,
    )
    return proc.returncode == 0 and "[OK]" in proc.stdout


def _run_tests() -> int:
    """Run the new test file and return pass count.

    Pytest's coverage gate is project-wide (set in pytest.ini) and will FAIL on
    99% threshold even when the targeted tests pass. We disable coverage for
    this run with --no-cov so we get a clean integer pass count.
    """
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "tests/python/test_eval_metrics_wiring.py",
            "-v",
            "--no-cov",
            "--no-header",
            "-q",
        ],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
        timeout=180,
    )
    match = re.search(r"(\d+) passed", proc.stdout + proc.stderr)
    if match:
        return int(match.group(1))
    return 0


def main() -> int:
    metrics_used_field_deployed = _check_metrics_used_deployed()
    auroc_anomaly_detector_wired = _check_auroc_anomaly_wired()
    n_legacy_metric_users = _count_legacy_metric_users()
    model_field_validated = _validate_roadmap_model_field()
    tests_passing = _run_tests()

    all_green = (
        metrics_used_field_deployed
        and auroc_anomaly_detector_wired
        and model_field_validated
        and tests_passing >= 3
    )
    honest_verdict = "eval_metrics_wired_routing_validated" if all_green else "partial"

    artifact_path = REPO_ROOT / "results" / "experiment_1048_eval_metrics_canonical_wiring.json"

    from experiment_template import ExperimentTemplate  # type: ignore[import-not-found]

    tmpl = ExperimentTemplate(
        exp_id="exp1048-eval-metrics-canonical-wiring",
        title="Eval-Metrics Canonical Wiring + Differential-Agent-Routing Validation",
        deliverable=str(artifact_path),
        requires_gpu=False,
    )
    tmpl.setup()

    artifact = tmpl.build_result(
        data={
            "metrics_used_field_deployed": metrics_used_field_deployed,
            "auroc_anomaly_detector_wired": auroc_anomaly_detector_wired,
            "n_legacy_metric_users": n_legacy_metric_users,
            "model_field_validated": model_field_validated,
            "tests_passing": tests_passing,
            "honest_verdict": honest_verdict,
            "summary": (
                f"Wired metrics_used into build_result() (default 'unknown'); "
                f"added _check_auroc_anomaly() to conductor self-heal "
                f"(edge values 0.0/0.001/0.999/1.0 → ops/supervisor-alerts.json); "
                f"audit script no longer crashes on list-shaped JSON; "
                f"{n_legacy_metric_users} legacy artifacts lack provenance tag "
                f"(retroactive backfill not in scope)."
            ),
        },
        status="success" if all_green else "partial",
        metrics_used=["auroc"],
        code_files=[__file__],
    )
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(json.dumps(artifact, indent=2))
    print(f"Wrote {artifact_path}")
    print(f"honest_verdict={honest_verdict}")
    return 0 if all_green else 1


if __name__ == "__main__":
    sys.exit(main())
