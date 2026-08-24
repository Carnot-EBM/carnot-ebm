"""Tests for eval-metrics canonical wiring (Exp 1048).

What this verifies and why:
    1. ``build_result()`` always emits a ``metrics_used`` field so that downstream
       audits can identify which metric implementation produced the numbers in
       any artifact. Pre-2026-04-28 artifacts predate this and are tagged
       ``"unknown"`` by default.
    2. ``scripts/audit_metric_provenance.py`` runs end-to-end without raising on
       the current results corpus. The script crashed on list-shaped JSON
       artifacts (some experiment files store a list at the top level, not a
       dict) — the fix guards with ``isinstance(data, dict)``.
    3. ``scripts/validate_prior_failures.py`` accepts the ``model: opus``
       differential-agent-routing field on roadmap tasks. The Pydantic schema in
       ``scripts/roadmap_schema.py`` enumerates ``Literal["sonnet", "opus"]``
       (plus ``None``); a regression that drops ``opus`` would silently route
       complex tasks to Sonnet and reproduce the .80 wedge.

Spec: REQ-EVAL-001 (canonical metric implementations),
      REQ-EVAL-005 (provenance audit tooling),
      differential-agent-routing change proposal.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))


def test_build_result_emits_metrics_used_field(tmp_path: Path, monkeypatch) -> None:
    """build_result() must always include a metrics_used key.

    Default is the sentinel ``"unknown"`` so an experiment that forgot to pass
    metrics_used does not silently appear to have used the canonical helper.
    """
    from experiment_template import ExperimentTemplate  # noqa: WPS433 (test-time import)

    monkeypatch.setenv("CARNOT_EXPERIMENT_ARTIFACT_ROOT", str(tmp_path))
    deliverable = tmp_path / "experiment_x.json"
    tmpl = ExperimentTemplate(
        exp_id="exp9999-test",
        title="metrics_used field unit test",
        deliverable=str(deliverable),
        requires_gpu=False,
    )
    # setup() is intentionally skipped so we test build_result() in isolation
    # without touching the global env-propagation guard or the live env asserts.
    tmpl._t0 = 0.0  # build_result reads perf_counter() - _t0 for duration_s
    tmpl._started_at = "2026-04-29T00:00:00Z"

    artifact_default = tmpl.build_result(data={"foo": 1}, status="success")
    assert "metrics_used" in artifact_default
    assert artifact_default["metrics_used"] == "unknown"

    artifact_explicit = tmpl.build_result(
        data={"foo": 1},
        status="success",
        metrics_used=["auroc"],
    )
    assert artifact_explicit["metrics_used"] == ["auroc"]
    assert "metrics_provenance" in artifact_explicit
    assert any(
        v.startswith("carnot.eval.metrics.auroc:")
        for v in artifact_explicit["metrics_provenance"].values()
    )


def test_audit_metric_provenance_runs_without_exception() -> None:
    """The audit script must complete on the current corpus.

    Origin: 2026-04-28 inverted-AUROC discovery required a manual grep pass.
    The audit script automates that, but it was crashing on list-shaped JSON
    files. After the isinstance(data, dict) guard, it must run cleanly.
    """
    audit_script = REPO_ROOT / "scripts" / "audit_metric_provenance.py"
    assert audit_script.exists()
    proc = subprocess.run(
        [sys.executable, str(audit_script)],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
        timeout=60,
    )
    # 0 = clean, 1 = flagged-buggy with --flag-buggy. We pass no flags so 0 expected.
    assert proc.returncode == 0, (
        f"audit_metric_provenance.py exit={proc.returncode}\n"
        f"stdout: {proc.stdout[:500]}\nstderr: {proc.stderr[:500]}"
    )
    assert "Metrics Provenance Audit" in proc.stdout


def test_validate_prior_failures_accepts_model_field(tmp_path: Path) -> None:
    """Roadmap schema must accept tasks with `model: opus` and `model: sonnet`.

    Regression target: differential-agent-routing change proposal added
    Literal["sonnet", "opus"] | None to ResearchTask.model. A schema change
    that removed opus would route complex tasks to Sonnet and reproduce the
    .80 milestone wedge (exp1028 bootstrap-and-bail).
    """
    fake_roadmap = {
        "milestone": "2026.04.99",
        "milestone_title": "Test milestone",
        "milestone_doc": "openspec/change-proposals/test.md",
        "tasks": [
            {
                "id": "exp9001-fake",
                "milestone": "2026.04.99",
                "deliverable": "results/experiment_9001_fake.json",
                "title": "Fake task with opus model field",
                "model": "opus",
                "prompt": "Fake prompt body",
            },
            {
                "id": "exp9002-fake",
                "milestone": "2026.04.99",
                "deliverable": "results/experiment_9002_fake.json",
                "title": "Fake task with sonnet model field",
                "model": "sonnet",
                "prompt": "Fake prompt body",
            },
            {
                "id": "exp9003-fake",
                "milestone": "2026.04.99",
                "deliverable": "results/experiment_9003_fake.json",
                "title": "Fake task without model field falls through to default",
                "prompt": "Fake prompt body",
            },
        ],
    }
    fake_path = tmp_path / "fake-roadmap.yaml"
    fake_path.write_text(yaml.safe_dump(fake_roadmap))

    validate_script = REPO_ROOT / "scripts" / "validate_prior_failures.py"
    proc = subprocess.run(
        [sys.executable, str(validate_script), str(fake_path)],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
        timeout=60,
    )
    assert proc.returncode == 0, (
        f"validate_prior_failures.py exit={proc.returncode}\n"
        f"stdout: {proc.stdout}\nstderr: {proc.stderr}"
    )
    assert "[OK]" in proc.stdout


def test_auroc_anomaly_detector_writes_alert(tmp_path: Path, monkeypatch) -> None:
    """The conductor's AUROC anomaly check writes a supervisor-alerts.json line
    when an artifact reports auroc in {0.0, 0.001, 0.999, 1.0}.

    These are the edge values that historically signalled the inverted-AUROC
    bug pattern (exp995/exp1003 shipped 0.0 and 1.0 due to copy-paste sign
    errors). Catching them through self-heal pages the operator instead of
    silently polluting the milestone narrative.
    """
    import json
    import importlib

    rc = importlib.import_module("scripts.research_conductor")

    # Build a fake project tree so PROJECT_ROOT-relative paths land in tmp_path.
    fake_project = tmp_path
    (fake_project / "results").mkdir()
    (fake_project / "ops").mkdir()
    deliverable_rel = "results/experiment_9999_fake.json"
    (fake_project / deliverable_rel).write_text(
        json.dumps({"experiment": "exp9999", "status": "success", "auroc": 1.0})
    )

    monkeypatch.setattr(rc, "PROJECT_ROOT", fake_project)

    rc._check_auroc_anomaly({"id": "exp9999-test", "deliverable": deliverable_rel})

    alerts_path = fake_project / "ops" / "supervisor-alerts.json"
    assert alerts_path.exists()
    line = alerts_path.read_text().strip().splitlines()[-1]
    record = json.loads(line)
    assert record["alert_type"] == "AUROC_ANOMALY"
    assert "exp9999-test" in record["detail"]


def test_auroc_anomaly_detector_silent_on_normal_values(tmp_path: Path, monkeypatch) -> None:
    """A healthy AUROC (e.g. 0.83) must NOT trigger the anomaly alert.

    Without this assertion the anomaly detector would either page on every
    experiment or fail to discriminate — both render the alert useless.
    """
    import json
    import importlib

    rc = importlib.import_module("scripts.research_conductor")
    fake_project = tmp_path
    (fake_project / "results").mkdir()
    (fake_project / "ops").mkdir()
    deliverable_rel = "results/experiment_9998_fake.json"
    (fake_project / deliverable_rel).write_text(
        json.dumps({"experiment": "exp9998", "status": "success", "auroc": 0.83})
    )

    monkeypatch.setattr(rc, "PROJECT_ROOT", fake_project)
    rc._check_auroc_anomaly({"id": "exp9998-test", "deliverable": deliverable_rel})

    alerts_path = fake_project / "ops" / "supervisor-alerts.json"
    assert not alerts_path.exists() or alerts_path.read_text().strip() == ""
