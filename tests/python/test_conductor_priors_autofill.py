"""Tests for conductor prior-failure autofill.

Spec: REQ-INFRA-078, SCENARIO-INFRA-092, SCENARIO-INFRA-093,
      SCENARIO-INFRA-094
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "scripts" / "conductor_priors_autofill.py"


def _load_module():
    """Load the standalone script without requiring scripts/ to be a package."""
    spec = importlib.util.spec_from_file_location("conductor_priors_autofill", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["conductor_priors_autofill"] = mod
    spec.loader.exec_module(mod)
    return mod


def _roadmap_text(tasks_text: str) -> str:
    return (
        "milestone: 2026.04.99\n"
        "milestone_title: Autofill Fixture\n"
        "tasks:\n"
        f"{tasks_text}"
    )


def _prior(exp_id: str, verdict: str):
    return SimpleNamespace(
        experiment_id=exp_id,
        title=f"{exp_id} title",
        verdict=verdict,
        status_label="fixture",
        scope=exp_id,
        artifact_path=None,
    )


class FakeLedger:
    def __init__(self, priors_by_task: dict[str, list[object]] | None = None) -> None:
        self.priors_by_task = priors_by_task or {}
        self.calls: list[dict] = []

    def matching_priors(self, task: dict) -> list[object]:
        self.calls.append(task)
        return self.priors_by_task.get(task["id"], [])


def test_script_reads_yaml_without_error(tmp_path: Path) -> None:
    """REQ-INFRA-078: the script loads a roadmap and scans every task."""
    mod = _load_module()
    roadmap = tmp_path / "research-roadmap-next.yaml"
    roadmap.write_text(
        _roadmap_text(
            "- id: exp1-fresh\n"
            "  title: Fresh Task\n"
            "  prompt: |\n"
            "    no-op\n"
        ),
        encoding="utf-8",
    )

    summary = mod.autofill_roadmap(roadmap, dry_run=True, ledger=FakeLedger())

    assert summary.tasks_scanned == 1
    assert summary.stubs_generated == 0
    assert summary.already_populated == 0


def test_script_skips_already_populated_tasks(tmp_path: Path) -> None:
    """SCENARIO-INFRA-092: tasks with non-empty prior_failures are not queried."""
    mod = _load_module()
    roadmap = tmp_path / "research-roadmap-next.yaml"
    roadmap.write_text(
        _roadmap_text(
            "- id: exp1-populated\n"
            "  title: Populated Task\n"
            "  prior_failures:\n"
            "    - experiment_id: exp0-old\n"
            "      verdict: failed\n"
            "      addressed_by: already handled\n"
            "      retire_if_same_verdict: true\n"
            "  prompt: |\n"
            "    no-op\n"
            "- id: exp2-empty\n"
            "  title: Empty Task\n"
            "  prompt: |\n"
            "    no-op\n"
        ),
        encoding="utf-8",
    )
    ledger = FakeLedger()

    summary = mod.autofill_roadmap(roadmap, dry_run=True, ledger=ledger)

    assert summary.tasks_scanned == 2
    assert summary.already_populated == 1
    assert [call["id"] for call in ledger.calls] == ["exp2-empty"]


def test_script_classifies_successful_upstreams(tmp_path: Path) -> None:
    """SCENARIO-INFRA-093: non-failure verdicts become successful_upstream stubs."""
    mod = _load_module()
    roadmap = tmp_path / "research-roadmap-next.yaml"
    roadmap.write_text(
        _roadmap_text(
            "- id: exp2-builds-on-prior\n"
            "  title: Builds on Prior\n"
            "  prompt: |\n"
            "    no-op\n"
        ),
        encoding="utf-8",
    )
    ledger = FakeLedger(
        {"exp2-builds-on-prior": [_prior("exp1-successful-upstream", "complete")]}
    )

    summary = mod.autofill_roadmap(roadmap, dry_run=False, ledger=ledger)
    data = yaml.safe_load(roadmap.read_text(encoding="utf-8"))
    prior_failures = data["tasks"][0]["prior_failures"]

    assert summary.stubs_generated == 1
    assert prior_failures[0]["experiment_id"] == "exp1-successful-upstream"
    assert prior_failures[0]["classification"] == "successful_upstream"
    assert "Autofilled" in prior_failures[0]["addressed_by"]
    assert prior_failures[0]["retire_if_same_verdict"] is False


def test_dry_run_produces_no_file_changes(tmp_path: Path) -> None:
    """SCENARIO-INFRA-094: dry-run reports stubs without changing bytes on disk."""
    mod = _load_module()
    roadmap = tmp_path / "research-roadmap-next.yaml"
    original = _roadmap_text(
        "- id: exp2-builds-on-prior\n"
        "  title: Builds on Prior\n"
        "  prompt: |\n"
        "    no-op\n"
    )
    roadmap.write_text(original, encoding="utf-8")
    ledger = FakeLedger(
        {"exp2-builds-on-prior": [_prior("exp1-successful-upstream", "success")]}
    )

    summary = mod.autofill_roadmap(roadmap, dry_run=True, ledger=ledger)

    assert summary.stubs_generated == 1
    assert roadmap.read_text(encoding="utf-8") == original
