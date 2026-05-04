"""Tests for conductor prior-failure autofill.

Spec: REQ-INFRA-078, SCENARIO-INFRA-092, SCENARIO-INFRA-093,
      SCENARIO-INFRA-094
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
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


def test_script_classifies_true_failures_and_preserves_insert_shape(tmp_path: Path) -> None:
    """REQ-INFRA-078: partial/failed tokens become review-needed true failures."""
    mod = _load_module()
    roadmap = tmp_path / "research-roadmap-next.yaml"
    roadmap.write_text(
        _roadmap_text(
            "- id: exp3-failed-retry\n"
            "  title: Failed Retry\n"
            "  prior_failures: []\n"
            "  prompt: |\n"
            "    - id: nested-prompt-text\n"
            "- id: exp4-no-prompt\n"
            "  title: No Prompt\n"
        )
        + "\n",
        encoding="utf-8",
    )
    ledger = FakeLedger(
        {
            "exp3-failed-retry": [
                {"experiment_id": "exp0-no-improvement", "verdict": "no_improvement"}
            ],
            "exp4-no-prompt": [_prior("exp0-failed", "failed")],
        }
    )

    summary = mod.autofill_roadmap(roadmap, dry_run=False, ledger=ledger)
    written = roadmap.read_text(encoding="utf-8")
    data = yaml.safe_load(written)
    prior_failures = data["tasks"][0]["prior_failures"]

    assert summary.stubs_generated == 2
    assert written.endswith("\n")
    assert written.count("prior_failures:") == 2
    assert prior_failures[0]["classification"] == "true_failure"
    assert prior_failures[0]["addressed_by"].startswith("REVIEW NEEDED:")
    assert prior_failures[0]["retire_if_same_verdict"] is True


def test_default_path_and_cli_use_loaded_ledger(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """REQ-INFRA-078: CLI defaults to next-roadmap, then active roadmap fallback."""
    mod = _load_module()
    next_path = tmp_path / "research-roadmap-next.yaml"
    active_path = tmp_path / "research-roadmap.yaml"
    next_path.write_text("tasks: []\n", encoding="utf-8")
    active_path.write_text("tasks:\n- just-a-string\n", encoding="utf-8")
    load_calls: list[Path] = []

    class LoadingLedger:
        @classmethod
        def load_from_artifacts(cls, repo_root: Path) -> FakeLedger:
            load_calls.append(repo_root)
            return FakeLedger()

    monkeypatch.setattr(mod, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(mod, "FailureLedger", LoadingLedger)

    assert mod.default_roadmap_path(tmp_path) == next_path
    next_path.unlink()
    assert mod.default_roadmap_path(tmp_path) == active_path
    assert mod.main(["--dry-run"]) == 0
    dry_run_output = capsys.readouterr().out
    assert "1 tasks scanned, 0 stubs generated, 0 already populated" in dry_run_output
    assert "dry-run: no file changes" in dry_run_output
    assert mod.main([]) == 0
    write_output = capsys.readouterr().out
    assert "1 tasks scanned, 0 stubs generated, 0 already populated" in write_output
    assert load_calls == [tmp_path, tmp_path]


def test_invalid_roadmap_shapes_raise(tmp_path: Path) -> None:
    """REQ-INFRA-078: malformed task lists fail clearly instead of rewriting."""
    mod = _load_module()
    roadmap = tmp_path / "research-roadmap-next.yaml"
    roadmap.write_text("tasks: not-a-list\n", encoding="utf-8")

    with pytest.raises(ValueError, match="list-valued tasks"):
        mod.autofill_roadmap(roadmap, dry_run=True, ledger=FakeLedger())

    with pytest.raises(ValueError, match="could not locate roadmap text"):
        mod._apply_insertions("tasks: []\n", [(0, [])])

    assert mod._yaml_scalar(None) == "null"
    no_newline = mod._apply_insertions(
        "tasks:\n- id: exp5-no-newline\n  title: No Newline",
        [
            (
                0,
                [
                    {
                        "experiment_id": "exp0",
                        "verdict": "success",
                        "classification": "successful_upstream",
                        "addressed_by": "covered",
                        "retire_if_same_verdict": False,
                    }
                ],
            )
        ],
    )
    assert not no_newline.endswith("\n")
