"""Tests for Exp 5164 standalone retro timing fallback.

Spec refs: REQ-REPORT-5164, SCENARIO-REPORT-5164-PRIMARY,
SCENARIO-REPORT-5164-MTIME, SCENARIO-REPORT-5164-BOUNDARY,
SCENARIO-REPORT-5164-FALSE-ZERO.
"""

from __future__ import annotations

from datetime import datetime
import importlib.util
import json
import os
from pathlib import Path
from typing import Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO_ROOT / "openspec/capabilities/research-reporting/spec.md"
MODULE_PATH = REPO_ROOT / "scripts/retro_timing_fallback.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("retro_timing_fallback", MODULE_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _touch_iso(path: Path, iso_text: str) -> None:
    timestamp = datetime.fromisoformat(iso_text).timestamp()
    os.utime(path, (timestamp, timestamp))


class FakeGit:
    def __init__(self, activation: str, path_times: dict[str, str], legacy_log: str = ""):
        self.activation = activation
        self.path_times = path_times
        self.legacy_log = legacy_log
        self.calls: list[tuple[str, ...]] = []

    def __call__(self, args: Sequence[str], cwd: Path) -> str:
        self.calls.append(tuple(args))
        if "--grep=\\[conductor\\] Activate milestone 2026.07.999" in args:
            return f"abc123 {self.activation}\n"
        if "--grep=\\[conductor\\]" in args:
            return self.legacy_log
        if "--format=%ai" in args and "--" in args:
            rel_path = args[-1]
            return f"{self.path_times.get(rel_path, '')}\n"
        return ""


def test_req_report_5164_spec_anchor_exists() -> None:
    """REQ-REPORT-5164: OpenSpec declares the standalone fallback contract."""

    spec_text = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-REPORT-5164" in spec_text
    assert "SCENARIO-REPORT-5164-PRIMARY" in spec_text
    assert "SCENARIO-REPORT-5164-MTIME" in spec_text
    assert "SCENARIO-REPORT-5164-BOUNDARY" in spec_text
    assert "SCENARIO-REPORT-5164-FALSE-ZERO" in spec_text
    assert "scripts/retro_timing_fallback.py" in spec_text


def test_scenario_report_5164_primary_self_reported_fields_rank_slowest(
    tmp_path: Path,
) -> None:
    """SCENARIO-REPORT-5164-PRIMARY: duration_s and substrate drive rows."""

    mod = _load_module()
    tasks = [
        {
            "id": "exp1",
            "title": "live task",
            "deliverable": "results/experiment_1_live.json",
        },
        {
            "id": "exp2",
            "title": "summary task",
            "deliverable": "results/experiment_2_summary.json",
        },
    ]
    _write_json(
        tmp_path / "results/experiment_1_live.json",
        {"duration_s": 120.0, "inference_substrate": "live_llm_inference"},
    )
    _write_json(
        tmp_path / "results/experiment_2_summary.json",
        {"duration_s": 30.0, "inference_substrate": "aggregation_from_upstream_artifacts"},
    )
    fake_git = FakeGit(
        activation="2026-07-01 10:00:00 +0000",
        path_times={
            "results/experiment_1_live.json": "2026-07-01 10:01:00 +0000",
            "results/experiment_2_summary.json": "2026-07-01 10:04:00 +0000",
        },
    )

    summary = mod.build_retro_timing_fallback(
        "2026.07.999", tasks=tasks, repo_root=tmp_path, git_runner=fake_git
    )

    assert summary["experiments_completed"] == 2
    assert summary["compute_bound_experiments_count"] == 1
    assert summary["total_wall_time_minutes"] == 3.0
    assert summary["gpu_idle_on_compute_bound_tasks"] is False
    assert summary["slowest_experiments"][0]["experiment"] == "live task"
    assert summary["slowest_experiments"][0]["duration_minutes"] == 2.0
    assert summary["experiment_times"][0]["duration_source"] == "self_reported"


def test_scenario_report_5164_git_timestamp_fallback_precedes_filesystem_mtime(
    tmp_path: Path,
) -> None:
    """SCENARIO-REPORT-5164-MTIME: git timestamps repair missing fields."""

    mod = _load_module()
    tasks = [
        {"id": "exp1", "title": "fallback one", "deliverable": "results/experiment_10_one.json"},
        {"id": "exp2", "title": "fallback two", "deliverable": "results/experiment_11_two.json"},
    ]
    for task in tasks:
        path = tmp_path / task["deliverable"]
        _write_json(path, {"honest_verdict": "complete: fixture"})
        _touch_iso(path, "2026-07-01T12:00:00+00:00")
    fake_git = FakeGit(
        activation="2026-07-01 10:00:00 +0000",
        path_times={
            "results/experiment_10_one.json": "2026-07-01 10:05:00 +0000",
            "results/experiment_11_two.json": "2026-07-01 10:15:00 +0000",
        },
    )

    summary = mod.build_retro_timing_fallback(
        "2026.07.999", tasks=tasks, repo_root=tmp_path, git_runner=fake_git
    )

    assert summary["experiments_completed"] == 2
    assert summary["total_wall_time_minutes"] == 10.0
    assert summary["timestamp_sources"] == {
        "results/experiment_10_one.json": "git_log",
        "results/experiment_11_two.json": "git_log",
    }
    assert summary["experiment_times"][1]["duration_minutes"] == 10.0
    assert any("--format=%ai" in call for call in fake_git.calls)


def test_scenario_report_5164_activation_boundary_excludes_earlier_artifact(
    tmp_path: Path,
) -> None:
    """SCENARIO-REPORT-5164-BOUNDARY: pre-activation fallback rows are excluded."""

    mod = _load_module()
    tasks = [
        {"id": "before", "title": "before", "deliverable": "results/experiment_20_before.json"},
        {"id": "after", "title": "after", "deliverable": "results/experiment_21_after.json"},
    ]
    for task in tasks:
        _write_json(tmp_path / task["deliverable"], {"honest_verdict": "complete: fixture"})
    fake_git = FakeGit(
        activation="2026-07-01 10:00:00 +0000",
        path_times={
            "results/experiment_20_before.json": "2026-07-01 09:59:00 +0000",
            "results/experiment_21_after.json": "2026-07-01 10:03:00 +0000",
        },
    )

    summary = mod.build_retro_timing_fallback(
        "2026.07.999", tasks=tasks, repo_root=tmp_path, git_runner=fake_git
    )

    assert summary["experiments_completed"] == 1
    assert [row["deliverable"] for row in summary["experiment_times"]] == [
        "results/experiment_21_after.json"
    ]
    assert summary["excluded_pre_activation"] == ["results/experiment_20_before.json"]


def test_scenario_report_5164_synthetic_false_zero_reconstructs_without_exp_subject(
    tmp_path: Path,
) -> None:
    """SCENARIO-REPORT-5164-FALSE-ZERO: legacy literal Exp miss still reconstructs."""

    mod = _load_module()
    tasks = [
        {"id": "exp1", "title": "PHASE A1 live", "deliverable": "results/experiment_30_live.json"},
        {"id": "exp2", "title": "PHASE B1 audit", "deliverable": "results/experiment_31_audit.json"},
    ]
    _write_json(
        tmp_path / "results/experiment_30_live.json",
        {"duration_s": 60.0, "inference_substrate": "live_llm_inference"},
    )
    _write_json(
        tmp_path / "results/experiment_31_audit.json",
        {"duration_s": 1.0, "inference_substrate": "aggregation_from_upstream_artifacts"},
    )
    fake_git = FakeGit(
        activation="2026-07-01 10:00:00 +0000",
        path_times={
            "results/experiment_30_live.json": "2026-07-01 10:05:00 +0000",
            "results/experiment_31_audit.json": "2026-07-01 10:09:00 +0000",
        },
        legacy_log=(
            "bbb222 2026-07-01 10:09:00 +0000 [conductor] PHASE B1 audit\n"
            "aaa111 2026-07-01 10:05:00 +0000 [conductor] exp30 lower-case identity\n"
        ),
    )

    legacy_count = mod.legacy_literal_exp_subject_count(
        "2026.07.999", repo_root=tmp_path, git_runner=fake_git
    )
    summary = mod.build_retro_timing_fallback(
        "2026.07.999", tasks=tasks, repo_root=tmp_path, git_runner=fake_git
    )

    assert legacy_count == 0
    assert summary["experiments_completed"] == 2
    assert summary["total_wall_time_minutes"] == 4.0
    assert summary["compute_bound_experiments_count"] == 1


def test_req_report_5164_real_m450_reconstructs_known_good() -> None:
    """REQ-REPORT-5164: real .450 reconstructs non-zero timing and 4 compute arms."""

    mod = _load_module()

    summary = mod.build_retro_timing_fallback("2026.06.450", repo_root=REPO_ROOT)

    assert summary["experiments_completed"] == 10
    assert 200.0 <= summary["total_wall_time_minutes"] <= 225.0
    assert summary["compute_bound_experiments_count"] == 4
    assert summary["known_good_checks"]["m450_reconstruction_correct"] is True
