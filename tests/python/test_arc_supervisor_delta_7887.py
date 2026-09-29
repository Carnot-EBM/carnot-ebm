"""REQ-REPORT-7887: explicit V684 validation and honest live receipts."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pytest

from carnot.reporting.arc_supervisor_v684_delta import aggregate, build_manifest, cold_replay
from scripts.experiments import experiment_7887_v684_arc_supervisor_delta as cli


# SCENARIO-REPORT-7887-MANIFEST: only actual test and implementation files enter argv.
def test_manifest_names_real_paths(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[2]
    manifest = build_manifest(root, tmp_path, 123)
    names = {row["name"] for row in manifest}
    assert {"affected_pytest", "unit_coverage", "cli_success_coverage", "coverage_report"} <= names
    affected = next(row for row in manifest if row["name"] == "affected_pytest")
    assert "tests/python/test_arc_supervisor_delta_7874.py" in affected["argv"]
    assert "tests/python/test_arc_supervisor_delta_7887.py" in affected["argv"]
    assert all(
        (root / path).is_file()
        for path in (
            "python/carnot/reporting/arc_supervisor_v684_delta.py",
            "scripts/experiments/experiment_7887_v684_arc_supervisor_delta.py",
        )
    )
    assert all(row["classification"] in {"required", "diagnostic"} for row in manifest)
    assert all(row["deadline_s"] > 0 for row in manifest)
    assert not any(
        "test_arc_supervisor_v683_delta_7860.py" in arg for row in manifest for arg in row["argv"]
    )


# SCENARIO-REPORT-7887-MANIFEST: an absent affected source fails before dispatch.
def test_manifest_missing_path(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        build_manifest(tmp_path, tmp_path, 0)


# SCENARIO-REPORT-7887-LEDGER: per-arm metrics come from primitive eligible rows.
def test_aggregate_live_rows_and_empty() -> None:
    empty = aggregate(
        {
            "outcome_rows": [],
            "new_live_outcome_count": 0,
            "firings": 0,
            "new_level_solves": 0,
            "sample_size_budget": {"eligible": 0},
        }
    )
    assert empty["no_new_outcomes"] is True
    assert empty["per_game_results"] == {}
    assert empty["recommendation_rows"] == []
    rows = [
        {
            "status": "completed",
            "game": "g1",
            "seed": 1,
            "arm": "drop_goal_bias",
            "fired": True,
            "helped": True,
            "resolved_by_levelup": True,
            "actions_to_levelup": 4,
            "stagnations_unredirected": 2,
            "solve_provenance": "live_agent_self_discovery",
        },
        {
            "status": "censored",
            "game": "g1",
            "seed": 2,
            "arm": "drop_goal_bias",
            "fired": True,
            "helped": False,
            "resolved_by_levelup": False,
            "actions_to_levelup": None,
            "stagnations_unredirected": 3,
            "solve_provenance": "live_agent_self_discovery",
        },
        {
            "status": "excluded",
            "game": "g1",
            "seed": 3,
            "arm": "drop_goal_bias",
            "fired": True,
            "helped": True,
            "solve_provenance": "development_proxy",
        },
    ]
    result = aggregate(
        {
            "outcome_rows": rows,
            "new_live_outcome_count": 2,
            "firings": 2,
            "new_level_solves": 0,
            "sample_size_budget": {"eligible": 2},
        }
    )
    arm = result["per_game_results"]["g1"]["arms"]["drop_goal_bias"]
    assert (arm["firings"], arm["helped"], arm["regressions"]) == (2, 1, 1)
    assert arm["actions_to_levelup"] == [4]
    assert arm["stagnations_unredirected"] == [2, 3]
    assert result["recommendation_rows"] == []  # fewer than ten new firings


# SCENARIO-REPORT-7887-CLI: direct script success, missing args, forged firing, replay.
def test_direct_cli_empty_and_forged_replay(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[2]
    cli = root / "scripts/experiments/experiment_7887_v684_arc_supervisor_delta.py"
    output = tmp_path / "empty.json"
    argv = [
        sys.executable,
        str(cli),
        "--reduce-ledger",
        str(tmp_path),
        "--cutoff-ns",
        "0",
        "--output",
        str(output),
    ]
    good = subprocess.run(argv, capture_output=True, text=True, check=False, cwd=root)
    assert good.returncode == 0, good.stdout + good.stderr
    assert json.loads(output.read_text())["new_live_outcome_count"] == 0
    missing = subprocess.run(argv[:-2], capture_output=True, text=True, check=False, cwd=root)
    assert missing.returncode != 0
    candidate = tmp_path / "candidate.json"
    candidate.write_text(json.dumps({"validation_receipts": [], "outcome_rows": [], "firings": 0}))
    replay = subprocess.run(
        [sys.executable, str(cli), "--cold-replay", str(candidate)],
        capture_output=True,
        text=True,
        check=False,
        cwd=root,
    )
    assert replay.returncode == 0
    candidate.write_text(json.dumps({"validation_receipts": [], "outcome_rows": [], "firings": 1}))
    assert cold_replay(candidate) == ["primitive_firing_count"]
    forged = subprocess.run(
        [sys.executable, str(cli), "--cold-replay", str(candidate)],
        capture_output=True,
        text=True,
        check=False,
        cwd=root,
    )
    assert forged.returncode != 0


# SCENARIO-REPORT-7887-CLI: normal orchestration freezes a real manifest before reduction.
@pytest.mark.parametrize("failure", [None, "ruff_check", "terminal_adversarial", "blocked"])
def test_main_terminal_states(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str | None
) -> None:
    private = tmp_path / "private"
    private.mkdir()
    monkeypatch.setattr(cli.tempfile, "mkdtemp", lambda **_kwargs: str(private))
    checked = cli.precheck()
    if failure == "blocked":
        checked["failures"] = [
            {
                "upstream_id": "missing",
                "path": str(tmp_path / "missing"),
                "artifact_field": "sha256",
                "op": "==",
                "expected": "sha256:expected",
                "observed": "missing",
            }
        ]
    monkeypatch.setattr(cli, "precheck", lambda: checked)
    delta = {
        "outcome_rows": [],
        "new_live_outcome_count": 0,
        "firings": 0,
        "new_level_solves": 0,
        "sample_size_budget": {
            "intended": 0,
            "eligible": 0,
            "started": 0,
            "completed": 0,
            "censored": 0,
            "excluded": 0,
            "independent_n": 0,
        },
    }
    monkeypatch.setattr(cli, "summarize", lambda *_args: delta.copy())
    calls: list[str] = []

    def fake_run(_private: Path, row: dict, _started: float) -> dict:
        calls.append(row["name"])
        passed = row["name"] != failure
        return {
            "name": row["name"],
            "class": row["classification"],
            "passed": passed,
            "exit_code": 0 if passed else 1,
            "command_argv": row["argv"],
            "log_path": str(tmp_path / "log"),
            "log_sha256": "sha256:test",
        }

    monkeypatch.setattr(cli, "_run", fake_run)
    output = tmp_path / "output.json"
    rc = cli.main(["--date", "20260929", "--output", str(output)])
    result = json.loads(output.read_text())
    assert rc == (0 if failure is None else 1)
    assert result["new_live_outcome_count"] == 0
    assert result["honest_verdict"].startswith("complete_")
    assert result["arc_delta_ready_score"] == (1 if failure is None else 0)
    if failure == "blocked":
        assert calls == []
        assert result["gate_check_summary"][0]["observed"] == "missing"
    else:
        assert "affected_pytest" in calls
        assert "terminal_rows" in calls
        assert result["validation_command_manifest_path"]


# SCENARIO-REPORT-7887-GATE: a missing live policy declaration blocks before outcomes.
def test_precheck_missing_reachability(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli, "AGENT", tmp_path / "absent.py")
    monkeypatch.setattr(
        cli,
        "check_inputs",
        lambda *_args: {
            "checks": [],
            "failures": [],
            "cutoff_ns": 0,
            "prior_hashes": set(),
            "registry_levels": {},
        },
    )
    result = cli.precheck()
    assert len(result["failures"]) == 3
    assert all(row["observed"] is False for row in result["failures"])


# SCENARIO-REPORT-7887-CLI: private modes and missing required CLI operands.
def test_main_private_modes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    delta = {
        "outcome_rows": [],
        "new_live_outcome_count": 0,
        "firings": 0,
        "new_level_solves": 0,
        "sample_size_budget": {"eligible": 0},
    }
    monkeypatch.setattr(cli, "summarize", lambda *_args: delta.copy())
    output = tmp_path / "reduced.json"
    assert (
        cli.main(["--reduce-ledger", str(tmp_path), "--cutoff-ns", "0", "--output", str(output)])
        == 0
    )
    assert json.loads(output.read_text())["no_new_outcomes"] is True
    with pytest.raises(SystemExit):
        cli.main(["--reduce-ledger", str(tmp_path), "--cutoff-ns", "0"])
    candidate = tmp_path / "candidate.json"
    candidate.write_text(json.dumps({"validation_receipts": [], "outcome_rows": [], "firings": 0}))
    assert cli.main(["--cold-replay", str(candidate)]) == 0
    candidate.write_text(json.dumps({"validation_receipts": [], "outcome_rows": [], "firings": 1}))
    assert cli.main(["--cold-replay", str(candidate)]) == 1


# SCENARIO-REPORT-7887-GATE: owned children seal the exact closed log bytes.
def test_child_log_seal_and_collision(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    log = tmp_path / "child.log"
    log.write_text("child complete\n")

    def fake_commands(_root: Path, _specs: list, **_kwargs: object) -> list[dict]:
        return [
            {
                "name": "probe",
                "passed": True,
                "exit_code": 0,
                "log_path": str(log),
                "log_sha256": "ignored",
            }
        ]

    monkeypatch.setattr(cli, "run_commands", fake_commands)
    row = {
        "name": "probe",
        "argv": [sys.executable, "-V"],
        "classification": "required",
        "deadline_s": 10,
    }
    first = cli._run(tmp_path, row, 0.0)
    assert first["log_sha256"].startswith("sha256:")
    assert Path(first["log_path"]).read_bytes() == log.read_bytes()
    second = cli._run(tmp_path, row, 0.0)
    assert second["log_path"] == first["log_path"]
    Path(first["log_path"]).write_text("tampered")
    with pytest.raises(ValueError, match="sealed_log_collision"):
        cli._run(tmp_path, row, 0.0)
