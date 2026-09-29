"""REQ-REPORT-7874: only post-snapshot live supervisor evidence earns credit."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys

from carnot.reporting.arc_supervisor_v683_delta import check_inputs, summarize
from scripts.experiments.experiment_7874_v683_arc_supervisor_delta import commands


def _write(path: Path, value: object) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _receipt(attempt: str, provenance: str = "live_agent_self_discovery") -> dict:
    return {
        "game": "g1",
        "seed": 7,
        "attempt": attempt,
        "level": 0,
        "levels_completed": 1,
        "solve_provenance": provenance,
        "termination": {"reason": "level_up"},
        "trajectory_supervisor": {
            "mode": "applied",
            "enabled": True,
            "redirects": [
                {
                    "id": attempt,
                    "arm": "drop_goal_bias",
                    "fired": True,
                    "helped": True,
                    "resolved_by_levelup": True,
                    "actions_to_levelup": 4,
                }
            ],
        },
    }


def _producer(root: Path, rows: list[dict], *, tamper: bool = False) -> Path:
    raw = root / "results/raw/experiment_9001_v683_arc/rows.json"
    digest = _write(raw, {"rows": rows})
    _write(
        root / "results/experiment_9001_v683_arc.json",
        {
            "run_date": "20260929",
            "verdict_class": "null",
            "flagged_adversarial": False,
            "source_artifact_hashes": {
                raw.relative_to(root).as_posix(): "sha256:wrong" if tamper else digest
            },
        },
    )
    return raw


# SCENARIO-REPORT-7874-DELTA: duplicate, stale, and off-path rows cannot add N.
def test_delta_filters_duplicate_stale_and_off_path(tmp_path: Path) -> None:
    raw = _producer(
        tmp_path,
        [_receipt("a"), _receipt("a"), _receipt("b", "development_proxy")],
    )
    result = summarize(tmp_path, 0, set(), {"g1": 1})
    assert result["new_live_outcome_count"] == 1
    assert result["new_level_solves"] == 0
    assert result["per_game_results"]["g1"]["eligible"] == 1
    assert {r.get("reason") for r in result["outcome_rows"]} >= {
        "duplicate_attempt",
        "non_live_provenance",
    }
    old = summarize(tmp_path, raw.stat().st_mtime_ns + 1, set(), {"g1": 0})
    assert old["new_live_outcome_count"] == 0
    assert any(r.get("reason") == "before_cutoff" for r in old["outcome_rows"])


# SCENARIO-REPORT-7874-DELTA: a live level above the registry needs a mechanism hash.
def test_new_level_and_regression_rows(tmp_path: Path) -> None:
    win = _receipt("new")
    win["levels_completed"] = 2
    win["mechanism_hash"] = "sha256:mechanism"
    win["live_route_reachable"] = True
    loss = _receipt("loss")
    loss["trajectory_supervisor"]["redirects"][0].update(helped=False, resolved_by_levelup=False)
    _producer(tmp_path, [win, loss])
    delta = summarize(tmp_path, 0, set(), {"g1": 1})
    assert delta["new_level_solves"] == 1
    assert delta["per_game_results"]["g1"]["regressions"] == 1
    assert delta["per_game_results"]["g1"]["lost_wins"] == 1
    assert delta["per_game_results"]["g1"]["mechanism_hashes"] == ["sha256:mechanism"]


# SCENARIO-REPORT-7874-EMPTY: no source means a real zero, and a changed hash stays excluded.
def test_empty_and_tampered_source(tmp_path: Path) -> None:
    empty = summarize(tmp_path, 0, set(), {})
    assert empty["no_new_outcomes"] is True
    assert empty["new_live_outcome_count"] == 0
    assert empty["per_game_results"] == {}
    _producer(tmp_path, [_receipt("a")], tamper=True)
    bad = summarize(tmp_path, 0, set(), {})
    assert bad["new_live_outcome_count"] == 0
    assert any(r.get("reason") == "raw_hash_mismatch" for r in bad["outcome_rows"])


# SCENARIO-REPORT-7874-GATE: exact bytes and schema are checked before a delta.
def test_preconditions_fail_closed(tmp_path: Path) -> None:
    prior = tmp_path / "prior.json"
    source = tmp_path / "source.py"
    registry = tmp_path / "registry.yaml"
    prior_hash = _write(prior, {"verdict_class": "null", "source_artifact_hashes": {}})
    source.write_text("pass\n", encoding="utf-8")
    source_hash = "sha256:" + hashlib.sha256(source.read_bytes()).hexdigest()
    registry.write_text("schema_version: 1\ngames:\n  g1:\n    levels_reproduced: 1\n")
    registry_hash = "sha256:" + hashlib.sha256(registry.read_bytes()).hexdigest()
    checked = check_inputs(prior, source, registry, (prior_hash, source_hash, registry_hash))
    assert checked["failures"] == []
    assert checked["registry_levels"] == {"g1": 1}
    assert checked["cutoff_ns"] >= prior.stat().st_mtime_ns
    wrong = check_inputs(prior, source, registry, ("sha256:wrong", source_hash, registry_hash))
    assert wrong["failures"][0]["artifact_field"] == "sha256"
    assert wrong["failures"][0]["observed"] == prior_hash
    registry.write_text("schema_version: 2\ngames: []\n")
    new_hash = "sha256:" + hashlib.sha256(registry.read_bytes()).hexdigest()
    schema = check_inputs(prior, source, registry, (prior_hash, source_hash, new_hash))
    assert schema["failures"][0]["artifact_field"] == "schema_version"


# SCENARIO-REPORT-7874-EMPTY: the real CLI emits and replays a private zero delta.
def test_cli_success_and_failure(tmp_path: Path) -> None:
    cli = "scripts/experiments/experiment_7874_v683_arc_supervisor_delta.py"
    output = tmp_path / "delta.json"
    command = [
        sys.executable,
        cli,
        "--reduce-ledger",
        str(tmp_path),
        "--cutoff-ns",
        "0",
        "--output",
        str(output),
    ]
    good = subprocess.run(command, capture_output=True, text=True, check=False)
    assert good.returncode == 0, good.stdout + good.stderr
    assert json.loads(output.read_text())["new_live_outcome_count"] == 0
    bad = subprocess.run(command[:-2], capture_output=True, text=True, check=False)
    assert bad.returncode != 0
    candidate = tmp_path / "candidate.json"
    _write(candidate, {"validation_receipts": [], "outcome_rows": [], "firings": 0})
    replay = subprocess.run(
        [sys.executable, cli, "--cold-replay", str(candidate)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert replay.returncode == 0
    _write(candidate, {"validation_receipts": [], "outcome_rows": [], "firings": 1})
    failed = subprocess.run(
        [sys.executable, cli, "--cold-replay", str(candidate)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert failed.returncode != 0


# SCENARIO-REPORT-7874-GATE: frozen required commands name the real V683 closure.
def test_validation_manifest_paths(tmp_path: Path) -> None:
    fixed = commands(tmp_path, 123)
    affected = next(row for row in fixed if row["name"] == "affected_pytest")
    assert "tests/python/test_arc_supervisor_delta_7874.py" in affected["argv"]
    assert all(row["classification"] == "required" for row in fixed)
