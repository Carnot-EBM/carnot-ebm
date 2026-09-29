"""REQ-REPORT-7860: a new receipt is evidence only after the frozen cutoff."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

from carnot.reporting.arc_supervisor_receipt_delta import reduce_ledger


def _json(path: Path, value: object) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _producer(root: Path, number: int, rows: list[dict]) -> Path:
    raw = root / f"results/raw/experiment_{number}_v682_arc/rows.json"
    digest = _json(raw, {"rows": rows})
    _json(
        root / f"results/experiment_{number}_v682_arc.json",
        {
            "run_date": "20260929",
            "verdict_class": "null",
            "flagged_adversarial": False,
            "source_artifact_hashes": {raw.relative_to(root).as_posix(): digest},
        },
    )
    return raw


def _row(attempt: str, *, provenance: str = "live_agent_self_discovery") -> dict:
    return {
        "game": "g1",
        "seed": 7,
        "attempt": attempt,
        "solve_provenance": provenance,
        "termination": {"reason": "level_up"},
        "trajectory_supervisor": {
            "mode": "applied",
            "enabled": True,
            "stagnations_unredirected": 2,
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


# SCENARIO-REPORT-7860-LEDGER: duplicate source views and non-live provenance add no N.
def test_reordered_duplicate_and_provenance(tmp_path: Path) -> None:
    first = _row("a")
    second = _row("b")
    del second["trajectory_supervisor"]["redirects"][0]["helped"]
    _producer(tmp_path, 9001, [second, first, first, _row("c", provenance="outer_loop_re")])
    result = reduce_ledger(tmp_path, 0, set())
    assert [r["attempt"] for r in result["outcome_rows"] if r["status"] == "completed"] == [
        "a",
        "b",
    ]
    assert any(r.get("reason") == "duplicate_attempt" for r in result["outcome_rows"])
    assert any(r.get("reason") == "non_live_provenance" for r in result["outcome_rows"])
    assert result["firings"] == 2
    assert next(r for r in result["outcome_rows"] if r.get("attempt") == "b")["helped"] is None
    assert result["new_level_solves"] == 0
    output = tmp_path / "e2e-delta.json"
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/experiments/experiment_7860_v682_arc_supervisor_delta.py",
            "--reduce-ledger",
            str(tmp_path),
            "--cutoff-ns",
            "0",
            "--output",
            str(output),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert json.loads(output.read_text())["firings"] == result["firings"]


# SCENARIO-REPORT-7860-EMPTY: old and missing receipts cannot manufacture activity.
def test_cutoff_and_empty(tmp_path: Path) -> None:
    raw = _producer(tmp_path, 9002, [_row("old")])
    result = reduce_ledger(tmp_path, raw.stat().st_mtime_ns + 1, set())
    assert result["no_new_outcomes"] is True
    assert result["firings"] == 0
    assert result["recommendation_rows"] == []
    assert any(r.get("reason") == "before_cutoff" for r in result["outcome_rows"])
    assert (
        reduce_ledger(tmp_path, 0, {"sha256:" + hashlib.sha256(raw.read_bytes()).hexdigest()})[
            "firings"
        ]
        == 0
    )


# SCENARIO-REPORT-7860-INPUT: unknown metrics and malformed input remain visible.
def test_malformed_receipt_and_cli(tmp_path: Path) -> None:
    bad = _row("bad")
    del bad["trajectory_supervisor"]["redirects"][0]["actions_to_levelup"]
    _producer(tmp_path, 9003, [bad])
    result = reduce_ledger(tmp_path, 0, set())
    assert result["firings"] == 0
    assert any(r.get("reason") == "malformed_redirect" for r in result["outcome_rows"])
    cli = Path("scripts/experiments/experiment_7860_v682_arc_supervisor_delta.py")
    output = tmp_path / "delta.json"
    command = [
        sys.executable,
        str(cli),
        "--reduce-ledger",
        str(tmp_path),
        "--cutoff-ns",
        "0",
        "--output",
        str(output),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert json.loads(output.read_text())["firings"] == 0
    invalid = subprocess.run(command[:-3] + ["bad"], capture_output=True, text=True, check=False)
    assert invalid.returncode != 0


# SCENARIO-REPORT-7860-LEDGER: an independent cold reduction matches primitive rows.
def test_cold_counts(tmp_path: Path) -> None:
    _producer(tmp_path, 9004, [_row("one"), _row("two")])
    result = reduce_ledger(tmp_path, 0, set())
    assert result["firings"] == sum(
        row["fired"] is True for row in result["outcome_rows"] if row["status"] == "completed"
    )
    assert result["sample_size_budget"]["independent_n"] == 2


# SCENARIO-REPORT-7860-INPUT: a corrupt raw document cannot supply original keys.
def test_original_lookup_rejects_bad_shapes(tmp_path: Path) -> None:
    from carnot.reporting.arc_supervisor_receipt_delta import _receipt_lookup

    raw = tmp_path / "rows.json"
    raw.write_text("broken json", encoding="utf-8")
    assert _receipt_lookup(raw) == {}
    _json(
        raw,
        {
            "rows": [
                {"game": "g", "trajectory_supervisor": "bad"},
                {"game": "g", "trajectory_supervisor": {"redirects": "bad"}},
                {"game": "g", "trajectory_supervisor": {"redirects": ["bad", {"id": "ok"}]}},
            ]
        },
    )
    assert len(_receipt_lookup(raw)) == 1


# SCENARIO-REPORT-7860-INPUT: an authenticated-screen race or mismatched ID excludes rows.
def test_missing_source_and_original_attempt(tmp_path: Path, monkeypatch) -> None:
    from carnot.reporting import arc_supervisor_receipt_delta as delta

    missing = {
        "source_path": "results/raw/missing.json",
        "producer_path": "results/experiment_9005_v682_arc.json",
        "status": "completed",
        "game": "g1",
        "seed": 7,
        "redirect_id": "x",
    }
    monkeypatch.setattr(delta, "inspect_sources", lambda *_: [missing])
    assert (
        delta.reduce_ledger(tmp_path, 0, set())["outcome_rows"][0]["reason"]
        == "missing_authenticated_source"
    )
    raw = _producer(tmp_path, 9005, [_row("real")])
    mismatched = dict(missing, source_path=raw.relative_to(tmp_path).as_posix())
    monkeypatch.setattr(delta, "inspect_sources", lambda *_: [mismatched])
    assert (
        delta.reduce_ledger(tmp_path, 0, set())["outcome_rows"][0]["reason"]
        == "missing_original_attempt"
    )
