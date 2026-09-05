"""Spec: REQ-ARC-WMTE-7012, SCENARIO-ARC-WMTE-7012-A, SCENARIO-ARC-WMTE-7012-B,
SCENARIO-ARC-WMTE-7012-C, SCENARIO-ARC-WMTE-7012-D.

The refinement ledger ingests live-eval artifacts.

INCIDENT 2026-09-04. `ops/arc_supervisor_refinement_ledger.json` had not moved since
2026-08-27 (9 entries, 6 redirects) while five applied eval rows in
`results/arc_leaderboard_eval_runs/` held 25 redirects. The tool read only harness `rows.json`
documents; the eval writes `{"per_game": [...]}`. A consumer that cannot read its producer is
the same class as the supervisor-receipt gap of 2026-09-01.

Every test writes only under tmp_path.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from carnot.agentic.arc_supervisor_refinement import (
    EVAL_RUN_FIELDS_READ,
    EVAL_RUNS_DIR_NAME,
    empty_ledger,
    evaluate,
    extract_rows,
    ingest_files,
    scan_inputs,
)
from carnot.agentic.arc_trajectory_supervisor import ARM_DROP_GOAL_BIAS

NOW = "2026-09-04T00:00:00+00:00"


def _eval_row(game: str = "r11l", *, k: int | None = 3) -> dict[str, Any]:
    return {
        "game": game,
        "levels": 1,
        "actions": 2077,
        "frame_sequence": [{"loop_index": 0}],
        "trajectory_supervisor": {
            "enabled": True,
            "mode": "applied",
            "window": 120,
            "actions_observed": 2188,
            "arms_used": [],
            "arm_outcomes": {},
            "stagnations_unredirected": 12,
            "redirects": [
                {
                    "arm": ARM_DROP_GOAL_BIAS,
                    "action_index": 120,
                    "level": 0,
                    "diagnosis": "d",
                    "resolved_by_levelup": True,
                    "actions_to_levelup": 768,
                    "co_credited_count": k,
                }
            ],
        },
    }


def _eval_doc(rows: list[dict[str, Any]], *, complete: bool = True) -> dict[str, Any]:
    return {
        "experiment": "arc_leaderboard_eval",
        "policy": "e3",
        "budget": 20000,
        "random_seed": 20260719,
        "per_game": rows,
        "complete": complete,
        "honest_verdict": "x",
    }


# --- SCENARIO-A: the per_game shape is a rows document -------------------------------------


def test_per_game_rows_are_extracted_with_document_context() -> None:
    """SCENARIO-ARC-WMTE-7012-A: rows come out with `seed` and `arm` copied down."""
    rows = extract_rows(_eval_doc([_eval_row("r11l"), _eval_row("cd82")]))
    assert [r["game"] for r in rows] == ["r11l", "cd82"]
    assert all(r["seed"] == 20260719 for r in rows)
    assert all(r["arm"] == "eval:e3:budget20000" for r in rows)
    # a row that already carries the field keeps its own value
    own = _eval_row("x")
    own["seed"] = 7
    assert extract_rows(_eval_doc([own]))[0]["seed"] == 7
    # the harness shapes are unchanged
    assert extract_rows([{"a": 1}]) == [{"a": 1}]
    assert extract_rows({"rows": [{"a": 1}]}) == [{"a": 1}]
    assert extract_rows({"per_game": "not a list"}) == []


def test_an_eval_row_ingests_as_applied_evidence(tmp_path: Path) -> None:
    """SCENARIO-ARC-WMTE-7012-A: the entry carries game, levels, actions and the redirect's
    co-credit count."""
    path = tmp_path / "r11l-1.json"
    path.write_text(json.dumps(_eval_doc([_eval_row()])), encoding="utf-8")
    ledger = empty_ledger()
    counts = ingest_files(ledger, [path], NOW)
    assert counts["applied_new"] == 1
    (entry,) = ledger["entries"].values()
    assert entry["game"] == "r11l" and entry["levels"] == 1 and entry["actions"] == 2077
    assert entry["seed"] == 20260719 and entry["harness_arm"] == "eval:e3:budget20000"
    assert entry["redirects"][0]["co_credited_count"] == 3
    assert evaluate(ledger, NOW)["evidence"]["redirects"] == 1


# --- SCENARIO-B: a partial and its final dedupe --------------------------------------------


def test_the_partial_and_the_final_artifact_dedupe_to_one_entry(tmp_path: Path) -> None:
    """SCENARIO-ARC-WMTE-7012-B: the same game row banked twice is one piece of evidence."""
    row = _eval_row()
    (tmp_path / "r11l-9.partial.json").write_text(
        json.dumps(_eval_doc([row], complete=False)), encoding="utf-8"
    )
    (tmp_path / "r11l-9.json").write_text(json.dumps(_eval_doc([row])), encoding="utf-8")
    ledger = empty_ledger()
    counts = ingest_files(ledger, [tmp_path / "r11l-9.partial.json", tmp_path / "r11l-9.json"], NOW)
    assert counts["applied_new"] == 1 and counts["applied_duplicate"] == 1
    assert len(ledger["entries"]) == 1


# --- SCENARIO-C: a directory scan finds the eval-runs directory ----------------------------


def test_a_scan_takes_every_json_in_the_eval_runs_directory(tmp_path: Path) -> None:
    """SCENARIO-ARC-WMTE-7012-C: `*.json` inside `arc_leaderboard_eval_runs`, partials included,
    dotfiles and heartbeats excluded, `rows.json` elsewhere still found, nested clones pruned."""
    runs = tmp_path / "results" / EVAL_RUNS_DIR_NAME
    runs.mkdir(parents=True)
    (runs / "r11l-1.json").write_text(json.dumps(_eval_doc([_eval_row()])), encoding="utf-8")
    (runs / "cd82-2.partial.json").write_text(
        json.dumps(_eval_doc([_eval_row("cd82")], complete=False)), encoding="utf-8"
    )
    (runs / ".r11l-1.json.123.tmp").write_text("{", encoding="utf-8")
    # the REQ-7010 heartbeat lives here too; it is not a record and must not be swept
    (runs / "r11l-1.progress.json").write_text(json.dumps({"schema": "x"}), encoding="utf-8")
    harness = tmp_path / "supwindow"
    harness.mkdir()
    (harness / "rows.json").write_text(json.dumps({"rows": []}), encoding="utf-8")
    clone = tmp_path / "clone"
    (clone / EVAL_RUNS_DIR_NAME).mkdir(parents=True)
    (clone / ".git").write_text("gitdir: elsewhere", encoding="utf-8")
    (clone / EVAL_RUNS_DIR_NAME / "r11l-1.json").write_text(
        json.dumps(_eval_doc([_eval_row()])), encoding="utf-8"
    )

    found = scan_inputs([tmp_path])
    names = sorted(p.name for p in found)
    assert names == ["cd82-2.partial.json", "r11l-1.json", "rows.json"]
    assert all("clone" not in str(p) for p in found)


def test_a_json_outside_the_eval_runs_directory_is_not_swept(tmp_path: Path) -> None:
    """SCENARIO-ARC-WMTE-7012-C: the directory NAME is the scope, so an unrelated `*.json`
    (an experiment artifact that embeds receipts, say) is not double-counted."""
    other = tmp_path / "results"
    other.mkdir()
    (other / "experiment_6558_x.json").write_text(
        json.dumps(_eval_doc([_eval_row()])), encoding="utf-8"
    )
    assert scan_inputs([tmp_path]) == []


# --- SCENARIO-D: the consumer declares the fields it reads ---------------------------------


def test_the_declared_fields_are_the_ones_the_reader_uses() -> None:
    """SCENARIO-ARC-WMTE-7012-D: every declared field is a key this module reads, at the document
    level or the row level, and the container plus the receipt field are both declared. A
    declaration narrower than the reads is the class the consumer-field lint cannot see."""
    doc = _eval_doc([_eval_row()])
    row = doc["per_game"][0]
    assert set(EVAL_RUN_FIELDS_READ) <= set(doc) | set(row)
    for field in ("per_game", "random_seed", "policy", "budget"):
        assert field in EVAL_RUN_FIELDS_READ and field in doc
    for field in ("trajectory_supervisor", "game", "levels", "actions"):
        assert field in EVAL_RUN_FIELDS_READ and field in row
