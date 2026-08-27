"""REQ-ARC-WMTE-6720: unattended cross-run supervisor refinement.

Every test writes only under tmp_path. A test must never write tracked
state (Test-Run Record Integrity Discipline) — the real ledger under ops/
is reached only through the CLI's default path, which every test overrides.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from carnot.agentic.arc_supervisor_refinement import (
    MIN_FIRED_PER_ARM,
    STATUS_INSUFFICIENT,
    STATUS_NO_FIRINGS,
    STATUS_NO_RECEIPTS,
    STATUS_RECOMMENDATION,
    classify_receipt,
    empty_ledger,
    evaluate,
    ingest_files,
    load_ledger,
    main,
    render_report,
    scan_inputs,
    wilson_bounds,
)
from carnot.agentic.arc_trajectory_supervisor import (
    ARM_ALLOW_REINDUCTION,
    ARM_DROP_GOAL_BIAS,
    ARM_FORCE_DIVERSITY,
    ARM_ORDER,
)

NOW = "2026-08-27T00:00:00+00:00"


def _redirect(arm: str, resolved: bool, a2l: int | None = None, idx: int = 100) -> dict:
    return {
        "arm": arm,
        "action_index": idx,
        "level": 0,
        "diagnosis": "test",
        "resolved_by_levelup": resolved,
        "actions_to_levelup": a2l,
    }


def _applied_row(
    game: str = "tu93",
    seed: int = 1,
    redirects: list[dict] | None = None,
    stag: int = 0,
    wall_s: float = 100.0,
) -> dict:
    return {
        "game": game,
        "seed": seed,
        "arm": "S_llmon",
        "levels": 1,
        "wall_s": wall_s,
        "trajectory_supervisor": {
            "enabled": True,
            "mode": "applied",
            "window": 120,
            "actions_observed": 399,
            "arms_used": [],
            "arm_outcomes": {},
            "stagnations_unredirected": stag,
            "redirects": redirects or [],
        },
    }


def _write_rows(path: Path, rows: list[dict], wrap: bool = True) -> Path:
    doc: object = {"rows": rows} if wrap else rows
    path.write_text(json.dumps(doc), encoding="utf-8")
    return path


def _ingest(tmp_path: Path, rows: list[dict], name: str = "rows.json") -> tuple[dict, dict]:
    file_path = _write_rows(tmp_path / name, rows)
    ledger = empty_ledger()
    counts = ingest_files(ledger, [file_path], NOW)
    return ledger, counts


# --- SCENARIO-ARC-WMTE-6720-1: applied-only evidence filter -----------------


def test_applied_only_evidence_filter(tmp_path: Path) -> None:
    shadow_row = {
        "game": "tu93",
        "trajectory_supervisor": {
            "enabled": False,
            "mode": "shadow",
            "would_have_redirects": [_redirect("shadow_only_arm", True)],
        },
    }
    rows = [
        _applied_row(seed=1, redirects=[_redirect(ARM_DROP_GOAL_BIAS, True, 50)]),
        _applied_row(seed=2),
        shadow_row,
        {"game": "tu93", "trajectory_supervisor": {"error": "boom"}},
        {"game": "tu93"},
        _applied_row(seed=3, redirects=[_redirect("future_arm", False)]),
    ]
    ledger, counts = _ingest(tmp_path, rows)
    assert counts["applied_new"] == 3
    assert counts["shadow_observed"] == 1
    assert counts["error_rows"] == 1
    assert counts["rows_without_receipt"] == 1
    recommendation = evaluate(ledger, NOW)
    arms = [row["arm"] for row in recommendation["per_arm"]]
    # A shadow counterfactual must never reach the redirect pool.
    assert "shadow_only_arm" not in arms
    # An unknown applied arm is still listed, after the curated order.
    assert arms[: len(ARM_ORDER)] == list(ARM_ORDER)
    assert "future_arm" in arms
    assert recommendation["evidence"]["redirects"] == 2


def test_classify_receipt_kinds() -> None:
    # SCENARIO-ARC-WMTE-6720-1: each receipt kind maps to one classification.
    assert classify_receipt(_applied_row()) == "applied"
    assert classify_receipt({"trajectory_supervisor": {"mode": "shadow"}}) == "shadow"
    assert classify_receipt({"trajectory_supervisor": {"error": "x"}}) == "error"
    assert classify_receipt({"trajectory_supervisor": {"enabled": False}}) == "other"
    assert classify_receipt({}) == "absent"


# --- SCENARIO-ARC-WMTE-6720-2: dedupe and durability ------------------------


def test_dedupe_and_ledger_survives_source_deletion(tmp_path: Path) -> None:
    ledger_path = tmp_path / "ledger.json"
    rows_path = _write_rows(
        tmp_path / "rows.json",
        [_applied_row(seed=1, redirects=[_redirect(ARM_DROP_GOAL_BIAS, True, 40)])],
    )
    assert main([str(rows_path), "--ledger", str(ledger_path)]) == 0
    assert main([str(rows_path), "--ledger", str(ledger_path)]) == 0
    ledger = load_ledger(ledger_path)
    assert len(ledger["entries"]) == 1
    assert ledger["recommendation"]["ingest_counts"]["applied_duplicate"] == 1
    rows_path.unlink()
    # Re-evaluation from the ledger alone reproduces the same totals.
    assert main(["--ledger", str(ledger_path)]) == 0
    ledger = load_ledger(ledger_path)
    assert ledger["recommendation"]["evidence"]["receipts"] == 1
    assert ledger["recommendation"]["evidence"]["redirects"] == 1


def test_rerun_with_different_measurements_counts_as_new(tmp_path: Path) -> None:
    # SCENARIO-ARC-WMTE-6720-2: a re-run differs in measured fields, so it
    # ingests as new evidence; only byte-identical copies dedupe.
    rows = [
        _applied_row(seed=1, wall_s=100.0),
        _applied_row(seed=1, wall_s=101.5),
    ]
    ledger, counts = _ingest(tmp_path, rows)
    assert counts["applied_new"] == 2
    assert counts["applied_duplicate"] == 0
    assert len(ledger["entries"]) == 2


# --- SCENARIO-ARC-WMTE-6720-3: insufficient evidence is loud ----------------


def test_current_live_shape_reads_insufficient(tmp_path: Path) -> None:
    rows = []
    for seed, resolved in ((1, True), (2, False)):
        rows.append(
            _applied_row(
                seed=seed,
                redirects=[
                    _redirect(ARM_DROP_GOAL_BIAS, resolved, 30 if resolved else None),
                    _redirect(ARM_ALLOW_REINDUCTION, resolved, 40 if resolved else None),
                ],
            )
        )
        rows.append(
            _applied_row(
                game="ar25",
                seed=seed,
                redirects=[_redirect(ARM_FORCE_DIVERSITY, resolved, 50 if resolved else None)],
            )
        )
    ledger, _ = _ingest(tmp_path, rows)
    recommendation = evaluate(ledger, NOW)
    assert recommendation["status"] == STATUS_INSUFFICIENT
    assert recommendation["recommendations"] == []
    for row in recommendation["per_arm"]:
        assert row["floor_shortfall"] == MIN_FIRED_PER_ARM - row["fired"]
    report = render_report(recommendation)
    assert "INSUFFICIENT EVIDENCE" in report
    assert str(MIN_FIRED_PER_ARM) in report


# --- SCENARIO-ARC-WMTE-6720-4: retire floor is exact ------------------------


def _rows_with_arm_counts(arm: str, fired: int, helped: int) -> list[dict]:
    rows = []
    for index in range(fired):
        resolved = index < helped
        rows.append(
            _applied_row(
                seed=index,
                redirects=[_redirect(arm, resolved, 25 if resolved else None)],
            )
        )
    return rows


def test_retire_at_floor_not_below(tmp_path: Path) -> None:
    ledger, _ = _ingest(tmp_path, _rows_with_arm_counts(ARM_DROP_GOAL_BIAS, 10, 0))
    recommendation = evaluate(ledger, NOW)
    assert recommendation["status"] == STATUS_RECOMMENDATION
    kinds = {(item["kind"], item["arm"]) for item in recommendation["recommendations"]}
    assert kinds == {("retire_candidate", ARM_DROP_GOAL_BIAS)}
    ledger9, _ = _ingest(tmp_path, _rows_with_arm_counts(ARM_DROP_GOAL_BIAS, 9, 0), "nine.json")
    recommendation9 = evaluate(ledger9, NOW)
    assert recommendation9["status"] == STATUS_INSUFFICIENT
    assert recommendation9["recommendations"] == []


# --- SCENARIO-ARC-WMTE-6720-5: promotion needs interval separation ----------


def test_promotion_requires_interval_separation(tmp_path: Path) -> None:
    separated = _rows_with_arm_counts(ARM_FORCE_DIVERSITY, 12, 12)
    separated += [
        _applied_row(game="ar25", seed=100 + index, redirects=[red])
        for index, red in enumerate(
            _redirect(ARM_DROP_GOAL_BIAS, index == 0, 20 if index == 0 else None)
            for index in range(12)
        )
    ]
    ledger, _ = _ingest(tmp_path, separated)
    recommendation = evaluate(ledger, NOW)
    assert recommendation["status"] == STATUS_RECOMMENDATION
    kinds = {(item["kind"], item["arm"]) for item in recommendation["recommendations"]}
    assert kinds == {("raise_priority_candidate", ARM_FORCE_DIVERSITY)}
    # Every recommendation output carries the post-hoc caveat contract.
    assert recommendation["recommendation_only"] is True
    assert "not evidence of cause" in recommendation["causal_caveat"]

    overlapping = _rows_with_arm_counts(ARM_FORCE_DIVERSITY, 12, 8)
    overlapping += [
        _applied_row(game="ar25", seed=200 + index, redirects=[red])
        for index, red in enumerate(
            _redirect(ARM_DROP_GOAL_BIAS, index < 6, 20 if index < 6 else None)
            for index in range(12)
        )
    ]
    ledger2, _ = _ingest(tmp_path, overlapping, "overlap.json")
    recommendation2 = evaluate(ledger2, NOW)
    assert recommendation2["status"] == STATUS_INSUFFICIENT
    assert recommendation2["recommendations"] == []


def test_wilson_bounds_shape() -> None:
    # SCENARIO-ARC-WMTE-6720-5: the interval behaves at the exact counts the
    # rules read (0/n, n/n, and the empty case).
    assert wilson_bounds(0, 0) == (0.0, 1.0)
    lower, upper = wilson_bounds(0, 10)
    assert lower == 0.0
    assert upper == pytest.approx(0.2129, abs=1e-3)
    lower_full, upper_full = wilson_bounds(10, 10)
    assert upper_full == 1.0
    assert lower_full == pytest.approx(1 - 0.2129, abs=1e-3)


# --- SCENARIO-ARC-WMTE-6720-6: new-arm specification stays human ------------


def test_new_arm_specification_from_exhausted_receipt(tmp_path: Path) -> None:
    exhausted = _applied_row(
        seed=7,
        stag=2,
        redirects=[_redirect(arm, False) for arm in ARM_ORDER],
    )
    not_exhausted = _applied_row(
        seed=8,
        stag=0,
        redirects=[_redirect(arm, False) for arm in ARM_ORDER],
    )
    ledger, _ = _ingest(tmp_path, [exhausted, not_exhausted])
    recommendation = evaluate(ledger, NOW)
    spec = recommendation["new_arm_specification"]
    assert spec is not None
    assert recommendation["status"] == STATUS_RECOMMENDATION
    assert spec["audience"] == "human"
    assert "never generates an arm implementation" in spec["instruction"]
    assert len(spec["cells"]) == 1
    assert spec["cells"][0]["game"] == "tu93"
    assert spec["cells"][0]["stagnations_unredirected"] == 2
    report = render_report(recommendation)
    assert "NEW ARM SPECIFICATION" in report


def test_no_new_arm_specification_without_stagnation(tmp_path: Path) -> None:
    # SCENARIO-ARC-WMTE-6720-6: firing every arm alone is not the trigger;
    # stagnation must have CONTINUED after the table ran out.
    ledger, _ = _ingest(
        tmp_path,
        [_applied_row(seed=9, stag=0, redirects=[_redirect(arm, False) for arm in ARM_ORDER])],
    )
    assert evaluate(ledger, NOW)["new_arm_specification"] is None


# --- SCENARIO-ARC-WMTE-6720-7: anti-churn and clone pruning -----------------


def test_no_firings_and_no_receipts_exit_zero(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    ledger_path = tmp_path / "ledger.json"
    rows_path = _write_rows(tmp_path / "rows.json", [_applied_row(seed=1), _applied_row(seed=2)])
    assert main([str(rows_path), "--ledger", str(ledger_path)]) == 0
    out = capsys.readouterr().out
    assert "NO FIRINGS" in out
    assert load_ledger(ledger_path)["recommendation"]["status"] == STATUS_NO_FIRINGS

    fresh = tmp_path / "fresh.json"
    assert main(["--ledger", str(fresh)]) == 0
    out = capsys.readouterr().out
    assert "NO RECEIPTS INGESTED" in out
    assert load_ledger(fresh)["recommendation"]["status"] == STATUS_NO_RECEIPTS


def test_directory_scan_prunes_nested_clones(tmp_path: Path) -> None:
    scan = tmp_path / "scan"
    (scan / "good").mkdir(parents=True)
    _write_rows(scan / "good" / "rows.json", [_applied_row(seed=1)])
    clone = scan / "clone"
    (clone / ".git").mkdir(parents=True)
    _write_rows(clone / "rows.json", [_applied_row(seed=2)])
    worktree = scan / "wt"
    worktree.mkdir()
    (worktree / ".git").write_text("gitdir: /elsewhere\n", encoding="utf-8")
    _write_rows(worktree / "rows.json", [_applied_row(seed=3)])
    found = scan_inputs([scan])
    assert found == [scan / "good" / "rows.json"]
    # An explicit file plus the directory containing it stays one file.
    assert scan_inputs([scan, scan / "good" / "rows.json"]) == [scan / "good" / "rows.json"]


def test_missing_input_and_bad_ledger_fail_loud(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    ledger_path = tmp_path / "ledger.json"
    assert main([str(tmp_path / "absent.json"), "--ledger", str(ledger_path)]) == 1
    assert "ERROR" in capsys.readouterr().err
    bad = tmp_path / "bad_ledger.json"
    bad.write_text(json.dumps({"schema": "wrong"}), encoding="utf-8")
    with pytest.raises(ValueError):
        load_ledger(bad)
    assert main(["--ledger", str(bad)]) == 1
    assert "ERROR" in capsys.readouterr().err


def test_bare_list_rows_document_shape(tmp_path: Path) -> None:
    # REQ-ARC-WMTE-6720 rule 1: both historical rows-document shapes ingest.
    file_path = _write_rows(tmp_path / "bare.json", [_applied_row(seed=1)], wrap=False)
    ledger = empty_ledger()
    counts = ingest_files(ledger, [file_path], NOW)
    assert counts["applied_new"] == 1


def test_json_output_mode(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    # REQ-ARC-WMTE-6720 rule 6: machine-readable output for unattended use.
    ledger_path = tmp_path / "ledger.json"
    rows_path = _write_rows(
        tmp_path / "rows.json",
        [_applied_row(seed=1, redirects=[_redirect(ARM_DROP_GOAL_BIAS, True, 10)])],
    )
    assert main([str(rows_path), "--ledger", str(ledger_path), "--json"]) == 0
    out = capsys.readouterr().out
    payload = json.loads(out[: out.rindex("}") + 1])
    assert payload["status"] == STATUS_INSUFFICIENT
    assert payload["recommendation_only"] is True
