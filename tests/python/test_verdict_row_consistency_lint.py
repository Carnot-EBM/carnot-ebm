"""REQ-OPS-VERDICT-ROW-6261: a verdict must survive contact with its own rows.

Spec: REQ-OPS-VERDICT-ROW-6261 / SCENARIO-OPS-VERDICT-ROW-6261-DEGENERATE-CONTROL

Every test here is built from a REAL artifact shape that misled a human in the 2026-08-11/12
session, not from an invented one. Each check is proven to bite: the fixtures are constructed so
that deleting the check makes a test fail, which is the mutation standard CLAUDE.md's QA-Layer
discipline asks for.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))

from verdict_row_consistency_lint import (  # noqa: E402
    _metric_like,
    check_all_rows_null,
    check_artifact,
    check_degenerate_control,
    check_no_headroom,
    check_wins_vs_losses,
)


def _write(tmp_path: Path, obj: dict) -> Path:
    p = tmp_path / "experiment_x.json"
    p.write_text(json.dumps(obj))
    return p


# --- ALL_ROWS_NULL: the exp6254 shape -------------------------------------------------


def test_all_rows_null_is_caught() -> None:
    rows = [{"game": g, "dense_held": None, "moe_best_held": None} for g in "abc"]
    out = check_all_rows_null({"honest_verdict": "complete_measured"}, rows)
    assert out and out[0].startswith("ALL_ROWS_NULL")


def test_one_real_value_is_enough_to_clear_all_rows_null() -> None:
    rows = [{"dense_held": None}, {"dense_held": 0.5}, {"dense_held": None}]
    assert check_all_rows_null({"honest_verdict": "complete"}, rows) == []


def test_all_rows_null_blocks_while_other_classes_only_warn(tmp_path: Path) -> None:
    p = _write(
        tmp_path,
        {"honest_verdict": "complete_x", "per_game_results": [{"held_fidelity": None}] * 3},
    )
    status, findings = check_artifact(p)
    assert status == "findings"
    assert any(f.startswith("ALL_ROWS_NULL") for f in findings)


# --- DEGENERATE_CONTROL: the exp6252 shape, nested per-arm ----------------------------


def test_a_control_identical_to_the_baseline_is_caught_through_nesting() -> None:
    """The founding bug. exp6252 nests arms as row["arms"][name][field]; the first version of
    this lint scanned only top-level keys and found nothing on its own motivating case."""
    rows = [
        {"game": f"g{i}", "arms": {"flat_none": {"nodes": n}, "uniform_random": {"nodes": n}}}
        for i, n in enumerate([10, 20, 30, 40, 50])
    ]
    rows = [{**r, **{f"arms.{a}.nodes": v["nodes"] for a, v in r["arms"].items()}} for r in rows]
    out = check_degenerate_control({}, rows)
    assert out and out[0].startswith("DEGENERATE_CONTROL")


def test_a_control_that_genuinely_differs_is_not_flagged() -> None:
    rows = [
        {"arms.flat_none.nodes": 10, "arms.uniform_random.nodes": 11},
        {"arms.flat_none.nodes": 20, "arms.uniform_random.nodes": 25},
        {"arms.flat_none.nodes": 30, "arms.uniform_random.nodes": 44},
        {"arms.flat_none.nodes": 40, "arms.uniform_random.nodes": 8},
    ]
    assert check_degenerate_control({}, rows) == []


# --- NO_HEADROOM_MAJORITY: the exp6251 shape ------------------------------------------


def test_rows_pinned_at_floor_and_ceiling_are_caught() -> None:
    rows = [
        {"best_of_n_held": 0.5265},
        {"best_of_n_held": 0.0},
        {"best_of_n_held": 0.5952},
        {"best_of_n_held": 1.0},
    ]
    out = check_no_headroom({}, rows)
    assert out and out[0].startswith("NO_HEADROOM_MAJORITY")


def test_metric_like_is_suffix_anchored_not_substring() -> None:
    """`best_of_n_held` was thrown away because the count-filter matched `_n` mid-name -- the
    same unanchored-substring bug the project's TAUTOLOGY check hit with "meta"/"meta_tensor"."""
    assert _metric_like("best_of_n_held") is True
    assert _metric_like("held_change_fidelity") is True
    assert _metric_like("valid_n") is False
    assert _metric_like("parallel_wall_s") is False
    assert _metric_like("states_expanded_count") is False


# --- WINS_NOT_EXCEEDING_LOSSES --------------------------------------------------------


def test_a_positive_verdict_with_more_losses_than_wins_is_caught() -> None:
    d = {"honest_verdict": "complete_gate_met_improved", "gate_met": True}
    rows = [{"delta": 0.3}, {"delta": -0.4}, {"delta": -0.1}]
    out = check_wins_vs_losses(d, rows)
    assert out and out[0].startswith("WINS_NOT_EXCEEDING_LOSSES")


def test_a_verdict_that_already_admits_a_null_is_not_nagged() -> None:
    """A tool that flags honest negatives teaches people to ignore it."""
    d = {"honest_verdict": "complete_gate_not_met_no_reliable_signal"}
    rows = [{"delta": 0.3}, {"delta": -0.4}, {"delta": -0.1}]
    assert check_wins_vs_losses(d, rows) == []


# --- structural ------------------------------------------------------------------------


def test_an_artifact_with_no_rows_is_SKIPPED_not_passed(tmp_path: Path) -> None:
    """Absence of rows is absence of evidence. Reporting it as clean would be the
    guard-is-green-while-blind failure this project keeps finding."""
    p = _write(tmp_path, {"honest_verdict": "complete", "summary": "no rows here"})
    status, _ = check_artifact(p)
    assert status == "skipped"


def test_a_clean_artifact_reports_ok(tmp_path: Path) -> None:
    p = _write(
        tmp_path,
        {
            "honest_verdict": "complete_gate_not_met",
            "per_game_results": [
                {"held_fidelity": 0.3, "delta": 0.1},
                {"held_fidelity": 0.6, "delta": 0.2},
                {"held_fidelity": 0.45, "delta": 0.05},
            ],
        },
    )
    status, findings = check_artifact(p)
    assert status == "ok", findings
