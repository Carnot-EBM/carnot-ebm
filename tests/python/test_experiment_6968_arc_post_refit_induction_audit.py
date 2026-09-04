"""Tests for the frozen ARC post-refit induction audit.

Spec refs: REQ-ARC-WMTE-6968 and SCENARIO-ARC-WMTE-6968-*.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from carnot import experiment_6968_arc_post_refit_induction_audit as exp


ROOT = Path(__file__).resolve().parents[2]


def _rows() -> list[dict[str, object]]:
    """Return a small ordered corpus with shown and held-out change/no-op rows."""

    return [
        {
            "transition_id": "s0",
            "index": 0,
            "grid": [[0, 0], [0, 0]],
            "next_grid": [[1, 0], [0, 0]],
            "action": 6,
            "data": {"x": 0},
            "level_before": 0,
            "level_after": 0,
        },
        {
            "transition_id": "s1",
            "index": 1,
            "grid": [[1, 0], [0, 0]],
            "next_grid": [[1, 0], [0, 0]],
            "action": 1,
            "data": None,
            "level_before": 0,
            "level_after": 0,
        },
        {
            "transition_id": "h0",
            "index": 2,
            "grid": [[0, 0], [0, 0]],
            "next_grid": [[0, 1], [0, 0]],
            "action": 6,
            "data": {"x": 1},
            "level_before": 0,
            "level_after": 0,
        },
        {
            "transition_id": "h1",
            "index": 3,
            "grid": [[0, 1], [0, 0]],
            "next_grid": [[0, 1], [0, 0]],
            "action": 1,
            "data": None,
            "level_before": 0,
            "level_after": 0,
        },
    ]


def _engine_source(*, memorizes: bool = False) -> str:
    """Return a general rule or a literal shown-row lookup engine."""

    if memorizes:
        return """import numpy as np
_ROW_TABLE = {((0, 0, 0, 0), 6, 0): ((1, 0), (0, 0))}
def engine(grid, action, data):
    g = np.asarray(grid).copy()
    key = (tuple(int(v) for v in g.flat), int(action), (data or {}).get('x'))
    return np.asarray(_ROW_TABLE[key]) if key in _ROW_TABLE else g
def is_level_complete(grid):
    return False
"""
    return """import numpy as np
def engine(grid, action, data):
    g = np.asarray(grid).copy()
    if action == 6:
        g[0, int(data['x'])] = 1
    return g
def is_level_complete(grid):
    return False
"""


def _write_fixture(repo: Path, *, memorizes: bool = False) -> tuple[str, Path]:
    """Write one complete content-addressed audit fixture below a temporary root."""

    engine_source = _engine_source(memorizes=memorizes)
    engine_hash = exp.sha256_bytes(engine_source.encode())
    engine_sha16 = engine_hash.removeprefix("sha256:")[:16]
    engine_path = (
        repo / "results/arc_e3/r11l/attempts" / f"wm_20260904T020536_575501__{engine_sha16}.py"
    )
    engine_path.parent.mkdir(parents=True)
    engine_path.write_text(engine_source, encoding="utf-8")
    transition_path = repo / "results/arc_e3/r11l/attempts/transitions.json"
    transition_source = {
        "schema": "carnot.arc_induction_attempt_transitions.v1",
        "attempt_engine_sha256": engine_hash,
        "prompt_sha256": "sha256:prompt",
        "prompt_row_ids": ["s0", "s1"],
        "repair_feedback_row_ids": [],
        "rows": _rows(),
    }
    transition_path.write_text(json.dumps(transition_source), encoding="utf-8")
    run_path = repo / "results/arc_leaderboard_eval_runs/r11l-42.json"
    run_path.parent.mkdir(parents=True)
    run_path.write_text(
        json.dumps(
            {
                "complete": True,
                "policy": "e3",
                "run_started_at": "2026-09-03T20:52:51-04:00",
                "run_completed_at": "2026-09-04T04:00:00-04:00",
                "headline_score": -999,
                "per_game": [
                    {
                        "game": "r11l",
                        "generator_provenance": {"n_ctx": 98304},
                        "policy_diagnostics": {
                            "induction_attempts": [
                                {
                                    "reason": "stall",
                                    "refinement_rounds": [
                                        {
                                            "engine_source_sha256": {
                                                "sha256_16": engine_sha16,
                                                "chars": len(engine_source),
                                            },
                                            "engine_emitted_at": "2026-09-04T02:05:36Z",
                                            "prompt_sha256": "sha256:prompt",
                                            "transition_source_path": str(
                                                transition_path.relative_to(repo)
                                            ),
                                        }
                                    ],
                                }
                            ]
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    scorer = repo / exp.SCORER_PATH
    scorer.parent.mkdir(parents=True, exist_ok=True)
    scorer.write_text("# frozen scorer\n", encoding="utf-8")
    registry = repo / exp.REGISTRY_PATH
    registry.parent.mkdir(parents=True, exist_ok=True)
    registry.write_text("games: {}\n", encoding="utf-8")
    return engine_sha16, run_path


def test_req_arc_wmte_6968_spec_precedes_code_and_names_contract() -> None:
    """REQ-ARC-WMTE-6968 declares every field and required scenario."""

    text = (ROOT / exp.SPEC_PATH).read_text(encoding="utf-8")
    section = text[text.index("### REQ-ARC-WMTE-6968") :]

    assert all(f"`{field}`" in section for field in exp.REQUIRED_ARTIFACT_FIELDS)
    assert all(
        f"SCENARIO-ARC-WMTE-6968-{name}" in section
        for name in (
            "RUN-SELECTION",
            "PROVENANCE",
            "SPLIT-PURITY",
            "TRANSITION-SCORING",
            "CONTROLS",
            "MEMORIZATION",
            "BLOCKED",
            "NO-SOLVE",
        )
    )


def test_scenario_arc_wmte_6968_run_selection_uses_provenance_not_score(
    tmp_path: Path,
) -> None:
    """SCENARIO-ARC-WMTE-6968-RUN-SELECTION ignores score and partial rows."""

    target, selected_path = _write_fixture(tmp_path)
    selected = json.loads(selected_path.read_text(encoding="utf-8"))
    better = deepcopy(selected)
    better["complete"] = False
    better["headline_score"] = 1_000_000
    (selected_path.parent / "r11l-43.partial.json").write_text(json.dumps(better))
    wrong_context = deepcopy(selected)
    wrong_context["headline_score"] = 2_000_000
    wrong_context["per_game"][0]["generator_provenance"]["n_ctx"] = 49152
    (selected_path.parent / "r11l-44.json").write_text(json.dumps(wrong_context))

    candidates = exp.discover_run_candidates(tmp_path, target_engine_sha16=target)
    chosen, reason = exp.select_run(candidates)

    assert chosen is not None
    assert chosen["path"] == str(selected_path.relative_to(tmp_path))
    assert chosen["n_ctx"] == 98304
    assert reason == "selected_by_environment_engine_hash_and_timestamp"


def test_scenarios_arc_wmte_6968_provenance_and_split_purity_fail_closed(
    tmp_path: Path,
) -> None:
    """SCENARIO-ARC-WMTE-6968-PROVENANCE and SPLIT-PURITY reject ambiguity and leakage."""

    target, _ = _write_fixture(tmp_path)
    engine, engine_rows = exp.resolve_engine(tmp_path, target)
    assert engine is not None
    assert len(engine_rows) == 1
    copy_path = engine.with_name(f"wm_20260904T020537_000000__{target}.py")
    copy_path.write_bytes(engine.read_bytes())
    ambiguous, ambiguous_rows = exp.resolve_engine(tmp_path, target)
    assert ambiguous is None
    assert len(ambiguous_rows) == 2

    rows = _rows()
    split, purity, failures = exp.rebuild_split(
        rows,
        prompt_row_ids=["s0", "s1", "h0"],
        repair_feedback_row_ids=["h1"],
    )
    assert split["shown_row_ids"] == ["s0", "s1", "h0"]
    assert split["heldout_row_ids"] == ["h1"]
    assert any(row["repair_feedback_visible"] for row in purity)
    assert {row["failed_check"] for row in failures} == {"heldout_rows_absent_from_repair_feedback"}


def test_scenario_arc_wmte_6968_transition_metrics_cover_change_noop_and_exception() -> None:
    """SCENARIO-ARC-WMTE-6968-TRANSITION-SCORING records symmetric cell metrics."""

    changed = _rows()[0]
    scored = exp.score_prediction(changed, [[1, 1], [0, 0]])
    failed = exp.score_prediction(changed, None, exception="ValueError: bad engine")

    assert scored["exact_transition_correct"] is False
    assert scored["exact_cell_accuracy"] == 0.75
    assert scored["changed_cell_recall"] == 1.0
    assert scored["changed_cell_precision"] == 0.5
    assert scored["changing_transition_correct"] is False
    assert scored["noop_correct"] is None
    assert failed["status"] == "exception"
    assert failed["exception"] == "ValueError: bad engine"
    assert failed["exact_cell_accuracy"] == 0.0


def test_scenarios_arc_wmte_6968_fresh_process_controls_and_memorization(
    tmp_path: Path,
) -> None:
    """SCENARIO-ARC-WMTE-6968-CONTROLS and MEMORIZATION expose lookup-table collapse."""

    engine_path = tmp_path / "memorizer.py"
    engine_path.write_text(_engine_source(memorizes=True), encoding="utf-8")
    rows = _rows()
    shown, heldout = rows[:2], rows[2:]

    engine_rows, execution_rows = exp.execute_engine_fresh(engine_path, rows, timeout_s=5)
    controls = exp.score_controls(shown, heldout)
    signatures = exp.detect_memorization(
        engine_path.read_text(encoding="utf-8"), engine_rows, controls
    )

    assert len(engine_rows) == 4
    assert all(row["terminal"] is True for row in engine_rows)
    assert execution_rows[0]["fresh_process"] is True
    assert execution_rows[0]["worker_pid"] != execution_rows[0]["parent_pid"]
    assert {row["control"] for row in controls} == {
        "identity",
        "constant_delta",
        "nearest_shown_delta",
        "row_table_memorization",
    }
    assert len(controls) == 8
    assert any(row["signature"] == "literal_row_or_delta_table" for row in signatures)
    assert any(row["signature"] == "prefix_to_heldout_collapse" for row in signatures)


def test_scenario_arc_wmte_6968_worker_denies_repository_writes(tmp_path: Path) -> None:
    """SCENARIO-ARC-WMTE-6968-TRANSITION-SCORING confines generated code."""

    marker = tmp_path / "escaped.txt"
    engine = tmp_path / "writer.py"
    engine.write_text(
        "def engine(grid, action, data):\n"
        f"    open({str(marker)!r}, 'w').write('bad')\n"
        "    return grid\n",
        encoding="utf-8",
    )

    rows, execution = exp.execute_engine_fresh(engine, [_rows()[0]], timeout_s=5)

    assert rows[0]["status"] == "exception"
    assert "PermissionError" in rows[0]["exception"]
    assert execution[0]["write_guard_enabled"] is True
    assert not marker.exists()


def test_req_arc_wmte_6968_complete_artifact_has_positive_paired_gate(
    tmp_path: Path,
) -> None:
    """REQ-ARC-WMTE-6968 opens only on leakage-free paired held-out superiority."""

    target, _ = _write_fixture(tmp_path)
    artifact = exp.build_artifact(date="20260904", repo_root=tmp_path, target_engine_sha16=target)

    assert set(exp.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert set(artifact["field_principles"]) == set(exp.REQUIRED_ARTIFACT_FIELDS)
    assert artifact["arc_induction_audit_complete_score"] == 1
    assert artifact["arc_induction_generalization_positive_score"] == 1
    assert artifact["verdict_class"] == "positive"
    assert artifact["honest_verdict"].startswith("positive_")
    assert artifact["prefix_exact_accuracy"] == 1.0
    assert artifact["heldout_exact_accuracy"] == 1.0
    assert artifact["heldout_changing_accuracy"] == 1.0
    assert artifact["heldout_noop_accuracy"] == 1.0
    assert artifact["paired_control_delta_rows"][0]["interval_low"] > 0
    assert artifact["gate_check_summary"] == []
    assert artifact["reproducibility_checksum"].startswith("sha256:")
    assert all(
        artifact[field] is False
        for field in (
            "solve_claimed",
            "level_claimed",
            "registry_updated",
            "submitted_to_leaderboard",
            "verifier_is_oracle",
        )
    )


def test_scenarios_arc_wmte_6968_blocked_and_no_solve_fields_are_terminal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-ARC-WMTE-6968-BLOCKED and NO-SOLVE emit a complete diagnostic schema."""

    output = tmp_path / "blocked.json"
    artifact = exp.build_artifact(date="20260904", repo_root=tmp_path)
    monkeypatch.setattr(exp, "REPO_ROOT", tmp_path)

    assert set(exp.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert artifact["honest_verdict"] == "blocked_arc_post_refit_induction_audit"
    assert artifact["verdict_class"] == "blocked"
    assert artifact["arc_induction_audit_complete_score"] == 0
    assert artifact["arc_induction_generalization_positive_score"] == 0
    assert artifact["gate_check_summary"]
    assert all(
        {"failed_check", "expected_value", "observed_value"} <= set(row)
        for row in artifact["gate_check_summary"]
    )
    assert exp.main(["--date", "20260904", "--output", str(output)]) == 1
    assert json.loads(output.read_text(encoding="utf-8"))["verdict_class"] == "blocked"


def test_req_arc_wmte_6968_real_checkout_blocks_without_frozen_attempt_rows() -> None:
    """REQ-ARC-WMTE-6968 does not replace missing live rows with a fresh rollout."""

    artifact = exp.build_artifact(date="20260904", repo_root=ROOT)

    assert artifact["honest_verdict"] == "blocked_arc_post_refit_induction_audit"
    assert artifact["transition_hash"] is None
    assert artifact["prompt_hash"] is None
    assert any(
        row["failed_check"] in {"completed_target_run_count", "immutable_transition_source"}
        for row in artifact["gate_check_summary"]
    )


def test_req_arc_wmte_6968_defensive_read_split_and_metric_boundaries(tmp_path: Path) -> None:
    """REQ-ARC-WMTE-6968 fails closed on malformed evidence and grid shapes."""

    missing = tmp_path / "missing.json"
    malformed = tmp_path / "malformed.json"
    scalar = tmp_path / "scalar.json"
    malformed.write_text("{", encoding="utf-8")
    scalar.write_text("[]", encoding="utf-8")
    run_dir = tmp_path / exp.RUNS_PATH
    run_dir.mkdir(parents=True)
    (run_dir / "bad.json").write_text("{", encoding="utf-8")
    (run_dir / "odd.json").write_text(
        json.dumps(
            {
                "per_game": [
                    None,
                    {"game": "other"},
                    {
                        "game": "r11l",
                        "policy_diagnostics": {"induction_attempts": [None]},
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    manifest = tmp_path / exp.ATTEMPTS_PATH / "manifest.jsonl"
    manifest.parent.mkdir(parents=True)
    manifest.write_text("not-json\n[]\n", encoding="utf-8")

    assert exp._read_json_object(missing) is None
    assert exp._read_json_object(malformed) is None
    assert exp._read_json_object(scalar) is None
    assert exp.discover_run_candidates(tmp_path) == []
    assert exp._safe_relative_source(tmp_path, None) is None
    assert exp._safe_relative_source(tmp_path, "/absolute.json") is None
    assert exp._safe_relative_source(tmp_path, "../escape.json") is None
    with pytest.raises(ValueError, match="two_dimensional"):
        exp._array([1, 2])

    wrong_shape = exp.score_prediction(_rows()[0], [[1, 0, 0]])
    invalid_prediction = exp.score_prediction(_rows()[0], [1, 0])
    assert wrong_shape["exact_cell_accuracy"] == 0.0
    assert wrong_shape["changed_cell_precision"] == 0.0
    assert invalid_prediction["status"] == "exception"
    assert exp._delta_signature({"grid": [[0]], "next_grid": [[0, 0]]}) == ()
    no_change_controls = exp.score_controls([_rows()[1]], [_rows()[3]])
    assert len(no_change_controls) == 4
    assert exp.paired_control_deltas([], []) == []
    assert (
        exp.paired_control_deltas(
            [{"transition_id": "x", "changing_transition_correct": True}],
            [{"transition_id": "y", "changing_transition_correct": True, "control": "identity"}],
        )
        == []
    )


@pytest.mark.parametrize("failure", ["exit", "timeout", "json"])
def test_req_arc_wmte_6968_fresh_worker_failures_become_terminal_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    """REQ-ARC-WMTE-6968 records child exit, timeout, and protocol failures."""

    engine = tmp_path / "engine.py"
    engine.write_text(_engine_source(), encoding="utf-8")

    class FakeProcess:
        """Provide one controlled subprocess result without executing generated code."""

        pid = 999
        returncode = 0 if failure != "exit" else 7

        def __init__(self) -> None:
            self.calls = 0

        def communicate(self, _payload: str | None = None, timeout: float | None = None):
            self.calls += 1
            if failure == "timeout" and self.calls == 1:
                raise exp.subprocess.TimeoutExpired("worker", timeout)
            if failure == "json":
                return "not json", ""
            return "", "child failed"

        def kill(self) -> None:
            self.returncode = -9

    monkeypatch.setattr(exp.subprocess, "Popen", lambda *args, **kwargs: FakeProcess())

    rows, receipt = exp.execute_engine_fresh(engine, [_rows()[0]], timeout_s=0.01)

    assert rows[0]["status"] == "exception"
    assert receipt[0]["process_error"] is not None
    assert receipt[0]["terminal"] is True


@pytest.mark.parametrize("failure", ["source_missing", "hash", "schema", "leakage"])
def test_scenario_arc_wmte_6968_complete_receipt_still_blocks_bad_transition_evidence(
    tmp_path: Path, failure: str
) -> None:
    """SCENARIO-ARC-WMTE-6968-BLOCKED rejects each transition-source failure class."""

    target, run_path = _write_fixture(tmp_path)
    run = json.loads(run_path.read_text(encoding="utf-8"))
    round_row = run["per_game"][0]["policy_diagnostics"]["induction_attempts"][0][
        "refinement_rounds"
    ][0]
    source_path = tmp_path / round_row["transition_source_path"]
    source = json.loads(source_path.read_text(encoding="utf-8"))
    if failure == "source_missing":
        round_row["transition_source_path"] = "results/missing.json"
        run_path.write_text(json.dumps(run), encoding="utf-8")
    elif failure == "hash":
        source["attempt_engine_sha256"] = "sha256:wrong"
        source_path.write_text(json.dumps(source), encoding="utf-8")
    elif failure == "schema":
        source["rows"][0]["grid"] = [0, 0]
        source_path.write_text(json.dumps(source), encoding="utf-8")
    else:
        source["repair_feedback_row_ids"] = ["h0"]
        source_path.write_text(json.dumps(source), encoding="utf-8")

    artifact = exp.build_artifact(date="20260904", repo_root=tmp_path, target_engine_sha16=target)

    assert artifact["verdict_class"] == "blocked"
    assert artifact["gate_check_summary"]


def test_req_arc_wmte_6968_relative_main_output_stays_below_repo(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-ARC-WMTE-6968 keeps the command output relative to its selected checkout."""

    monkeypatch.setattr(exp, "REPO_ROOT", tmp_path)

    assert exp.main(["--date", "20260904", "--output", "relative.json"]) == 1
    assert (tmp_path / "relative.json").is_file()


def test_req_arc_wmte_6968_sentinel_names_the_gate_that_actually_failed(
    tmp_path: Path,
) -> None:
    """The early-return sentinel says it was not evaluated and names the real failure.

    Origin (2026-09-04 known-issues): the hardcoded-False sentinel reported
    `failed_check: immutable_transition_source` on a run where the transition source
    was never read, sending a reader down a wrong hypothesis — same class as
    REQ-CONDUCTOR-VERDICT-3's `artifact_not_updated_past_bootstrap`.
    """

    # An empty checkout: zero candidates, so every upstream selection gate fails.
    artifact = exp.build_artifact(date="20260904", repo_root=tmp_path)

    sentinel = next(
        row
        for row in artifact["gate_check_summary"]
        if row["failed_check"] == "immutable_transition_source"
    )
    observed = str(sentinel["observed_value"])
    assert observed.startswith("not_evaluated_upstream_gate_failed:"), observed
    # The named cause is a REAL upstream gate, not the sentinel's own name.
    assert "immutable_transition_source" not in observed.split(":", 1)[1]
    assert observed.split(":", 1)[1], "the sentinel must name at least one cause"
