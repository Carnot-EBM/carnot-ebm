"""Tests for the frozen live-engine generalization audit.

Spec refs: REQ-ARC-WMTE-6981 and SCENARIO-ARC-WMTE-6981-*.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import shutil

import pytest

from carnot import experiment_6981_arc_live_engine_generalization_audit as exp


ROOT = Path(__file__).resolve().parents[2]


def _rows() -> list[dict[str, object]]:
    """Return an ordered prefix and tail with one change and one no-op in each."""

    return [
        {
            "transition_id": "s0",
            "index": 0,
            "grid": [[0, 0], [0, 0]],
            "next_grid": [[1, 0], [0, 0]],
            "action": 1,
            "data": {"x": 0},
            "level_before": 0,
            "level_after": 0,
        },
        {
            "transition_id": "s1",
            "index": 1,
            "grid": [[1, 0], [0, 0]],
            "next_grid": [[1, 0], [0, 0]],
            "action": 2,
            "data": None,
            "level_before": 0,
            "level_after": 0,
        },
        {
            "transition_id": "h0",
            "index": 2,
            "grid": [[0, 0], [1, 0]],
            "next_grid": [[0, 1], [1, 0]],
            "action": 1,
            "data": {"x": 1},
            "level_before": 0,
            "level_after": 0,
        },
        {
            "transition_id": "h1",
            "index": 3,
            "grid": [[0, 1], [1, 0]],
            "next_grid": [[0, 1], [1, 0]],
            "action": 2,
            "data": None,
            "level_before": 0,
            "level_after": 0,
        },
    ]


def _engine_source() -> str:
    """Return one general action rule with a reachable fixture goal."""

    return """import numpy as np
def engine(grid, action, data):
    g = np.asarray(grid).copy()
    if int(action) == 1:
        g[0, int((data or {}).get('x', 1))] = 1
    return g
def is_level_complete(grid):
    g = np.asarray(grid)
    return bool(g.ndim == 2 and g.shape[1] > 1 and g[0, 1] == 1)
"""


def _write_source(repo: Path, relative: str, value: str | dict[str, object]) -> dict[str, str]:
    """Write one fixture source and return its immutable path and digest."""

    path = repo / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    text = value if isinstance(value, str) else json.dumps(value, sort_keys=True)
    path.write_text(text, encoding="utf-8")
    return {"path": relative, "sha256": exp.sha256_path(path)}


def _write_fixture(repo: Path, *, ts: str = "20260904T030000_000000") -> Path:
    """Write one complete post-6968 manifest row and all content-addressed sources."""

    prior = repo / exp.PRIOR_ARTIFACT_PATH
    prior.parent.mkdir(parents=True, exist_ok=True)
    prior.write_text(
        json.dumps(
            {
                "selected_run_provenance": {
                    "selected": {"engine_emitted_at": "20260904T020536_575501"}
                }
            }
        ),
        encoding="utf-8",
    )
    engine_text = _engine_source()
    engine_hash = exp.sha256_bytes(engine_text.encode())
    engine_sha16 = engine_hash.removeprefix("sha256:")[:16]
    engine = _write_source(
        repo,
        f"results/arc_e3/r11l/attempts/wm_{ts}__{engine_sha16}.py",
        engine_text,
    )
    prompt = _write_source(
        repo,
        f"results/arc_e3/r11l/attempts/prompt_{ts}.json",
        {
            "included_transition_ids": ["s0", "s1"],
            "repair_feedback_transition_ids": [],
            "used_game_source": False,
            "used_hand_derived_rules": False,
            "heldout_revealed_before_generation": False,
        },
    )
    environment = _write_source(
        repo,
        f"results/arc_e3/r11l/attempts/environment_{ts}.json",
        {"game": "r11l", "policy": "e3", "n_ctx": 98304},
    )
    scorer = _write_source(
        repo,
        "scripts/arc_e3_induced_model_quality.py",
        (ROOT / exp.SCORER_PATH).read_text(encoding="utf-8"),
    )
    live_policy = _write_source(
        repo,
        "python/carnot/agentic/arc_competition_agent.py",
        (ROOT / exp.LIVE_POLICY_PATH).read_text(encoding="utf-8"),
    )
    transition_document: dict[str, object] = {
        "schema": "carnot.arc_live_engine_audit_source.v1",
        "engine_sha256": engine_hash,
        "prompt_sha256": prompt["sha256"],
        "environment_sha256": environment["sha256"],
        "scorer_sha256": scorer["sha256"],
        "live_policy_sha256": live_policy["sha256"],
        "prompt_row_ids": ["s0", "s1"],
        "repair_feedback_row_ids": [],
        "rows": _rows(),
        "first_step_candidate_groups": [
            {
                "group_id": "first-0",
                "current_grid": [[0, 0], [0, 0]],
                "shipped_baseline_order": ["noop", "change"],
                "candidates": [
                    {
                        "candidate_id": "noop",
                        "action": 2,
                        "data": None,
                        "realized_first_step_value": 0.0,
                    },
                    {
                        "candidate_id": "change",
                        "action": 1,
                        "data": {"x": 1},
                        "realized_first_step_value": 1.0,
                    },
                ],
            }
        ],
    }
    transitions = _write_source(
        repo,
        f"results/arc_e3/r11l/attempts/transitions_{ts}.json",
        transition_document,
    )
    run_document = {
        "complete": True,
        "policy": "e3",
        "run_started_at": "2026-09-04T02:50:00Z",
        "run_completed_at": "2026-09-04T03:10:00Z",
        "per_game": [{"game": "r11l", "engine_sha256": engine_hash}],
    }
    run = _write_source(repo, f"results/arc_leaderboard_eval_runs/r11l-{ts}.json", run_document)
    registry = repo / exp.REGISTRY_PATH
    registry.parent.mkdir(parents=True, exist_ok=True)
    registry.write_text("games: {}\n", encoding="utf-8")
    manifest = repo / exp.MANIFEST_PATH
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(
        json.dumps(
            {
                "ts": ts,
                "file": Path(engine["path"]).name,
                "sha256_16": engine_sha16,
                "score": -1000,
                "run": run,
                "engine": engine,
                "prompt": prompt,
                "transitions": transitions,
                "environment": environment,
                "scorer": scorer,
                "live_policy": live_policy,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return manifest


def test_req_arc_wmte_6981_spec_precedes_code_and_names_contract() -> None:
    """REQ-ARC-WMTE-6981 declares every field and required scenario."""

    text = (ROOT / exp.SPEC_PATH).read_text(encoding="utf-8")
    section = text[text.index("### REQ-ARC-WMTE-6981") :]

    assert all(f"`{field}`" in section for field in exp.REQUIRED_ARTIFACT_FIELDS)
    assert all(
        f"SCENARIO-ARC-WMTE-6981-{name}" in section
        for name in (
            "CHRONOLOGICAL-SELECTION",
            "HASH-VERIFICATION",
            "SPLIT-PURITY",
            "RESTRICTED-SCORING",
            "FROZEN-CONTROLS",
            "LIVE-REACHABILITY",
            "FIRST-STEP-RANKING",
            "BLOCKED",
            "NO-SOLVE",
        )
    )


def test_scenario_arc_wmte_6981_chronological_selection_ignores_score(tmp_path: Path) -> None:
    """SCENARIO-ARC-WMTE-6981-CHRONOLOGICAL-SELECTION uses time, not quality."""

    manifest = _write_fixture(tmp_path)
    first = json.loads(manifest.read_text(encoding="utf-8"))
    later = deepcopy(first)
    later["ts"] = "20260904T040000_000000"
    later["score"] = 1_000_000
    manifest.write_text(
        manifest.read_text(encoding="utf-8") + json.dumps(later, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    rows, cutoff = exp.discover_post_6968_rows(tmp_path)
    selected, reason = exp.select_earliest_eligible(rows)

    assert cutoff == "20260904T020536_575501"
    assert selected is not None
    assert selected["ts"] == "20260904T030000_000000"
    assert reason == "earliest_eligible_post_exp6968_manifest_timestamp"
    assert len(rows) == 2


def test_scenario_arc_wmte_6981_hash_verification_blocks_changed_bytes(tmp_path: Path) -> None:
    """SCENARIO-ARC-WMTE-6981-HASH-VERIFICATION reports both digests."""

    manifest = _write_fixture(tmp_path)
    row = json.loads(manifest.read_text(encoding="utf-8"))
    prompt = tmp_path / row["prompt"]["path"]
    prompt.write_text(prompt.read_text(encoding="utf-8") + "\n", encoding="utf-8")

    artifact = exp.build_artifact(date="20260904", repo_root=tmp_path)

    assert artifact["verdict_class"] == "blocked"
    failure = next(
        item for item in artifact["gate_check_summary"] if item["failed_check"] == "prompt_hash"
    )
    assert failure["expected_value"] == row["prompt"]["sha256"]
    assert failure["observed_value"] == exp.sha256_path(prompt)
    assert artifact["engine_execution_rows"] == []


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("included_transition_ids", ["s0", "s1", "h0"]),
        ("repair_feedback_transition_ids", ["h0"]),
        ("used_game_source", True),
        ("used_hand_derived_rules", True),
        ("heldout_revealed_before_generation", True),
    ],
)
def test_scenario_arc_wmte_6981_split_purity_fails_closed(
    tmp_path: Path, field: str, value: object
) -> None:
    """SCENARIO-ARC-WMTE-6981-SPLIT-PURITY rejects every forbidden input class."""

    manifest = _write_fixture(tmp_path)
    row = json.loads(manifest.read_text(encoding="utf-8"))
    prompt_path = tmp_path / row["prompt"]["path"]
    prompt = json.loads(prompt_path.read_text(encoding="utf-8"))
    prompt[field] = value
    prompt_path.write_text(json.dumps(prompt, sort_keys=True), encoding="utf-8")
    row["prompt"]["sha256"] = exp.sha256_path(prompt_path)
    transition_path = tmp_path / row["transitions"]["path"]
    transition = json.loads(transition_path.read_text(encoding="utf-8"))
    transition["prompt_sha256"] = row["prompt"]["sha256"]
    transition_path.write_text(json.dumps(transition, sort_keys=True), encoding="utf-8")
    row["transitions"]["sha256"] = exp.sha256_path(transition_path)
    manifest.write_text(json.dumps(row, sort_keys=True) + "\n", encoding="utf-8")

    artifact = exp.build_artifact(date="20260904", repo_root=tmp_path)

    assert artifact["verdict_class"] == "blocked"
    assert any(item["failed_check"] == "split_purity" for item in artifact["gate_check_summary"])
    assert artifact["engine_execution_rows"] == []


def test_scenarios_arc_wmte_6981_restricted_scoring_and_controls(tmp_path: Path) -> None:
    """SCENARIO-ARC-WMTE-6981-RESTRICTED-SCORING and FROZEN-CONTROLS are complete."""

    engine = tmp_path / "engine.py"
    engine.write_text(_engine_source(), encoding="utf-8")
    rows = _rows()

    scored, receipt = exp.execute_engine_fresh(engine, rows, timeout_s=5)
    controls = exp.score_controls(rows[:2], rows[2:])

    assert len(scored) == len(rows)
    assert all(row["terminal"] is True and row["latency_s"] >= 0 for row in scored)
    assert receipt[0]["worker_pid"] != receipt[0]["parent_pid"]
    assert receipt[0]["restricted_process"] is True
    assert {row["control"] for row in controls} == {
        "identity",
        "constant_delta",
        "nearest_shown_delta",
        "row_table_memorization",
    }
    assert all(row["terminal"] is True for row in controls)


def test_scenario_arc_wmte_6981_live_fixture_proves_action_influence(tmp_path: Path) -> None:
    """SCENARIO-ARC-WMTE-6981-LIVE-REACHABILITY needs a production policy call."""

    engine = tmp_path / "engine.py"
    engine.write_text(_engine_source(), encoding="utf-8")
    engine_hash = exp.sha256_path(engine)

    trace, fixture = exp.trace_and_run_live_influence_fixture(engine, engine_hash)

    assert all(row["terminal"] is True for row in trace)
    assert [row["symbol"] for row in trace] == [
        "make_carnot_agent",
        "E3AgentPolicy",
        "load_engine",
        "_world_model_candidates",
        "plan_in_model",
        "next_move",
    ]
    assert fixture[0]["factory_constructed_e3_policy"] is True
    assert fixture[0]["loaded_engine_hash"] == engine_hash
    assert fixture[0]["selected_candidate_name"] == "loaded_world_model.py"
    assert fixture[0]["engine_first_action"] != fixture[0]["no_engine_first_action"]
    assert fixture[0]["reachable"] is True


def test_scenario_arc_wmte_6981_first_step_ranking_cannot_use_future_values(
    tmp_path: Path,
) -> None:
    """SCENARIO-ARC-WMTE-6981-FIRST-STEP-RANKING freezes ranks before outcomes."""

    engine = tmp_path / "engine.py"
    engine.write_text(_engine_source(), encoding="utf-8")
    groups = [
        {
            "group_id": "g0",
            "current_grid": [[0, 0], [0, 0]],
            "shipped_baseline_order": ["noop", "change"],
            "candidates": [
                {"candidate_id": "noop", "action": 2, "data": None, "future_value": 99},
                {
                    "candidate_id": "change",
                    "action": 1,
                    "data": {"x": 1},
                    "future_value": -99,
                },
            ],
        }
    ]

    candidates_a, rankings_a = exp.rank_first_step_groups(engine, groups)
    changed_future = deepcopy(groups)
    changed_future[0]["candidates"][0]["future_value"] = -1000
    changed_future[0]["candidates"][1]["future_value"] = 1000
    candidates_b, rankings_b = exp.rank_first_step_groups(engine, changed_future)

    assert [row["candidate_id"] for row in candidates_a] == ["noop", "change"]
    assert rankings_a[0]["engine_order"] == ["change", "noop"]
    assert rankings_a[0]["shipped_baseline_order"] == ["noop", "change"]
    assert rankings_a[0]["ranking_inputs"] == [
        "current_grid",
        "action",
        "data",
        "frozen_engine_prediction",
    ]
    assert rankings_a[0]["engine_order"] == rankings_b[0]["engine_order"]
    assert rankings_a[0]["terminal"] is True
    assert len(candidates_b) == 2


def test_req_arc_wmte_6981_complete_artifact_has_all_terminal_fields(tmp_path: Path) -> None:
    """REQ-ARC-WMTE-6981 emits the full positive fixture contract without solve credit."""

    _write_fixture(tmp_path)

    artifact = exp.build_artifact(date="20260904", repo_root=tmp_path)

    assert set(exp.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert set(artifact["field_principles"]) == set(exp.REQUIRED_ARTIFACT_FIELDS)
    assert artifact["arc_engine_audit_complete_score"] == 1
    assert artifact["live_path_reachable_score"] == 1
    assert artifact["arc_generalization_positive_score"] == 1
    assert artifact["verdict_class"] == "positive"
    assert artifact["honest_verdict"].startswith("positive_")
    assert artifact["heldout_changing_accuracy"] == 1.0
    assert artifact["heldout_noop_accuracy"] == 1.0
    assert artifact["paired_control_delta_rows"][0]["interval_low"] > 0
    assert artifact["gate_check_summary"] == []
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


def test_scenario_arc_wmte_6981_blocked_real_checkout_is_not_partial() -> None:
    """SCENARIO-ARC-WMTE-6981-BLOCKED does not invent missing archived sources."""

    artifact = exp.build_artifact(date="20260904", repo_root=ROOT)

    assert set(exp.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert artifact["honest_verdict"] == "blocked_arc_live_engine_generalization_audit"
    assert artifact["verdict_class"] == "blocked"
    assert artifact["arc_engine_audit_complete_score"] == 0
    assert artifact["arc_generalization_positive_score"] == 0
    assert artifact["gate_check_summary"]
    assert artifact["engine_execution_rows"] == []
    assert any(
        row["failed_check"] == "eligible_post_exp6968_live_engine"
        for row in artifact["gate_check_summary"]
    )


def test_scenario_arc_wmte_6981_no_solve_and_relative_main_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-ARC-WMTE-6981-NO-SOLVE keeps the registry and output boundary explicit."""

    _write_fixture(tmp_path)
    registry = tmp_path / exp.REGISTRY_PATH
    before = registry.read_bytes()
    monkeypatch.setattr(exp, "REPO_ROOT", tmp_path)

    assert exp.main(["--date", "20260904", "--output", "result.json"]) == 0
    artifact = json.loads((tmp_path / "result.json").read_text(encoding="utf-8"))

    assert registry.read_bytes() == before
    assert artifact["solve_claimed"] is False
    assert artifact["level_claimed"] is False
    assert artifact["registry_updated"] is False
    assert artifact["submitted_to_leaderboard"] is False
