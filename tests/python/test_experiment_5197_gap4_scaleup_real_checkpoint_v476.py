"""Tests for Exp 5197 GAP-4 real checkpoint continuation.

Spec refs: REQ-REPORT-5197, SCENARIO-REPORT-5197,
SCENARIO-REPORT-5197-LOCAL-GENERATOR.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_5197_gap4_scaleup_real_checkpoint_v476 as mod


JsonDict = dict[str, Any]
REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "research-reporting" / "spec.md"


def _row(index: int, *, vote: bool, gated: bool, domain: str = "arc1") -> JsonDict:
    return {
        "pilot_key": f"{domain}:{index}:task_{index}",
        "domain": domain,
        "task": f"task_{index}",
        "entry_i": index,
        "cluster_id": f"{domain}:task_{index}",
        "vote_top2": vote,
        "gated_top2": gated,
        "demo_perfect": gated,
        "pred_is_gold": gated and not vote,
        "pred_in_pool": gated,
        "oracle_hit": vote or gated,
        "n_cands": 3,
    }


def _candidate(grid: list[list[int]], votes: int, correct: bool) -> JsonDict:
    return {"grid": grid, "votes": votes, "correct": correct}


def test_req_report_5197_spec_declares_real_checkpoint_contract() -> None:
    """REQ-REPORT-5197: OpenSpec declares the v476 checkpoint artifact contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    for marker in (
        "REQ-REPORT-5197",
        "SCENARIO-REPORT-5197",
        "SCENARIO-REPORT-5197-LOCAL-GENERATOR",
        mod.RESULT_RELATIVE_PATH,
        mod.CHECKPOINT_RELATIVE_PATH,
    ):
        assert marker in spec
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert field in spec
        assert principle["principle"] in spec


def test_scenario_report_5197_checkpoint_is_json_list_and_skips_prior_rows(
    tmp_path: Path,
) -> None:
    """SCENARIO-REPORT-5197: only unscored rows are added and checkpointed."""

    prior = [_row(0, vote=True, gated=True)]
    candidates = prior + [_row(1, vote=False, gated=True), _row(2, vote=False, gated=False)]

    rows, partial, remaining, new_rows, wrote = mod.score_new_rows_checkpointed(
        root=tmp_path,
        prior_rows=prior,
        candidate_rows=candidates,
        now=lambda: 0.0,
        soft_budget_s=1000.0,
    )

    checkpoint_path = tmp_path / mod.CHECKPOINT_RELATIVE_PATH
    checkpoint_payload = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    assert isinstance(checkpoint_payload, list)
    assert [row["pilot_key"] for row in rows] == [
        "arc1:0:task_0",
        "arc1:1:task_1",
        "arc1:2:task_2",
    ]
    assert partial is False
    assert remaining == []
    assert new_rows == 2
    assert wrote is True
    assert len(checkpoint_payload) == 3
    assert mod.load_checkpoint(tmp_path) == rows

    legacy = tmp_path / mod.CHECKPOINT_RELATIVE_PATH
    legacy.write_text(json.dumps({"rows": [_row(3, vote=True, gated=True)]}), encoding="utf-8")
    assert mod.load_checkpoint(tmp_path)[0]["pilot_key"] == "arc1:3:task_3"


def test_req_report_5197_exact_test_uses_scipy_min6_rule() -> None:
    """REQ-REPORT-5197: scipy binomtest preserves the zero-loss min-6 floor."""

    six = [_row(i, vote=False, gated=True) for i in range(6)]
    five = six[:5]
    none = [_row(9, vote=True, gated=True)]

    assert mod.exact_test(six) == {
        "wins": 6,
        "losses": 0,
        "ties": 0,
        "p_value_two_sided": 0.03125,
        "passes_min6_rule": True,
    }
    assert mod.exact_test(five)["passes_min6_rule"] is False
    assert mod.exact_test(five)["p_value_two_sided"] == 0.0625
    assert mod.exact_test(none)["p_value_two_sided"] == 1.0


def test_scenario_report_5197_local_generator_summarizes_real_call_rows(
    tmp_path: Path,
) -> None:
    """SCENARIO-REPORT-5197-LOCAL-GENERATOR: local calls produce win/loss counts."""

    checkpoint = {
        "local_model_used": "Qwen3.6-35B-A3B",
        "tasks": {
            "fallback": [{"draw_index": 0, "code": "", "status": "no_code"}],
            "loss": [{"draw_index": 0, "code": "loss", "status": "graded"}],
            "win": [
                {"draw_index": 0, "code": "win", "status": "graded"},
                {"draw_index": 1, "code": "", "status": "no_code"},
            ],
        },
    }
    checkpoint_path = tmp_path / mod.LOCAL_QWEN_CHECKPOINT_RELATIVE_PATH
    checkpoint_path.parent.mkdir(parents=True)
    checkpoint_path.write_text(json.dumps(checkpoint), encoding="utf-8")

    entries = [
        {
            "task": "win",
            "candidates": [
                _candidate([[0]], 10, False),
                _candidate([[1]], 9, False),
                _candidate([[2]], 1, True),
            ],
        },
        {
            "task": "loss",
            "candidates": [
                _candidate([[0]], 10, False),
                _candidate([[1]], 9, True),
                _candidate([[2]], 1, False),
            ],
        },
        {
            "task": "fallback",
            "candidates": [
                _candidate([[0]], 10, True),
                _candidate([[1]], 1, False),
            ],
        },
    ]

    def scorer(entry: JsonDict, raw_text: str) -> JsonDict:
        if "win" in raw_text:
            return {"status": "graded", "demo_perfect": True, "pred_hash": "[[2]]"}
        if "loss" in raw_text:
            return {"status": "graded", "demo_perfect": True, "pred_hash": "[[2]]"}
        return {"status": "no_code", "demo_perfect": False, "pred_hash": None}

    result = mod.score_local_call_rows(
        root=tmp_path,
        target_n=3,
        prompt_loader=lambda _root: entries,
        scorer=scorer,
        hash_fn=lambda grid: json.dumps(grid),
    )

    assert result["n_calls"] == 3
    assert result["model_used"] == mod.LOCAL_MODEL_USED
    assert result["discordant_wins"] == 1
    assert result["discordant_losses"] == 1
    assert result["source_checkpoint_model"] == "Qwen3.6-35B-A3B"
    assert [row["task"] for row in result["scored_rows"]] == ["fallback", "loss", "win"]


def test_req_report_5197_builds_valid_pool_exhausted_artifact() -> None:
    """SCENARIO-REPORT-5197: pool exhaustion is reported without crossing the floor."""

    rows = [_row(i, vote=False, gated=True) for i in range(4)]
    rows.extend(_row(10 + i, vote=True, gated=True, domain="arc2") for i in range(2))
    local = {
        "n_calls": 30,
        "model_used": mod.LOCAL_MODEL_USED,
        "discordant_wins": 0,
        "discordant_losses": 0,
    }
    artifact = mod.build_artifact(
        scaleup_rows=rows,
        prior_n=6,
        new_rows_scored=0,
        source_pool_rows_available=6,
        checkpoint_file_written=True,
        local_generator_result=local,
        duration_s=1.5,
        partial=False,
        remaining_rows=[],
        source_artifacts=[{"path": "results/source.json", "exists": True}],
    )

    assert artifact["n_reached"]["value"] == 6
    assert artifact["source_pool_exhausted_before_new_rows"] is True
    assert artifact["exact_test_discordant_wins"]["value"] == 4
    assert artifact["exact_test_discordant_losses"]["value"] == 0
    assert artifact["exact_test_p_value_two_sided"]["value"] == 0.125
    assert artifact["exact_test_passes_min6_rule"]["value"] is False
    assert artifact["inference_substrate"]["value"] == mod.INFERENCE_SUBSTRATE
    assert artifact["gap4_status_recommendation"] == "scale_up_recommended"
    assert (
        artifact["decentralization_tier_local_generator_result"]["value"][
            "closed_weight_cloud_generator_comparison"
        ]["codex_first_discordant_wins"]
        == 4
    )
    assert artifact["honest_verdict"].startswith("complete_")
    assert artifact["reproducibility_checksum"]["value"] == mod.payload_checksum(artifact)
    assert mod.artifact_schema_errors(artifact) == []


def test_artifact_schema_rejects_required_shape_errors() -> None:
    """REQ-REPORT-5197: schema checks protect required v476 fields."""

    artifact = mod.build_artifact(
        scaleup_rows=[_row(i, vote=False, gated=True) for i in range(6)],
        prior_n=6,
        new_rows_scored=0,
        source_pool_rows_available=6,
        checkpoint_file_written=True,
        local_generator_result={
            "n_calls": 30,
            "model_used": mod.LOCAL_MODEL_USED,
            "discordant_wins": 0,
            "discordant_losses": 0,
        },
        duration_s=0.0,
        partial=False,
        remaining_rows=[],
        source_artifacts=[],
    )
    bad = dict(artifact)
    bad["honest_verdict"] = "not_terminal"
    bad["field_principles"] = {}
    bad["n_reached"] = {"value": 999}
    bad["checkpoint_file_written"] = {"value": False}
    bad["exact_test_discordant_wins"] = {"value": 6}
    bad["exact_test_discordant_losses"] = {"value": 0}
    bad["exact_test_p_value_two_sided"] = {"value": True}
    bad["exact_test_passes_min6_rule"] = {"value": False}
    bad["decentralization_tier_local_generator_result"] = {"value": {"n_calls": 1}}
    bad["random_seed"] = {"value": 0}
    bad["inference_substrate"] = {"value": "live_llm_inference"}
    bad["reproducibility_checksum"] = {"value": "sha256:bad"}

    errors = mod.artifact_schema_errors(bad)

    assert "honest_verdict_terminal_prefix" in errors
    assert "field_principles" in errors
    assert "n_reached_bounds" in errors
    assert "checkpoint_file_written_true" in errors
    assert "exact_test_passes_min6_rule" in errors
    assert "exact_test_p_value_two_sided" in errors
    assert "decentralization_tier_local_generator_result.model_used" in errors
    assert "random_seed" in errors
    assert "inference_substrate" in errors
    assert "reproducibility_checksum" in errors

    missing = dict(artifact)
    missing.pop("duration_s")
    missing["reproducibility_checksum"] = {"value": mod.payload_checksum(missing)}
    assert "missing required field duration_s" in mod.artifact_schema_errors(missing)

    with pytest.raises(ValueError):
        mod.write_artifact(Path("/tmp"), bad)


def test_scenario_report_5197_run_writes_artifact_and_checkpoint(tmp_path: Path) -> None:
    """SCENARIO-REPORT-5197: injected end-to-end run writes stable files."""

    rows = [_row(i, vote=True, gated=True) for i in range(2)]
    artifact = mod.run(
        root=tmp_path,
        prior_row_loader=lambda _root: rows,
        candidate_row_loader=lambda _root: rows,
        local_result_loader=lambda _root: {
            "n_calls": 30,
            "model_used": mod.LOCAL_MODEL_USED,
            "discordant_wins": 0,
            "discordant_losses": 0,
        },
        source_artifact_loader=lambda _root: [],
        now=lambda: 10.0,
    )

    result_path = tmp_path / mod.RESULT_RELATIVE_PATH
    checkpoint_path = tmp_path / mod.CHECKPOINT_RELATIVE_PATH
    assert result_path.exists()
    assert checkpoint_path.exists()
    assert json.loads(checkpoint_path.read_text(encoding="utf-8")) == rows
    assert artifact["n_reached"]["value"] == 2
    assert mod.artifact_schema_errors(json.loads(result_path.read_text(encoding="utf-8"))) == []


def test_helpers_handle_invalid_inputs_and_source_artifacts(tmp_path: Path) -> None:
    """REQ-REPORT-5197: helper edges keep bad inputs honest and reproducible."""

    assert mod._read_json(tmp_path / "missing.json") == {}
    broken = tmp_path / "broken.json"
    broken.write_text("{", encoding="utf-8")
    assert mod._read_json(broken) == {}
    assert mod.load_prior_rows(tmp_path) == []
    assert mod.load_candidate_rows(REPO)
    assert mod.resolve_soft_budget_s({}) == mod.DEFAULT_SOFT_BUDGET_S
    assert mod.resolve_soft_budget_s({mod.SOFT_BUDGET_ENV: "5.5"}) == 5.5
    assert mod.resolve_soft_budget_s({mod.SOFT_BUDGET_ENV: "-1"}) == mod.DEFAULT_SOFT_BUDGET_S
    assert mod.resolve_soft_budget_s({mod.SOFT_BUDGET_ENV: "bad"}) == mod.DEFAULT_SOFT_BUDGET_S

    source_file = tmp_path / mod.PRIOR_RESULT_RELATIVE_PATH
    source_file.parent.mkdir(parents=True)
    source_file.write_text("{}", encoding="utf-8")
    sources = mod.describe_source_artifacts(tmp_path)
    assert sources[0]["exists"] is True
    assert sources[0]["sha256"].startswith("sha256:")
    assert any(row["exists"] is False for row in sources[1:])

    list_checkpoint = tmp_path / mod.CHECKPOINT_RELATIVE_PATH
    list_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    list_checkpoint.write_text("not json", encoding="utf-8")
    assert mod.load_checkpoint(tmp_path) == []


def test_req_report_5197_edge_branches_preserve_checkpoint_and_schema_honesty(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-REPORT-5197: defensive branches remain explicit and covered."""

    assert mod._wrapped_value("plain") == "plain"
    assert mod.payload_checksum({"experiment": mod.EXPERIMENT}).startswith("sha256:")

    prior_path = tmp_path / mod.PRIOR_RESULT_RELATIVE_PATH
    prior_path.parent.mkdir(parents=True)
    prior_path.write_text(
        json.dumps({"scaleup_rows": [_row(7, vote=True, gated=True), "bad"]}),
        encoding="utf-8",
    )
    assert [row["pilot_key"] for row in mod.load_prior_rows(tmp_path)] == ["arc1:7:task_7"]

    checkpoint_path = tmp_path / mod.CHECKPOINT_RELATIVE_PATH
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path.write_text(json.dumps([_row(1, vote=True, gated=True)]), encoding="utf-8")
    ticks = iter([0.0, 1.0])
    rows, partial, remaining, new_rows, wrote = mod.score_new_rows_checkpointed(
        root=tmp_path,
        prior_rows=[_row(0, vote=True, gated=True)],
        candidate_rows=[
            _row(0, vote=True, gated=True),
            _row(1, vote=True, gated=True),
            _row(2, vote=False, gated=True),
            _row(3, vote=False, gated=True),
        ],
        now=lambda: next(ticks),
        soft_budget_s=0.5,
    )
    assert [row["pilot_key"] for row in rows] == ["arc1:0:task_0", "arc1:1:task_1"]
    assert [row["pilot_key"] for row in remaining] == ["arc1:2:task_2", "arc1:3:task_3"]
    assert partial is True
    assert new_rows == 0
    assert wrote is True

    assert mod._selected_candidate_index([_candidate([[1]], 1, False)], "[[2]]", json.dumps) is None
    assert mod._selected_local_samples({"tasks": "bad"}) == []
    assert mod._selected_local_samples({"tasks": {"bad": "x"}}, target_n=0) == []
    round_robin = mod._selected_local_samples(
        {
            "tasks": {
                "a": [{"draw_index": 0}, {"draw_index": 1}],
                "bad": "x",
                "b": [{"draw_index": 0}, {"draw_index": 2}],
                "empty": [],
            }
        },
        target_n=3,
    )
    assert [(row["task"], row["draw_index"]) for row in round_robin] == [
        ("a", 0),
        ("b", 0),
        ("a", 1),
    ]
    assert (
        len(
            mod._selected_local_samples(
                {"tasks": {"a": [{"draw_index": 0}, {"draw_index": 1}], "bad": "x"}},
                target_n=10,
            )
        )
        == 2
    )

    local_checkpoint = tmp_path / "local.json"
    local_checkpoint.write_text(
        json.dumps({"tasks": {"missing": [{"draw_index": 0, "code": "print(1)"}]}}),
        encoding="utf-8",
    )
    monkeypatch.setattr(mod.exp5177, "_gap4_helpers", lambda: (None, None, json.dumps))
    local_result = mod.score_local_call_rows(
        root=tmp_path,
        checkpoint_rel_path="local.json",
        target_n=1,
        prompt_loader=lambda _root: [],
        scorer=lambda _entry, _text: {},
    )
    assert local_result["scored_rows"] == [
        {
            "task": "missing",
            "status": "missing_prompt_entry",
            "vote_top2": False,
            "local_gated_top2": False,
        }
    ]

    empty_artifact = mod.build_artifact(
        scaleup_rows=[],
        prior_n=0,
        new_rows_scored=0,
        source_pool_rows_available=0,
        checkpoint_file_written=True,
        local_generator_result={
            "n_calls": 0,
            "model_used": mod.LOCAL_MODEL_USED,
            "discordant_wins": 0,
            "discordant_losses": 0,
        },
        duration_s=-1.0,
        partial=False,
        remaining_rows=[],
        source_artifacts=[],
    )
    assert empty_artifact["gap4_status_recommendation"] == "checkpoint_progress_only"
    assert empty_artifact["duration_s"] == 0.0
    assert mod.artifact_schema_errors(empty_artifact) == []

    drifted = dict(empty_artifact)
    drifted["scaleup_rows"] = [_row(0, vote=False, gated=True)]
    drifted["exact_test_discordant_wins"] = {"value": 0}
    drifted["exact_test_discordant_losses"] = {"value": 1}
    drifted["decentralization_tier_local_generator_result"] = {"value": []}
    errors = mod.artifact_schema_errors(drifted)
    assert "exact_test_discordant_wins" in errors
    assert "exact_test_discordant_losses" in errors
    assert "decentralization_tier_local_generator_result" in errors

    no_rows = dict(empty_artifact)
    no_rows["scaleup_rows"] = "not a list"
    no_rows["exact_test_discordant_wins"] = {"value": 6}
    no_rows["exact_test_discordant_losses"] = {"value": 0}
    no_rows["exact_test_p_value_two_sided"] = {"value": 0.03125}
    no_rows["exact_test_passes_min6_rule"] = {"value": False}
    assert "exact_test_passes_min6_rule" in mod.artifact_schema_errors(no_rows)
