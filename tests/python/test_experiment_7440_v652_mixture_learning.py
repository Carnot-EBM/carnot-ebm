"""Tests for REQ-AUTO-7440 and SCENARIO-AUTO-7440-*.

These tests keep causal online updates separate from later evaluation. They
also treat source groups, rather than fitted seeds, as independent evidence.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_7440_v652_mixture_learning as exp


ROOT = Path(__file__).resolve().parents[2]


def _rows(count: int) -> list[dict[str, Any]]:
    """Build a label-bearing stream whose prediction fields do not use labels."""

    output = []
    for index in range(count):
        output.append(
            {
                "observation_id": f"obs-{index:03d}",
                "row_key": f"row-{index:03d}",
                "group_id": f"group-{index:03d}",
                "task_type": ("QA", "Summary", "Data2txt")[index % 3],
                "features": {
                    name: ((index + offset) % 11) / 10.0
                    for offset, name in enumerate(exp.SOURCE_FEATURE_NAMES)
                },
                "label": index % 2,
            }
        )
    return output


def _passing_receipts() -> list[dict[str, Any]]:
    """Name each command required by a complete terminal fixture."""

    return [
        {"name": name, "required": True, "passed": True, "exit_code": 0}
        for name in (*exp.AFFECTED_CHECK_NAMES, *exp.TERMINAL_CHECK_NAMES)
    ]


def test_req_auto_7440_spec_and_structured_prerequisites() -> None:
    """REQ-AUTO-7440 exists and both same-milestone producers authenticate."""

    text = (ROOT / exp.SPEC_PATH).read_text(encoding="utf-8")
    assert "### REQ-AUTO-7440:" in text
    for number in range(1, 6):
        assert f"SCENARIO-AUTO-7440-{number:02d}" in text
    checks, hashes, loaded = exp.collect_preconditions(ROOT)
    assert checks and all(row["passed"] for row in checks)
    assert loaded["prototype"]["mixture_prototype_ready_score"] == 1
    assert loaded["decisions"]["decision_capture_complete_score"] == 1
    assert hashes[exp.PROTOTYPE_PATH.as_posix()]["original_verdict_class"] == "circular_positive"
    assert hashes[exp.DECISIONS_PATH.as_posix()]["original_flagged_adversarial"] is False


def test_scenario_auto_7440_02_uniform_schedule_has_exact_budget() -> None:
    """SCENARIO-AUTO-7440-02 gives every row a positive known propensity."""

    rows = _rows(70)
    first = exp.build_uniform_reveal_schedule(rows, seed=7440, block_size=32)
    second = exp.build_uniform_reveal_schedule(rows, seed=7440, block_size=32)
    assert first == second
    by_block: dict[int, list[dict[str, Any]]] = {}
    for row in first:
        by_block.setdefault(row["block_index"], []).append(row)
        assert row["propensity"] > 0.0
    assert [sum(row["revealed"] for row in block) for block in by_block.values()] == [8, 8, 1]
    assert [block[0]["propensity"] for block in by_block.values()] == pytest.approx(
        [0.25, 0.25, 1 / 6]
    )
    with pytest.raises(ValueError, match="positive"):
        exp.build_uniform_reveal_schedule(rows, seed=1, block_size=0)
    duplicate = deepcopy(rows)
    duplicate[1]["observation_id"] = duplicate[0]["observation_id"]
    with pytest.raises(ValueError, match="unique"):
        exp.build_uniform_reveal_schedule(duplicate, seed=1)


@pytest.mark.parametrize("delay", [0, 8])
def test_scenario_auto_7440_01_replay_is_causal_and_exactly_once(delay: int) -> None:
    """SCENARIO-AUTO-7440-01 commits stored losses after prediction once."""

    rows = _rows(40)
    schedule = exp.build_uniform_reveal_schedule(rows, seed=7440)
    result = exp.replay_cell(
        exp.build_fixture_states(seed=65201),
        rows,
        schedule,
        ordering="hash_order",
        delay=delay,
        seed=65201,
    )
    assert len(result["prediction_rows"]) == len(rows) * len(exp.ARMS)
    # Four mixtures plus the standalone adaptive spline consume each reveal.
    assert len(result["feedback_rows"]) == 10 * 5
    assert result["future_label_reads"] == 0
    assert result["duplicate_updates"] == 0
    assert all(row["prediction_before_feedback"] for row in result["prediction_rows"])
    assert all(
        row["shadow_only"] and row["certified_safe"] is False for row in result["prediction_rows"]
    )
    assert all(
        row["probability_source"] == "stored_at_prediction" for row in result["feedback_rows"]
    )
    assert all(row["arrival_index"] >= row["prediction_index"] for row in result["feedback_rows"])
    assert all(row["update_count"] == 1 for row in result["feedback_rows"])
    assert all(
        row["parent_state_hash"] != row["child_state_hash"] for row in result["checkpoint_lineage"]
    )
    learned = [row for row in result["weight_trajectory_rows"] if row["arm"] == exp.LEARNED_MIXTURE]
    equal = [row for row in result["weight_trajectory_rows"] if row["arm"] == exp.EQUAL_MIXTURE]
    assert learned and any(row["weights_before"] != row["weights_after"] for row in learned)
    assert equal and all(set(row["weights_after"].values()) == {0.25} for row in equal)


def test_scenario_auto_7440_04_shadow_actions_report_undefined_risk() -> None:
    """SCENARIO-AUTO-7440-04 keeps adaptive typed actions out of deployment."""

    thresholds = {"accept_threshold": 0.8, "reject_threshold": 0.2}
    assert exp.shadow_action(0.9, thresholds) == "accept"
    assert exp.shadow_action(0.1, thresholds) == "reject"
    assert exp.shadow_action(0.5, thresholds) == "escalate"
    metrics = exp.reduce_arm_metrics(
        [
            {
                "label": 1,
                "probability": 0.9,
                "proposed_action": "accept",
                "revealed": True,
                "propensity": 0.25,
                "domain_changed": False,
            },
            {
                "label": 0,
                "probability": 0.1,
                "proposed_action": "reject",
                "revealed": False,
                "propensity": 0.25,
                "domain_changed": True,
            },
        ]
    )
    assert metrics["shadow_coverage"] == 1.0
    assert metrics["deployment_policy"] == "all_escalate"
    assert metrics["deployment_coverage"] == 0.0
    empty = exp.reduce_arm_metrics(
        [
            {
                "label": 1,
                "probability": 0.6,
                "proposed_action": "escalate",
                "revealed": True,
                "propensity": 0.25,
                "domain_changed": False,
            }
        ]
    )
    assert empty["shadow_harmful_action_rate"] is None
    assert empty["selected_action_count"] == 0
    with pytest.raises(ValueError, match="non-empty"):
        exp.reduce_arm_metrics([])


def test_scenario_auto_7440_03_bootstrap_averages_seeds_within_group() -> None:
    """SCENARIO-AUTO-7440-03 resamples groups in one 12-contrast family."""

    rows = exp.synthetic_metric_rows(groups=96, seeds=5)
    intervals = exp.paired_moving_block_intervals(rows, draws=200, seed=7440)
    assert len(intervals) == 24
    assert {row["block_length"] for row in intervals} == {32, 64}
    assert all(row["family_contrasts"] == 12 for row in intervals)
    assert all(row["paired_source_groups"] == 96 for row in intervals)
    assert all(row["fit_seeds_averaged_before_resampling"] == 5 for row in intervals)
    with pytest.raises(ValueError, match="draws"):
        exp.paired_moving_block_intervals(rows, draws=0)
    with pytest.raises(ValueError, match="complete"):
        exp.paired_moving_block_intervals(rows[:-1], draws=10)


def test_scenario_auto_7440_05_value_gates_are_conjunctive() -> None:
    """SCENARIO-AUTO-7440-05 preserves a valid null when one benefit fails."""

    reports, intervals, controls = exp.synthetic_reduction_inputs(passing=True)
    positive = exp.reduce_online_value(reports, intervals, controls)
    assert positive["online_capture_complete_score"] == 1
    assert positive["online_value_score"] == 1
    assert all(row["passed"] for row in positive["gates"])

    intervals = deepcopy(intervals)
    intervals[0]["upper"] = 0.0
    null = exp.reduce_online_value(reports, intervals, controls)
    assert null["online_capture_complete_score"] == 1
    assert null["online_value_score"] == 0
    assert any(not row["passed"] for row in null["gates"])


def test_req_auto_7440_fixture_artifact_reduces_and_rejects_mutation(tmp_path: Path) -> None:
    """REQ-AUTO-7440 binds raw shards, declarations, controls, and reduction."""

    artifact = exp.build_fixture_artifact(tmp_path, validation_receipts=_passing_receipts())
    assert artifact["MODEL_SPECS"] == []
    assert artifact["model_invoked"] is False
    assert artifact["invocation_counts"] == exp.ZERO_INVOCATION_COUNTS
    assert artifact["inference_substrate_class"] == "no_model_load"
    assert artifact["execution_venue"] == "host"
    assert artifact["online_capture_complete_score"] == 1
    assert artifact["promotion_score"] == 0
    assert artifact["continuous_self_learning_task"] is True
    assert artifact["shadow_decisions_only"] is True
    assert exp.validate_artifact(artifact, root=tmp_path) == []
    assert exp.independent_reduce(artifact, root=tmp_path) == artifact["independent_reduction"]

    changed = deepcopy(artifact)
    shard = tmp_path / changed["row_shards"][0]["path"]
    shard.write_text(json.dumps({"changed": True}) + "\n", encoding="utf-8")
    assert "row_shard_invalid" in exp.validate_artifact(changed, root=tmp_path)


def test_req_auto_7440_blocked_artifact_names_exact_failed_field() -> None:
    """REQ-AUTO-7440 reports external absence as blocked, never partial."""

    checks = exp.upstream_field_checks(
        {"mixture_prototype_ready_score": 0, "verdict_class": None},
        {
            "decision_capture_complete_score": 1,
            "verdict_class": "null",
            "flagged_adversarial": False,
        },
    )
    failed = next(row for row in checks if not row["passed"])
    blocked = exp.build_blocked_artifact(failed)
    assert blocked["verdict_class"] == "blocked"
    assert blocked["honest_verdict"].startswith("blocked_")
    assert blocked["gate_check_summary"]["field"] == "mixture_prototype_ready_score"
    assert blocked["gate_check_summary"]["observed"] == 0


def test_req_auto_7440_cli_modes_are_strict(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """REQ-AUTO-7440 exposes only the declared run and fresh-process readers."""

    parsed = exp.parse_args(["--date", exp.RUN_DATE, "--root", str(tmp_path)])
    assert parsed.root == tmp_path
    with pytest.raises(SystemExit, match="--date"):
        exp.main(["--date", "wrong"])
    candidate = tmp_path / "candidate.json"
    candidate.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(exp, "_load_object", lambda _path: {"independent_reduction": {"ok": 1}})
    monkeypatch.setattr(exp, "validate_artifact", lambda _value, root: [])
    assert (
        exp.main(["--date", exp.RUN_DATE, "--root", str(tmp_path), "--cold-replay", str(candidate)])
        == 0
    )
    monkeypatch.setattr(exp, "independent_reduce", lambda _value, root: {"ok": 1})
    assert (
        exp.main(
            [
                "--date",
                exp.RUN_DATE,
                "--root",
                str(tmp_path),
                "--independent-reduce",
                str(candidate),
            ]
        )
        == 0
    )
    called: list[tuple[Path, str, Path]] = []
    monkeypatch.setattr(
        exp,
        "run_experiment",
        lambda root, date, output_path: called.append((root, date, output_path)),
    )
    assert exp.main(["--date", exp.RUN_DATE, "--root", str(tmp_path)]) == 0
    assert called == [(tmp_path.resolve(), exp.RUN_DATE, exp.RESULT_PATH)]
    assert "errors" in capsys.readouterr().out


def test_req_auto_7440_defensive_reducer_and_validator_branches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """REQ-AUTO-7440 rejects incomplete pairs, shards, fields, and CLI input."""

    assert exp._load_object(tmp_path / "missing.json") == {}
    sequence = tmp_path / "sequence.json"
    sequence.write_text("[]", encoding="utf-8")
    assert exp._load_object(sequence) == {}

    rows = _rows(8)
    schedule = exp.build_uniform_reveal_schedule(rows, seed=1)
    with pytest.raises(ValueError, match="registered"):
        exp.replay_cell(
            exp.build_fixture_states(seed=65201),
            rows,
            schedule,
            ordering="wrong",
            delay=0,
            seed=65201,
        )
    with pytest.raises(ValueError, match="cover"):
        exp.replay_cell(
            exp.build_fixture_states(seed=65201),
            rows,
            schedule[:-1],
            ordering="hash_order",
            delay=0,
            seed=65201,
        )

    metric_rows = exp.synthetic_metric_rows(groups=8, seeds=5)
    changed_seed = deepcopy(metric_rows)
    target = next(
        row for row in changed_seed if row["arm"] == exp.FROZEN_SPLINE_ARM and row["seed"] == 4
    )
    target["seed"] = 99
    with pytest.raises(ValueError, match="complete seed"):
        exp.paired_moving_block_intervals(changed_seed, draws=2)
    missing_cell = [
        row for row in metric_rows if not (row["ordering"] == "hash_order" and row["delay"] == 0)
    ]
    with pytest.raises(ValueError, match="registered cells"):
        exp.paired_moving_block_intervals(missing_cell, draws=2)

    malformed = tmp_path / "malformed.jsonl"
    malformed.write_text(json.dumps({"row_type": "prediction"}) + "\n", encoding="utf-8")
    manifest = [{"path": "malformed.jsonl", "sha256": exp.sha256_file(malformed), "rows": 2}]
    with pytest.raises(ValueError, match="row_shard"):
        exp._load_row_shards(tmp_path, manifest)
    manifest[0]["rows"] = 1
    with pytest.raises(ValueError, match="required_raw"):
        exp.independent_reduce({"row_shards": manifest}, root=tmp_path)

    artifact = exp.build_fixture_artifact(tmp_path, validation_receipts=_passing_receipts())
    changed = deepcopy(artifact)
    changed.pop("schema")
    changed.update(
        {
            "experiment_id": "wrong",
            "milestone": "wrong",
            "MODEL_SPECS": ["forbidden"],
            "invocation_counts": {},
            "inference_substrate_class": "wrong",
            "execution_venue": "wrong",
            "promotion_score": 1,
            "shadow_decisions_only": False,
            "independent_reduction": {},
            "online_capture_complete_score": 0,
            "online_value_score": 1,
            "validation_receipts": [],
            "reproducibility_checksum": "wrong",
        }
    )
    errors = exp.validate_artifact(changed, root=tmp_path)
    assert set(errors) >= {
        "required_field_missing:schema",
        "artifact_identity_invalid",
        "artifact_schedule_invalid",
        "current_model_declaration_invalid",
        "current_invocation_counts_invalid",
        "inference_substrate_class_invalid",
        "execution_venue_invalid",
        "promotion_score_invalid",
        "shadow_decision_scope_invalid",
        "independent_reduction_mismatch",
        "online_capture_complete_score_mismatch",
        "online_value_score_mismatch",
        "required_validation_incomplete",
        "reproducibility_checksum_mismatch",
    }

    monkeypatch.setattr(exp, "_load_object", lambda _path: {})
    assert (
        exp.main(
            ["--date", exp.RUN_DATE, "--root", str(tmp_path), "--independent-reduce", str(sequence)]
        )
        == 1
    )
    assert "error" in capsys.readouterr().out


def test_scenario_auto_7440_05_terminal_classification_branches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-AUTO-7440-05 separates invalid, positive, and null terminals."""

    artifact = exp.build_fixture_artifact(tmp_path, validation_receipts=_passing_receipts())
    disqualified = exp._build_artifact(
        root=tmp_path,
        preconditions=[],
        source_hashes={},
        row_shards=artifact["row_shards"],
        unit_rows=[],
        validation_receipts=[],
        phase_spans=[],
        started_at="2026-09-20T00:00:00+00:00",
        started_ns=1,
        ended_ns=2,
        bootstrap_draws=2,
        candidate=False,
        fixture=True,
        latency_ns={},
        source_group_count=16,
    )
    assert disqualified["verdict_class"] == "disqualified"
    assert disqualified["online_capture_complete_score"] == 0

    reports, intervals, controls = exp.synthetic_reduction_inputs(passing=True)
    positive_reduction = exp.reduce_online_value(reports, intervals, controls)
    positive_reduction.update(
        {
            "condition_reports": reports,
            "paired_moving_block_intervals": intervals,
            "controls": controls,
            "prediction_row_count": 1,
            "feedback_row_count": 1,
            "lineage_row_count": 1,
            "causal_capture_valid": True,
        }
    )
    monkeypatch.setattr(exp, "independent_reduce", lambda _value, root: positive_reduction)
    positive = exp._build_artifact(
        root=tmp_path,
        preconditions=[],
        source_hashes={},
        row_shards=[],
        unit_rows=[],
        validation_receipts=_passing_receipts(),
        phase_spans=[],
        started_at="2026-09-20T00:00:00+00:00",
        started_ns=1,
        ended_ns=2,
        bootstrap_draws=2,
        candidate=False,
        fixture=True,
        latency_ns={},
        source_group_count=1,
    )
    assert positive["verdict_class"] == "positive"


def test_scenario_auto_7440_01_invalid_lineage_clears_capture(tmp_path: Path) -> None:
    """SCENARIO-AUTO-7440-01 prevents malformed lineage from completing capture."""

    artifact = exp.build_fixture_artifact(tmp_path, validation_receipts=_passing_receipts())
    rows = exp._load_row_shards(tmp_path, artifact["row_shards"])
    next(row for row in rows if row["row_type"] == "checkpoint_lineage")["exactly_once"] = False
    manifests = exp.write_row_shards(
        tmp_path,
        rows,
        relative_dir=Path("invalid_rows"),
        prefix="invalid",
        max_bytes=512_000,
    )
    changed = {**artifact, "row_shards": manifests}
    reduced = exp.independent_reduce(changed, root=tmp_path)
    assert reduced["causal_capture_valid"] is False
    assert reduced["online_capture_complete_score"] == 0
    assert reduced["online_value_score"] == 0
