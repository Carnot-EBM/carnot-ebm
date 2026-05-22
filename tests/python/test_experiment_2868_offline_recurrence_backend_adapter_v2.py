"""Tests for Exp 2868 offline recurrence backend adapter.

Spec: REQ-LEARN-2868,
      SCENARIO-LEARN-2868.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from carnot.eval.offline_recurrence_backend_adapter_v2 import (
    BACKEND_MODULE_PATH,
    REQUIRED_ARTIFACT_FIELDS,
    ExperimentConfig,
    OfflineRecurrenceReplayBackend,
    load_smoke_trace_rows,
    run_experiment,
)


def test_scenario_learn_2868_replay_emits_energy_summary_without_mutation() -> None:
    """SCENARIO-LEARN-2868: replay summarizes revised energies without model mutation."""

    rows = [
        {
            "example_id": "fover-1",
            "source": "fover",
            "energy_before": 1.0,
            "revised_energy": 0.25,
            "correctness_before": False,
            "correctness_after": False,
            "localized_violations": ["arithmetic_gap"],
        },
        {
            "example_id": "fover-2",
            "energy_before": 0.5,
            "energy_after_each_loop": [0.4, 0.35],
            "correctness_before": True,
            "correctness_after": True,
        },
    ]
    original_rows = copy.deepcopy(rows)

    replay = OfflineRecurrenceReplayBackend().replay(rows)

    assert rows == original_rows
    assert replay["backend_module_path"] == BACKEND_MODULE_PATH
    assert replay["backend_api"]["callable"] == "OfflineRecurrenceReplayBackend.replay"
    assert replay["n_rows"] == 2
    assert replay["energy_delta_mean"] == pytest.approx(0.45)
    assert replay["per_loop_energy_summary"]["loop_1"]["n"] == 2
    assert replay["per_loop_energy_summary"]["loop_1"]["mean_energy"] == pytest.approx(0.325)
    assert replay["per_loop_energy_summary"]["loop_2"]["mean_delta_from_initial"] == pytest.approx(
        0.15
    )
    assert replay["per_example_trace"][0]["energy_after_each_loop"] == [0.25]
    assert replay["per_example_trace"][1]["source"] == "verifier_trace"
    assert replay["live_model_invoked"] is False
    assert replay["no_model_weight_mutation"] is True
    assert replay["replay_smoke_passed"] is True


def test_req_learn_2868_replay_accepts_verifier_scores_and_rejects_bad_rows() -> None:
    """REQ-LEARN-2868-1/2: scored trace rows normalize, malformed rows fail closed."""

    replay = OfflineRecurrenceReplayBackend().replay(
        [
            {
                "case_id": "case-a",
                "verifier_scores": {
                    "tier0r_curry_howard": 0.9,
                    "tier0u_logical_consistency": 0.2,
                    "tier0s_arithmetic_gap": 0.4,
                },
                "correctness_before": False,
                "correctness_after": False,
            }
        ]
    )

    trace = replay["per_example_trace"][0]
    assert trace["example_id"] == "case-a"
    assert trace["energy_before"] == pytest.approx(0.5)
    assert trace["energy_after_each_loop"] == [pytest.approx(0.2)]
    assert trace["localized_violations"] == ["tier0r_curry_howard"]

    with pytest.raises(ValueError, match="energy"):
        OfflineRecurrenceReplayBackend().replay([{"example_id": "bad-row"}])

    with pytest.raises(ValueError, match="revised energy"):
        OfflineRecurrenceReplayBackend().replay([{"example_id": "bad-row", "energy_before": 1.0}])


def test_req_learn_2868_replay_branch_defaults() -> None:
    """REQ-LEARN-2868-2: default IDs, labels, loop fallback, and guards are stable."""

    with pytest.raises(ValueError, match="max_loops"):
        OfflineRecurrenceReplayBackend(max_loops=0)

    replay = OfflineRecurrenceReplayBackend().replay(
        [
            {
                "label": "correct",
                "energy_before": 0.0,
                "energy_after_each_loop": [],
                "early_exit_reason": "already_passed",
            },
            {
                "trace_id": "flat",
                "label": "incorrect",
                "energy_before": 0.4,
                "energy_after": 0.4,
            },
            {
                "trace_id": "missing-label",
                "energy_before": 0.2,
                "energy_after": 0.2,
            },
        ]
    )

    first, second, third = replay["per_example_trace"]
    assert first["example_id"] == "trace-0"
    assert first["energy_after_each_loop"] == [0.0]
    assert first["correctness_before"] is True
    assert first["correctness_after"] is True
    assert first["early_exit_reason"] == "already_passed"
    assert second["example_id"] == "flat"
    assert second["correctness_before"] is False
    assert second["early_exit_reason"] == "offline_no_energy_improvement"
    assert third["correctness_before"] is False
    assert third["correctness_after"] is False


def test_req_learn_2868_load_smoke_trace_rows_uses_existing_fover_shape(tmp_path: Path) -> None:
    """REQ-LEARN-2868-1: FoVer-shaped verifier traces become replay rows."""

    assert load_smoke_trace_rows(tmp_path, n_rows=2, seed=2868) == []

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "fover_corpus.jsonl").write_text("\n{}\n", encoding="utf-8")
    assert load_smoke_trace_rows(tmp_path, n_rows=2, seed=2868) == []

    (data_dir / "fover_corpus.jsonl").write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "question_id": "q1",
                        "source": "fover_v4",
                        "label": "incorrect",
                        "confidence": 0.8,
                        "verifier": "heuristic",
                    }
                ),
                json.dumps(
                    {
                        "question_id": "q2",
                        "label": "correct",
                        "confidence": 0.75,
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    rows = load_smoke_trace_rows(tmp_path, n_rows=3, seed=0)
    assert [row["example_id"] for row in rows] == ["q1", "q2", "q1"]
    assert rows[0]["localized_violations"] == ["heuristic"]
    assert rows[1]["source"] == "fover"
    assert rows[1]["correctness_before"] is True


def test_req_learn_2868_run_experiment_writes_required_artifact(tmp_path: Path) -> None:
    """REQ-LEARN-2868-3: run_experiment writes the required Exp 2868 schema."""

    rows = [
        {
            "example_id": "smoke-1",
            "source": "unit",
            "energy_before": 0.8,
            "revised_energy": 0.3,
            "correctness_before": False,
            "correctness_after": True,
        }
    ]

    artifact = run_experiment(
        ExperimentConfig(
            repo_root=tmp_path,
            results_dir=tmp_path / "results",
            started_at=1.0,
            clock=lambda: 1.125,
        ),
        trace_rows=rows,
        tests_run=["unit-test-placeholder"],
    )

    assert set(REQUIRED_ARTIFACT_FIELDS).issubset(artifact)
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["offline_recurrence_backend_ready"] is True
    assert artifact["live_recurrence_backend_ready"] is False
    assert artifact["backend_module_path"] == BACKEND_MODULE_PATH
    assert artifact["replay_smoke_passed"] is True
    assert artifact["live_model_invoked"] is False
    assert artifact["no_model_weight_mutation"] is True
    assert artifact["tests_run"] == ["unit-test-placeholder"]
    assert artifact["duration_s"] == pytest.approx(0.125)
    assert len(artifact["reproducibility_checksum"]) == 64

    output = tmp_path / "results" / "experiment_2868_offline_recurrence_backend_adapter_v2.json"
    assert json.loads(output.read_text(encoding="utf-8")) == artifact

    repeat = run_experiment(
        ExperimentConfig(repo_root=tmp_path, results_dir=tmp_path / "results"),
        trace_rows=rows,
        tests_run=["unit-test-placeholder"],
        write=False,
    )
    assert repeat["reproducibility_checksum"] == artifact["reproducibility_checksum"]


def test_req_learn_2868_run_experiment_blocks_without_trace_rows(tmp_path: Path) -> None:
    """REQ-LEARN-2868-3: missing offline trace rows reports a blocked artifact."""

    artifact = run_experiment(
        ExperimentConfig(repo_root=tmp_path, results_dir=tmp_path / "results"),
        write=False,
    )

    assert artifact["honest_verdict"] == "blocked_no_verifier_trace_rows"
    assert artifact["offline_recurrence_backend_ready"] is False
    assert artifact["backend_module_path"] is None
    assert artifact["replay_smoke_passed"] is False
    assert artifact["n_rows"] == 0
    assert artifact["per_example_trace"] == []
