"""Tests for Exp 2857 LoopUS FR-11 self-learning v2.

Spec: REQ-LEARN-2857,
      SCENARIO-LEARN-2857,
      SCENARIO-LEARN-2857-BLOCKED.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from carnot.eval.loopus_fr11_self_learning_v2 import (
    CommandResult,
    ExperimentConfig,
    REQUIRED_ARTIFACT_FIELDS,
    run_experiment,
)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def test_scenario_learn_2857_blocked_missing_exp2856_artifact_writes_schema(
    tmp_path: Path,
) -> None:
    """SCENARIO-LEARN-2857-BLOCKED: missing Exp 2856 blocks without fake metrics."""

    def runner(command: str, cwd: Path) -> CommandResult:
        assert cwd == tmp_path
        if command.startswith("jq -e"):
            return CommandResult(
                2,
                "",
                "jq: error: Could not open file "
                "results/experiment_2856_loopus_recurrence_backend_adapter.json: "
                "No such file or directory",
            )
        return CommandResult(
            1,
            "",
            "FileNotFoundError: results/experiment_2856_loopus_recurrence_backend_adapter.json",
        )

    artifact = run_experiment(
        ExperimentConfig(
            repo_root=tmp_path,
            results_dir=tmp_path / "results",
            started_at=10.0,
            clock=lambda: 10.25,
        ),
        command_runner=runner,
    )

    assert set(REQUIRED_ARTIFACT_FIELDS).issubset(artifact)
    assert artifact["honest_verdict"] == "blocked_missing_exp2856_artifact"
    assert artifact["continuous_self_learning_task"] is True
    assert artifact["fr11_self_learning_ready"] is False
    assert artifact["n_examples"] == 0
    assert artifact["source_counts"] == {}
    assert artifact["max_loops"] == 3
    assert artifact["recurrence_success_rate"] == 0.0
    assert artifact["energy_delta_mean"] == 0.0
    assert artifact["correctness_delta"] == 0.0
    assert artifact["per_loop_energy_summary"] == {}
    assert artifact["early_exit_summary"] == {"blocked_missing_exp2856_artifact": 1}
    assert artifact["memory_hash_before"] == "not_checked_precondition_failed"
    assert artifact["memory_hash_after"] == "not_checked_precondition_failed"
    assert artifact["no_model_weight_mutation"] is True
    assert artifact["model_specs"][0]["status"] == "unavailable"
    assert len(artifact["reproducibility_checksum"]) == 64
    assert len(artifact["preconditions_checked"]) == 3
    assert artifact["preconditions_checked"][1]["passed"] is False
    assert artifact["preconditions_checked"][2]["passed"] is False
    assert artifact["adversarial_verify_passed"] is False
    assert artifact["adversarial_verify_flags"] == [
        "precondition_failed_missing_results/experiment_2856_loopus_recurrence_backend_adapter.json"
    ]
    assert artifact["duration_s"] == pytest.approx(0.25)

    saved = json.loads(
        (tmp_path / "results" / "experiment_2857_loopus_fr11_self_learning_v2.json").read_text(
            encoding="utf-8"
        )
    )
    assert saved == artifact

    repeat = run_experiment(
        ExperimentConfig(repo_root=tmp_path, results_dir=tmp_path / "results"),
        command_runner=runner,
        write=False,
    )
    assert repeat["reproducibility_checksum"] == artifact["reproducibility_checksum"]


def test_scenario_learn_2857_success_aggregates_measured_recurrence(
    tmp_path: Path,
) -> None:
    """SCENARIO-LEARN-2857: successful gated run reports measured recurrence effects."""

    _write_json(
        tmp_path / "results" / "experiment_2856_loopus_recurrence_backend_adapter.json",
        {
            "live_recurrence_backend_ready": True,
            "backend_module_path": "carnot.eval.fake_recurrence_backend",
            "model_specs": [
                {
                    "hf_id": "unsloth/gemma-4-26B-A4B-it-GGUF",
                    "headline_usable": True,
                }
            ],
        },
    )

    def runner(command: str, cwd: Path) -> CommandResult:
        assert cwd == tmp_path
        if command.startswith("jq -e"):
            return CommandResult(0, "true\n", "")
        return CommandResult(0, "carnot.eval.fake_recurrence_backend\n", "")

    measurement = {
        "source_counts": {"fover": 2, "humaneval": 1, "truthfulqa": 1},
        "per_example_trace": [
            {
                "energy_before": 1.0,
                "energy_after_each_loop": [0.6, 0.4],
                "correctness_before": False,
                "correctness_after": True,
                "early_exit_reason": "answer_passed",
            },
            {
                "energy_before": 0.2,
                "energy_after_each_loop": [0.25],
                "correctness_before": True,
                "correctness_after": True,
                "early_exit_reason": "energy_not_improved",
            },
            {
                "energy_before": 1.0,
                "energy_after_each_loop": [1.0, 0.9, 0.8],
                "correctness_before": False,
                "correctness_after": False,
                "early_exit_reason": "max_loops",
            },
            {
                "energy_before": 0.5,
                "energy_after_each_loop": [],
                "correctness_before": False,
                "correctness_after": False,
                "early_exit_reason": "no_recurrence_attempted",
            },
        ],
        "memory_hash_before": "before-sha",
        "memory_hash_after": "after-sha",
        "adversarial_verify_passed": True,
        "adversarial_verify_flags": [],
    }

    artifact = run_experiment(
        ExperimentConfig(
            repo_root=tmp_path,
            results_dir=tmp_path / "results",
            started_at=1.0,
            clock=lambda: 4.0,
        ),
        command_runner=runner,
        measurement_runner=lambda _config, _backend: measurement,
        write=False,
    )

    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["fr11_self_learning_ready"] is True
    assert artifact["n_examples"] == 4
    assert artifact["source_counts"] == {"fover": 2, "humaneval": 1, "truthfulqa": 1}
    assert artifact["recurrence_success_rate"] == pytest.approx(0.5)
    assert artifact["energy_delta_mean"] == pytest.approx(0.1875)
    assert artifact["correctness_delta"] == pytest.approx(0.25)
    assert artifact["per_loop_energy_summary"]["loop_1"]["n"] == 3
    assert artifact["per_loop_energy_summary"]["loop_1"]["mean_energy"] == pytest.approx(
        0.6166666667
    )
    assert artifact["per_loop_energy_summary"]["loop_2"]["n"] == 2
    assert artifact["per_loop_energy_summary"]["loop_3"]["mean_delta_from_initial"] == pytest.approx(
        0.2
    )
    assert artifact["early_exit_summary"] == {
        "answer_passed": 1,
        "energy_not_improved": 1,
        "max_loops": 1,
        "no_recurrence_attempted": 1,
    }
    assert artifact["memory_hash_before"] == "before-sha"
    assert artifact["memory_hash_after"] == "after-sha"
    assert artifact["model_specs"] == [
        {"hf_id": "unsloth/gemma-4-26B-A4B-it-GGUF", "headline_usable": True}
    ]
    assert artifact["adversarial_verify_passed"] is True
    assert artifact["adversarial_verify_flags"] == []
    assert artifact["duration_s"] == pytest.approx(3.0)


def test_req_learn_2857_existing_exp2856_blocker_classification(tmp_path: Path) -> None:
    """REQ-LEARN-2857-2: existing Exp 2856 failures keep precise blocked verdicts."""

    _write_json(
        tmp_path / "results" / "experiment_2856_loopus_recurrence_backend_adapter.json",
        {"live_recurrence_backend_ready": False},
    )

    def backend_not_ready(command: str, _cwd: Path) -> CommandResult:
        if command.startswith("jq -e"):
            return CommandResult(1, "false\n", "")
        return CommandResult(0, "carnot.eval.fake\n", "")

    artifact = run_experiment(
        ExperimentConfig(repo_root=tmp_path, results_dir=tmp_path / "results"),
        command_runner=backend_not_ready,
        write=False,
    )
    assert artifact["honest_verdict"] == "blocked_exp2856_backend_not_ready"

    def missing_backend_module(command: str, _cwd: Path) -> CommandResult:
        if command.startswith("jq -e"):
            return CommandResult(0, "true\n", "")
        return CommandResult(1, "", "KeyError: backend_module_path")

    artifact = run_experiment(
        ExperimentConfig(repo_root=tmp_path, results_dir=tmp_path / "results"),
        command_runner=missing_backend_module,
        write=False,
    )
    assert artifact["honest_verdict"] == "blocked_exp2856_backend_module_path"

    def repo_directory_failed(_command: str, _cwd: Path) -> CommandResult:
        return CommandResult(0, "ok\n", "")

    artifact = run_experiment(
        ExperimentConfig(
            repo_root=tmp_path / "missing-repo",
            results_dir=tmp_path / "results",
        ),
        command_runner=repo_directory_failed,
        write=False,
    )
    assert artifact["honest_verdict"] == "blocked_exp2856_precondition"
