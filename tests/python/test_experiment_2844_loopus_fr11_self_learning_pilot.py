"""Tests for Exp 2844 LoopUS-style FR-11 self-learning pilot.

Spec: REQ-LEARN-2844,
      SCENARIO-LEARN-2844,
      SCENARIO-LEARN-2844-BLOCKED.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

import carnot.eval.loopus_fr11_self_learning_pilot as mod
from carnot.eval.loopus_fr11_self_learning_pilot import (
    CandidateScore,
    ExperimentConfig,
    GenerationResult,
    PilotExample,
    PreconditionCheck,
    model_specs_from_exp2836,
    probe_preconditions,
    run_experiment,
    run_recurrence_pilot,
    select_mixed_examples,
)


def _write_json(path: Path, payload: dict[str, Any] | list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _write_exp2836(path: Path, model_path: Path, *, ready: bool = True) -> None:
    _write_json(
        path,
        {
            "sota_runtime_ready": ready,
            "selected_python": "/tmp/venv/bin/python",
            "smoke_load_results": [
                {
                    "hf_id": "unsloth/gemma-4-26B-A4B-it-GGUF",
                    "model_path": str(model_path),
                    "load_success": True,
                    "headline_usable": True,
                }
            ],
            "model_specs": {
                "primary": [
                    "unsloth/Qwen3.6-35B-A3B-GGUF",
                    "unsloth/gemma-4-31B-it-GGUF",
                    "unsloth/gemma-4-26B-A4B-it-GGUF",
                ],
                "legacy_cpu_smoke_only": ["Qwen3.5-0.8B", "gemma-4-E4B-it"],
            },
        },
    )


def _all_checks() -> list[PreconditionCheck]:
    return [
        PreconditionCheck("exp2836_artifact", True, "present"),
        PreconditionCheck("exp2836_sota_runtime_ready", True, "ready"),
        PreconditionCheck("exp2836_selected_python", True, "/tmp/venv/bin/python"),
        PreconditionCheck("mandated_sota_model_path", True, "/cache/model.gguf"),
        PreconditionCheck("fover_dataset", True, "25 rows"),
        PreconditionCheck("mbpp_dataset", True, "25 rows"),
        PreconditionCheck("carnot_energy_feedback", True, "verifier available"),
    ]


def test_req_learn_2844_recurrence_metrics_and_trace() -> None:
    """REQ-LEARN-2844-3/4: recurrence computes deltas, exits, and trace fields."""

    examples = [
        PilotExample("fover", "f0", "FoVer prompt", "correct", {}),
        PilotExample("mbpp", "m0", "MBPP prompt", "def f(): pass", {}),
    ]
    energies = {
        ("f0", 0): CandidateScore(1.0, False, ["arithmetic total is inconsistent"]),
        ("f0", 1): CandidateScore(0.2, True, []),
        ("m0", 0): CandidateScore(0.5, False, ["missing return statement"]),
        ("m0", 1): CandidateScore(0.49, False, ["missing return statement"]),
    }
    feedback_seen: list[str] = []

    def generate(example: PilotExample, loop_index: int, feedback: str) -> GenerationResult:
        feedback_seen.append(feedback)
        return GenerationResult(f"{example.example_id}:loop{loop_index}", loop_index + 2)

    def score(example: PilotExample, answer: str) -> CandidateScore:
        loop_index = int(answer.rsplit("loop", 1)[1])
        return energies[(example.example_id, loop_index)]

    result = run_recurrence_pilot(
        examples,
        generate=generate,
        score=score,
        max_loops=3,
        convergence_threshold=0.05,
    )

    assert result["n_examples"] == 2
    assert result["mean_energy_delta_loop0_to_final"] == pytest.approx(0.405)
    assert result["correctness_delta"] == pytest.approx(0.5)
    assert result["early_exit_rate"] == pytest.approx(1.0)
    assert result["total_token_cost"] == 10
    assert "arithmetic total is inconsistent" in feedback_seen[1]
    assert "missing return statement" in feedback_seen[3]
    assert result["per_example_trace"][0]["early_exit_reason"] == "answer_passed"
    assert result["per_example_trace"][1]["early_exit_reason"] == "energy_converged"
    assert result["per_example_trace"][0]["loops"][0]["localized_feedback"] == [
        "arithmetic total is inconsistent"
    ]

    pass_first = run_recurrence_pilot(
        [PilotExample("fover", "f1", "prompt", "correct", {})],
        generate=lambda example, loop_index, feedback: GenerationResult("ok", 1),
        score=lambda example, answer: CandidateScore(0.0, True, []),
    )
    assert pass_first["correctness_delta"] == pytest.approx(0.0)
    assert mod._feedback_text([]) == ""
    assert run_recurrence_pilot([], generate=generate, score=score)["n_examples"] == 0


def test_scenario_learn_2844_blocked_backend_writes_zero_sentinel_schema(
    tmp_path: Path,
) -> None:
    """SCENARIO-LEARN-2844-BLOCKED: no backend blocks without measured deltas."""

    model_path = tmp_path / "cache" / "model.gguf"
    model_path.parent.mkdir()
    model_path.write_bytes(b"gguf")
    _write_exp2836(tmp_path / "results" / "experiment_2836_sota_runtime_preflight.json", model_path)

    artifact = run_experiment(
        ExperimentConfig(
            repo_root=tmp_path,
            results_dir=tmp_path / "results",
            started_at=10.0,
            clock=lambda: 12.5,
        ),
        precondition_probe=lambda _config, _model_specs: _all_checks(),
        write=True,
    )

    assert artifact["honest_verdict"] == "blocked_live_recurrence_backend"
    assert artifact["continuous_self_learning_task"] is True
    assert artifact["n_examples"] == 0
    assert artifact["mean_energy_delta_loop0_to_final"] == 0.0
    assert artifact["correctness_delta"] == 0.0
    assert artifact["early_exit_rate"] == 0.0
    assert artifact["per_example_trace"] == []
    assert artifact["blocked_resources"] == ["live_recurrence_backend"]
    saved = json.loads(
        (tmp_path / "results" / "experiment_2844_loopus_fr11_self_learning_pilot.json").read_text(
            encoding="utf-8"
        )
    )
    assert saved == artifact

    blocked_precondition = run_experiment(
        ExperimentConfig(repo_root=tmp_path, results_dir=tmp_path / "results"),
        precondition_probe=lambda _config, _model_specs: [
            PreconditionCheck("mbpp_dataset", False, "offline")
        ],
        write=False,
    )
    assert blocked_precondition["honest_verdict"] == "blocked_mbpp_dataset"


def test_scenario_learn_2844_success_artifact_uses_injected_measurement(
    tmp_path: Path,
) -> None:
    """SCENARIO-LEARN-2844: successful measurement is summarized without rewriting deltas."""

    model_path = tmp_path / "cache" / "model.gguf"
    model_path.parent.mkdir()
    model_path.write_bytes(b"gguf")
    _write_exp2836(tmp_path / "results" / "experiment_2836_sota_runtime_preflight.json", model_path)
    pilot = {
        "n_examples": 50,
        "mean_energy_delta_loop0_to_final": 0.125,
        "correctness_delta": 0.04,
        "early_exit_rate": 0.6,
        "per_example_trace": [{"corpus": "fover", "example_id": "f0", "loops": []}],
        "total_token_cost": 123,
    }

    artifact = run_experiment(
        ExperimentConfig(
            repo_root=tmp_path,
            results_dir=tmp_path / "results",
            started_at=1.0,
            clock=lambda: 9.0,
        ),
        precondition_probe=lambda _config, _model_specs: _all_checks(),
        measurement_runner=lambda _config, _model_specs: pilot,
        write=False,
    )

    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["n_examples"] == 50
    assert artifact["mean_energy_delta_loop0_to_final"] == pytest.approx(0.125)
    assert artifact["correctness_delta"] == pytest.approx(0.04)
    assert artifact["early_exit_rate"] == pytest.approx(0.6)
    assert artifact["model_specs"]["selected_model_path"] == str(model_path)
    assert artifact["duration_s"] == pytest.approx(8.0)


def test_req_learn_2844_model_specs_and_deterministic_sample_selection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-LEARN-2844-1/2: model specs and deterministic 25+25 sample selection."""

    model_path = tmp_path / "model.gguf"
    model_path.write_bytes(b"gguf")
    exp2836 = {
        "sota_runtime_ready": True,
        "selected_python": "/venv/python",
        "smoke_load_results": [
            {
                "hf_id": "unsloth/gemma-4-26B-A4B-it-GGUF",
                "model_path": str(model_path),
                "load_success": True,
                "headline_usable": True,
            }
        ],
        "model_specs": {"primary": ["a"], "legacy_cpu_smoke_only": ["legacy"]},
    }
    specs = model_specs_from_exp2836(exp2836)
    assert specs["selected_python"] == "/venv/python"
    assert specs["selected_model_path"] == str(model_path)
    assert specs["selected_model_hf_id"] == "unsloth/gemma-4-26B-A4B-it-GGUF"
    assert specs["headline_required_any_of"] == ["a"]
    assert specs["legacy_cpu_smoke_only"] == ["legacy"]

    fover_rows = [
        {"question_id": f"f{i}", "step_text": f"FoVer row {i}", "label": "correct"}
        for i in range(30)
    ]
    _write_json(tmp_path / "data" / "fover_corpus_v4.json", fover_rows)
    mbpp_rows = [
        {
            "task_id": i,
            "prompt": f"write function {i}",
            "code": f"def f{i}(): return {i}",
            "test_list": [f"assert f{i}() == {i}"],
        }
        for i in range(30)
    ]
    monkeypatch.setattr(
        "carnot.eval.loopus_fr11_self_learning_pilot._load_mbpp_rows",
        lambda _limit: mbpp_rows,
    )

    first = select_mixed_examples(tmp_path, n_fover=25, n_mbpp=25, seed=20260522)
    second = select_mixed_examples(tmp_path, n_fover=25, n_mbpp=25, seed=20260522)

    assert first == second
    assert len(first) == 50
    assert sum(1 for item in first if item.corpus == "fover") == 25
    assert sum(1 for item in first if item.corpus == "mbpp") == 25
    assert all(item.prompt for item in first)


def test_req_learn_2844_default_preconditions_and_loader_edges(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-LEARN-2844-2: preconditions report concrete resources and loader edges."""

    assert mod.load_exp2836_preflight(tmp_path / "missing.json") == {}
    assert mod._load_fover_rows(tmp_path) == []
    with pytest.raises(ValueError, match="needed 2 rows"):
        mod._select_rows([{"x": 1}], 2, seed=1)

    jsonl = tmp_path / "data" / "fover_corpus.jsonl"
    jsonl.parent.mkdir()
    jsonl.write_text('{"question_id": "q0", "step_text": "row", "label": "correct"}\n')
    assert mod._load_fover_rows(tmp_path)[0]["question_id"] == "q0"

    model_path = tmp_path / "model.gguf"
    model_path.write_bytes(b"gguf")
    exp2836_path = tmp_path / "results" / "experiment_2836_sota_runtime_preflight.json"
    _write_exp2836(exp2836_path, model_path)
    monkeypatch.setattr(mod, "_load_mbpp_rows", lambda _limit: [{"task_id": i} for i in range(25)])

    checks = probe_preconditions(
        ExperimentConfig(repo_root=tmp_path, results_dir=tmp_path / "results", n_fover=1),
        model_specs_from_exp2836(mod.load_exp2836_preflight(exp2836_path)),
    )

    assert [check.resource for check in checks] == [
        "exp2836_artifact",
        "exp2836_sota_runtime_ready",
        "exp2836_selected_python",
        "mandated_sota_model_path",
        "fover_dataset",
        "mbpp_dataset",
        "carnot_energy_feedback",
    ]
    assert all(check.available for check in checks)

    monkeypatch.setattr(
        mod,
        "_load_fover_rows",
        lambda _root: (_ for _ in ()).throw(RuntimeError("fover failed")),
    )
    monkeypatch.setattr(
        mod,
        "_load_mbpp_rows",
        lambda _limit: (_ for _ in ()).throw(RuntimeError("mbpp failed")),
    )
    monkeypatch.setitem(sys.modules, "carnot.verify.sc_energy_verifier", None)
    failed = probe_preconditions(
        ExperimentConfig(repo_root=tmp_path, results_dir=tmp_path / "results"),
        model_specs_from_exp2836(mod.load_exp2836_preflight(exp2836_path)),
    )
    assert next(check for check in failed if check.resource == "fover_dataset").available is False
    assert next(check for check in failed if check.resource == "mbpp_dataset").available is False
    assert (
        next(check for check in failed if check.resource == "carnot_energy_feedback").available
        is False
    )
