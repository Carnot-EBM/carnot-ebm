"""Tests for Exp 4068 synchronous OFF-ARC resume-accumulate runner.

Spec refs: REQ-VERIFY-4068, SCENARIO-VERIFY-4068.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

import exp4068_offarc_transfer_power_sync as runner


def _check(resource: str, available: bool) -> dict[str, Any]:
    return {"resource": resource, "available": available}


def _task(task_id: str, *, hidden_expected: int = 3) -> runner.CodeTask:
    return runner.CodeTask(
        task_id=task_id,
        corpus="evalplus_fixture",
        prompt="Write add.",
        func_name="add",
        visible_tests=[
            runner.CodeTest("assert add(1, 1) == 2", "add", (1, 1), 2),
            runner.CodeTest("assert add(2, 2) == 4", "add", (2, 2), 4),
        ],
        hidden_tests=[
            runner.CodeTest(
                f"assert add(1, 2) == {hidden_expected}",
                "add",
                (1, 2),
                hidden_expected,
            )
        ],
    )


def _candidate(code: str, draw_index: int) -> runner.GeneratedCandidate:
    return runner.GeneratedCandidate(
        draw_index=draw_index,
        raw_text=code,
        code=code,
        generation_seconds=0.01,
        finish_reason="stop",
        truncated=False,
    )


def _executor(
    code: str, _func_name: str, args: tuple[Any, ...], _timeout: float
) -> tuple[Any, Exception | None]:
    if "return a + b" in code:
        return args[0] + args[1], None
    if "return 4" in code:
        return 4, None
    if "return 2" in code:
        return 2, None
    return 0, None


def _probe(
    *,
    corpus_key: str,
    evaluation_corpus: str,
    oracle_passrate: float,
    error: str | None = None,
) -> runner.OracleProbe:
    return runner.OracleProbe(
        corpus_key=corpus_key,
        evaluation_corpus=evaluation_corpus,
        oracle_passrate=oracle_passrate,
        n_tasks=8,
        error=error,
    )


def _route(oracle: float = 0.5) -> runner.CorpusRoute:
    return runner.route_corpus(
        evalplus_probe=lambda: _probe(
            corpus_key=runner.EVALPLUS_KEY,
            evaluation_corpus=runner.EVALPLUS_CORPUS,
            oracle_passrate=oracle,
        ),
        livecodebench_probe=lambda: pytest.fail("LiveCodeBench should not be probed"),
    )


def _saturated_evalplus_route() -> runner.CorpusRoute:
    return runner.CorpusRoute(
        corpus_key=runner.EVALPLUS_KEY,
        evaluation_corpus=runner.EVALPLUS_CORPUS,
        oracle_passrate=1.0,
        oracle_headroom_present=False,
        corpus_routed_reason="fixture saturated EvalPlus route",
        probes=[],
    )


def _per_task(rows: list[dict[str, bool]]) -> list[dict[str, Any]]:
    per_task = []
    for index, row in enumerate(rows):
        per_task.append(
            {
                "task_id": f"Task/{index}",
                "corpus": "fixture",
                "func_name": "f",
                "n_candidates": 5,
                "n_visible_tests": 2,
                "n_hidden_tests": 1,
                "armA_vote_pass1": row["a"],
                "armAplusplus_aces_pass1": row["app"],
                "armB_demofit_pass1": row["b"],
                "armC_symbolic_partition_pass1": row["c"],
                "oracle_hidden_pass": row["oracle"],
            }
        )
    return per_task


def test_req_4068_spec_declared() -> None:
    # REQ-VERIFY-4068: OpenSpec declares the synchronous no-background contract.
    spec = Path("openspec/capabilities/verification/spec.md").read_text(encoding="utf-8")
    for marker in (
        "REQ-VERIFY-4068",
        "SCENARIO-VERIFY-4068",
        "exp4068_offarc_transfer_power_sync.py",
        "single synchronous resume-accumulate runner",
        "SHALL NOT launch a background process",
        "corpus_routed_reason",
        "accumulated_n_tasks",
        "single_synchronous_resume_accumulate_no_background",
    ):
        assert marker in spec


def test_route_uses_evalplus_when_oracle_headroom_present() -> None:
    # REQ-VERIFY-4068: EvalPlus is chosen when the 12B oracle probe has headroom.
    route = _route(oracle=0.625)
    assert route.corpus_key == runner.EVALPLUS_KEY
    assert route.evaluation_corpus == runner.EVALPLUS_CORPUS
    assert route.oracle_passrate == pytest.approx(0.625)
    assert route.oracle_headroom_present is True
    assert "oracle headroom present" in route.corpus_routed_reason


def test_route_escalates_saturated_evalplus_to_livecodebench() -> None:
    # SCENARIO-VERIFY-4068: EvalPlus saturation routes upward instead of blocking.
    route = runner.route_corpus(
        evalplus_probe=lambda: _probe(
            corpus_key=runner.EVALPLUS_KEY,
            evaluation_corpus=runner.EVALPLUS_CORPUS,
            oracle_passrate=1.0,
        ),
        livecodebench_probe=lambda: _probe(
            corpus_key=runner.LIVECODEBENCH_KEY,
            evaluation_corpus=runner.LIVECODEBENCH_CORPUS,
            oracle_passrate=0.25,
        ),
    )
    assert route.corpus_key == runner.LIVECODEBENCH_KEY
    assert route.evaluation_corpus == runner.LIVECODEBENCH_CORPUS
    assert route.oracle_headroom_present is True
    assert ".375 fix" in route.corpus_routed_reason
    assert len(route.probes) == 2


def test_route_records_no_headroom_when_evalplus_saturated_and_livecodebench_errors() -> None:
    # REQ-VERIFY-4068: escalation errors become honest no-headroom, not launch failure.
    def livecodebench_error() -> runner.OracleProbe:
        raise RuntimeError("not cached")

    route = runner.route_corpus(
        evalplus_probe=lambda: _probe(
            corpus_key=runner.EVALPLUS_KEY,
            evaluation_corpus=runner.EVALPLUS_CORPUS,
            oracle_passrate=1.0,
        ),
        livecodebench_probe=livecodebench_error,
    )
    assert route.corpus_key == runner.EVALPLUS_KEY
    assert route.oracle_headroom_present is False
    assert "no probed corpus exposed usable headroom" in route.corpus_routed_reason
    assert route.probes[1]["error"].startswith("RuntimeError:")


def test_build_terminal_artifact_answers_demofit_headline_gate(tmp_path: Path) -> None:
    # REQ-VERIFY-4068: the terminal schema exposes the headline demo-fit CI gate.
    artifact = runner.build_terminal_artifact(
        per_task=_per_task(
            [
                {"a": False, "app": False, "b": True, "c": False, "oracle": True},
                {"a": False, "app": False, "b": True, "c": False, "oracle": True},
                {"a": False, "app": False, "b": True, "c": False, "oracle": False},
            ]
        ),
        route=_route(oracle=2 / 3),
        preconditions_checked=[_check("all", True)],
        model_specs={"local_generator": "fixture"},
        checkpoint_path=tmp_path / "checkpoint.json",
        source_candidate_checkpoint=tmp_path / "exp4045.json",
        started_s=0.0,
        ended_s=1.0,
        n_bootstrap=300,
        powered_task_floor=3,
    )
    runner.validate_terminal_artifact(artifact)
    assert artifact["honest_verdict"] == (
        "complete: offarc_demofit_transfers_to_code_ci_excl0_evalplus_n3"
    )
    assert artifact["evaluation_corpus"] == runner.EVALPLUS_CORPUS
    assert artifact["accumulated_n_tasks"] == 3
    assert artifact["oracle_headroom_present"] is True
    assert artifact["armB_demofit_passrate"] == pytest.approx(1.0)
    assert artifact["demofit_delta_pp"] == pytest.approx(100.0)
    assert artifact["demofit_bootstrap_ci95"][0] > 0.0
    assert artifact["demofit_ci_excludes_zero"] is True
    assert artifact["mechanism"] == runner.MECHANISM
    assert artifact["inference_substrate"] == runner.INFERENCE_SUBSTRATE


def test_build_terminal_artifact_reports_stronger_arm_context(tmp_path: Path) -> None:
    # SCENARIO-VERIFY-4068: ACES/SEP contextualize demo-fit when it touches zero.
    artifact = runner.build_terminal_artifact(
        per_task=_per_task(
            [
                {"a": False, "app": False, "b": False, "c": True, "oracle": True},
                {"a": False, "app": False, "b": False, "c": True, "oracle": True},
                {"a": False, "app": False, "b": False, "c": True, "oracle": False},
            ]
        ),
        route=_route(oracle=2 / 3),
        preconditions_checked=[_check("all", True)],
        model_specs={"local_generator": "fixture"},
        checkpoint_path=tmp_path / "checkpoint.json",
        source_candidate_checkpoint=tmp_path / "exp4045.json",
        started_s=0.0,
        ended_s=1.0,
        n_bootstrap=300,
        powered_task_floor=3,
    )
    assert artifact["best_arm"] == "armC_symbolic"
    assert artifact["best_arm_delta_pp"] == pytest.approx(100.0)
    assert artifact["best_arm_ci_excludes_zero"] is True
    assert artifact["demofit_ci_excludes_zero"] is False
    assert artifact["honest_verdict"] == (
        "complete: offarc_demofit_touches0_symbolic_excl0_evalplus_n3"
    )
    assert "GAP-CODE-EXEC-DEMOFIT" in artifact["missing_verifier_gaps"]


def test_artifact_validation_and_blocker_edge_cases(tmp_path: Path) -> None:
    # REQ-VERIFY-4068: malformed terminal artifacts fail closed with bare-type checks.
    result, error = runner.fast_restricted_executor(
        "def add(a, b):\n    return a + b\n", "add", (1, 2), 1.0
    )
    assert (result, error) == (3, None)
    result, error = runner.fast_restricted_executor("x =", "add", (1, 2), 1.0)
    assert result is None
    assert isinstance(error, SyntaxError)
    result, error = runner.fast_restricted_executor("def other():\n    return 1\n", "add", (), 1.0)
    assert result is None
    assert isinstance(error, NameError)
    result, error = runner.fast_restricted_executor(
        "def loop():\n    while True:\n        pass\n", "loop", (), 0.01
    )
    assert result is None
    assert isinstance(error, TimeoutError)

    artifact = runner.build_terminal_artifact(
        per_task=_per_task([{"a": False, "app": False, "b": True, "c": False, "oracle": False}]),
        route=_route(oracle=0.0),
        preconditions_checked=[_check("all", True)],
        model_specs={"local_generator": "fixture"},
        checkpoint_path=tmp_path / "checkpoint.json",
        source_candidate_checkpoint=tmp_path / "exp4045.json",
        started_s=0.0,
        ended_s=1.0,
        n_bootstrap=50,
        powered_task_floor=1,
    )
    runner.validate_terminal_artifact(artifact)
    assert runner.bootstrap_delta_ci95([], seed=1) == [0.0, 0.0]
    assert (
        runner.blocker_from_preconditions(
            [
                _check("local_gguf_cached", True),
                _check("llama_cpp_importable", True),
                _check("evalplus_loadable", False),
                _check("livecodebench_v6_loadable", False),
                _check("restricted_exec_importable", True),
            ]
        )
        == "blocked_no_code_corpus"
    )
    blocked = runner.build_blocked_artifact(
        honest_verdict="blocked_test_resource",
        preconditions_checked=[_check("test_resource", False)],
        output_path=tmp_path / "blocked.json",
        started_s=0.0,
        error="fixture",
    )
    assert blocked["error"] == "fixture"
    runner.validate_terminal_artifact(blocked)

    missing = dict(artifact)
    missing.pop("evaluation_corpus")
    with pytest.raises(ValueError, match="missing required field"):
        runner.validate_terminal_artifact(missing)

    for field, value, message in (
        ("honest_verdict", "success: wrong", "terminal prefix"),
        ("accumulated_n_tasks", 1.5, "bare int"),
        ("oracle_passrate", "0.0", "bare float"),
        ("oracle_headroom_present", 1, "bare bool"),
        ("demofit_bootstrap_ci95", [0.0], "two-element numeric list"),
        ("missing_verifier_gaps", "gap", "must be a list"),
        ("model_specs", [], "must be an object"),
        ("preconditions_checked", {}, "must be a list"),
        ("mechanism", "background", "synchronous no-background"),
        ("inference_substrate", "cached", "live_llm_inference"),
        ("reproducibility_checksum", "", "non-empty"),
    ):
        poisoned = dict(artifact)
        poisoned[field] = value
        with pytest.raises(ValueError, match=message):
            runner.validate_terminal_artifact(poisoned)


def test_verdict_variants_and_checkpoint_loader(tmp_path: Path) -> None:
    # SCENARIO-VERIFY-4068: terminal verdicts distinguish saturated and negative outcomes.
    saturated = runner.build_terminal_artifact(
        per_task=_per_task([{"a": True, "app": True, "b": True, "c": True, "oracle": True}]),
        route=_saturated_evalplus_route(),
        preconditions_checked=[],
        model_specs={},
        checkpoint_path=tmp_path / "checkpoint.json",
        source_candidate_checkpoint=tmp_path / "exp4045.json",
        started_s=0.0,
        ended_s=1.0,
        n_bootstrap=40,
        powered_task_floor=1,
    )
    assert saturated["honest_verdict"] == "complete: offarc_transfer_no_oracle_headroom_evalplus_n1"

    negative = runner.build_terminal_artifact(
        per_task=_per_task(
            [
                {"a": True, "app": False, "b": False, "c": False, "oracle": False},
                {"a": True, "app": False, "b": False, "c": False, "oracle": False},
            ]
        ),
        route=_route(oracle=0.0),
        preconditions_checked=[],
        model_specs={},
        checkpoint_path=tmp_path / "checkpoint.json",
        source_candidate_checkpoint=tmp_path / "exp4045.json",
        started_s=0.0,
        ended_s=1.0,
        n_bootstrap=40,
        powered_task_floor=2,
    )
    assert negative["honest_verdict"] == "complete: offarc_demofit_negative_ci_excl0_evalplus_n2"
    assert "UNSELECTABLE" not in " ".join(negative["missing_verifier_gaps"])

    unselectable = runner.build_terminal_artifact(
        per_task=_per_task([{"a": False, "app": False, "b": False, "c": False, "oracle": True}]),
        route=_route(oracle=0.0),
        preconditions_checked=[],
        model_specs={},
        checkpoint_path=tmp_path / "checkpoint.json",
        source_candidate_checkpoint=tmp_path / "exp4045.json",
        started_s=0.0,
        ended_s=1.0,
        n_bootstrap=40,
        powered_task_floor=1,
    )
    assert "UNSELECTABLE:Task/0" in unselectable["missing_verifier_gaps"]

    bad_checkpoint = tmp_path / "bad.checkpoint.json"
    bad_checkpoint.write_text("{", encoding="utf-8")
    assert runner._load_checkpoint(bad_checkpoint) == {}
    checkpoint = tmp_path / "ok.checkpoint.json"
    checkpoint.write_text(
        json.dumps(
            {
                "evaluations_by_task": {
                    "HumanEval/0": [
                        runner.CandidateEvaluation(
                            task_id="HumanEval/0",
                            draw_index=0,
                            status="ok",
                            code="def add(a, b):\n    return a + b\n",
                            visible_passes=[True],
                            hidden_passes=[True],
                            visible_outputs=[2],
                            hidden_outputs=[3],
                            fingerprint_outputs=[2],
                            generation_seconds=0.01,
                            truncated=False,
                            error=None,
                        ).__dict__
                    ]
                }
            }
        ),
        encoding="utf-8",
    )
    assert runner._load_checkpoint(checkpoint)["HumanEval/0"][0].draw_index == 0


def test_run_blocks_before_inference_when_mandatory_resource_missing(tmp_path: Path) -> None:
    # REQ-VERIFY-4068: missing mandatory resources write blocked_<resource> and stop.
    artifact = runner.run(
        output_path=tmp_path / "artifact.json",
        checkpoint_dir=tmp_path,
        legacy_checkpoint_path=tmp_path / "missing-exp4045.json",
        precondition_checker=lambda: [
            _check("local_gguf_cached", False),
            _check("llama_cpp_importable", True),
            _check("evalplus_loadable", True),
            _check("livecodebench_v6_loadable", False),
            _check("restricted_exec_importable", True),
        ],
        evalplus_task_loader=lambda limit: pytest.fail("loader should not run"),
        livecodebench_task_loader=lambda limit: pytest.fail("loader should not run"),
        sampler=lambda task, draw_index: pytest.fail("sampler should not run"),
        executor=_executor,
    )
    runner.validate_terminal_artifact(artifact)
    assert artifact["honest_verdict"] == "blocked_local_gguf_not_cached"
    assert artifact["accumulated_n_tasks"] == 0
    assert artifact["preconditions_checked"][0]["available"] is False


def test_run_routes_to_livecodebench_when_evalplus_probe_is_saturated(tmp_path: Path) -> None:
    # SCENARIO-VERIFY-4068: run() exercises the LiveCodeBench escalation path synchronously.
    legacy_checkpoint = tmp_path / "experiment_4045.checkpoint.json"
    correct = "def add(a, b):\n    return a + b\n"
    wrong = "def add(a, b):\n    return 2\n"
    legacy_checkpoint.write_text(
        json.dumps(
            {
                "evaluations_by_task": {
                    "HumanEval/0": [
                        {
                            "task_id": "HumanEval/0",
                            "draw_index": 0,
                            "status": "ok",
                            "code": correct,
                            "visible_passes": [True],
                            "hidden_passes": [True],
                            "visible_outputs": [2],
                            "hidden_outputs": [3],
                            "fingerprint_outputs": [2],
                            "generation_seconds": 0.01,
                            "truncated": False,
                            "error": None,
                        }
                    ]
                }
            }
        ),
        encoding="utf-8",
    )
    progress: list[str] = []
    artifact = runner.run(
        output_path=tmp_path / "artifact.json",
        checkpoint_dir=tmp_path,
        legacy_checkpoint_path=legacy_checkpoint,
        n_tasks=1,
        k=1,
        self_budget_s=0.0,
        probe_task_count=1,
        n_bootstrap=20,
        powered_task_floor=1,
        precondition_checker=lambda: [
            _check("local_gguf_cached", True),
            _check("llama_cpp_importable", True),
            _check("evalplus_loadable", True),
            _check("livecodebench_v6_loadable", True),
            _check("restricted_exec_importable", True),
        ],
        evalplus_task_loader=lambda limit: ([_task("HumanEval/0")][:limit], []),
        livecodebench_task_loader=lambda limit: ([_task("LCB/0")][:limit], []),
        sampler=lambda task, draw_index: _candidate(wrong, draw_index),
        executor=_executor,
        progress_printer=progress.append,
    )
    assert artifact["evaluation_corpus"] == runner.LIVECODEBENCH_CORPUS
    assert artifact["corpus"] == runner.LIVECODEBENCH_KEY
    assert artifact["stopped_reason"] == "self_budget_hit_after_task_1"
    assert any(line.startswith("[offarc] task ") for line in progress)


def test_run_lazily_loads_sampler_only_when_pool_needs_extension(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # REQ-VERIFY-4068: live sampler loading is deferred until a task lacks k candidates.
    wrong = "def add(a, b):\n    return 2\n"
    loaded: list[bool] = []

    def fake_make_sampler() -> Any:
        loaded.append(True)
        return lambda task, draw_index: _candidate(wrong, draw_index)

    monkeypatch.setattr(runner, "make_live_sampler", fake_make_sampler)
    artifact = runner.run(
        output_path=tmp_path / "artifact.json",
        checkpoint_dir=tmp_path,
        legacy_checkpoint_path=tmp_path / "missing-exp4045.json",
        n_tasks=1,
        k=1,
        self_budget_s=60.0,
        probe_task_count=1,
        n_bootstrap=20,
        powered_task_floor=2,
        precondition_checker=lambda: [
            _check("local_gguf_cached", True),
            _check("llama_cpp_importable", True),
            _check("evalplus_loadable", True),
            _check("livecodebench_v6_loadable", False),
            _check("restricted_exec_importable", True),
        ],
        evalplus_task_loader=lambda limit: ([_task("HumanEval/0")][:limit], []),
        livecodebench_task_loader=lambda limit: pytest.fail("LiveCodeBench should not load"),
        executor=_executor,
        progress_printer=lambda _line: None,
    )
    assert loaded == [True]
    assert artifact["accumulated_n_tasks"] == 1


def test_run_rescores_exp4045_pool_extends_tasks_and_checkpoints(tmp_path: Path) -> None:
    # SCENARIO-VERIFY-4068: synchronous run resumes, checkpoints, and reports N>0.
    legacy_checkpoint = tmp_path / "experiment_4045.checkpoint.json"
    output_path = tmp_path / "experiment_4068.json"
    correct = "def add(a, b):\n    return a + b\n"
    wrong = "def add(a, b):\n    return 2\n"
    legacy_checkpoint.write_text(
        json.dumps(
            {
                "completed_task_ids": ["HumanEval/0"],
                "ordered_task_ids": ["HumanEval/0"],
                "k_candidates_per_task": 5,
                "evaluations_by_task": {
                    "HumanEval/0": [
                        {
                            "task_id": "HumanEval/0",
                            "draw_index": 0,
                            "status": "ok",
                            "code": correct,
                            "visible_passes": [True],
                            "hidden_passes": [True],
                            "visible_outputs": [2],
                            "hidden_outputs": [3],
                            "fingerprint_outputs": [2],
                            "generation_seconds": 0.01,
                            "truncated": False,
                            "error": None,
                        }
                    ]
                },
            }
        ),
        encoding="utf-8",
    )
    progress: list[str] = []
    generated: list[tuple[str, int]] = []

    def sampler(task: runner.CodeTask, draw_index: int) -> runner.GeneratedCandidate:
        generated.append((task.task_id, draw_index))
        return _candidate(wrong, draw_index)

    artifact = runner.run(
        output_path=output_path,
        checkpoint_dir=tmp_path,
        legacy_checkpoint_path=legacy_checkpoint,
        n_tasks=2,
        k=2,
        self_budget_s=60.0,
        probe_task_count=2,
        n_bootstrap=100,
        powered_task_floor=2,
        precondition_checker=lambda: [
            _check("local_gguf_cached", True),
            _check("llama_cpp_importable", True),
            _check("evalplus_loadable", True),
            _check("livecodebench_v6_loadable", False),
            _check("restricted_exec_importable", True),
        ],
        evalplus_task_loader=lambda limit: ([_task("HumanEval/0"), _task("HumanEval/1")][:limit], []),
        livecodebench_task_loader=lambda limit: pytest.fail("LiveCodeBench should not load"),
        sampler=sampler,
        executor=_executor,
        progress_printer=progress.append,
    )
    runner.validate_terminal_artifact(artifact)
    assert output_path.exists()
    assert artifact["accumulated_n_tasks"] == 2
    assert artifact["evaluation_corpus"] == runner.EVALPLUS_CORPUS
    assert artifact["oracle_passrate"] < 0.95
    assert artifact["mechanism"] == runner.MECHANISM
    task_progress = [line for line in progress if line.startswith("[offarc] task ")]
    assert task_progress
    assert generated

    checkpoint_path = Path(artifact["stable_checkpoint_path"])
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    assert checkpoint["completed_task_ids"] == ["HumanEval/0", "HumanEval/1"]
    assert checkpoint["source_candidate_checkpoint"] == str(legacy_checkpoint)
    assert checkpoint["evaluations_by_task"]["HumanEval/0"][0]["code"] == correct
    assert checkpoint["evaluations_by_task"]["HumanEval/0"][0]["hidden_passes"] == [True]
