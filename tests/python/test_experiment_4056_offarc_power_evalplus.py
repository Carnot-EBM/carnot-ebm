"""Tests for Exp 4056 EvalPlus OFF-ARC build and resume runner.

Spec refs: REQ-VERIFY-4056, SCENARIO-VERIFY-4056.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import evalplus.data

import exp4056_offarc_power_evalplus_build as build
import offarc_power_evalplus_run as runner


def _task(task_id: str, *, hidden_expected: int = 3) -> runner.CodeTask:
    return runner.CodeTask(
        task_id=task_id,
        corpus="evalplus_humaneval",
        prompt="Write add.",
        func_name="add",
        visible_tests=[
            runner.CodeTest("assert add(1, 1) == 2", "add", (1, 1), 2),
            runner.CodeTest("assert add(2, 2) == 4", "add", (2, 2), 4),
        ],
        hidden_tests=[runner.CodeTest("assert add(1, 2) == 3", "add", (1, 2), hidden_expected)],
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
    if "return 2" in code:
        return 2, None
    return 0, None


def _check(resource: str, available: bool) -> dict[str, Any]:
    return {"resource": resource, "available": available}


def test_req_4056_spec_declared() -> None:
    # REQ-VERIFY-4056: OpenSpec declares the EvalPlus build/resume contract.
    spec = Path("openspec/capabilities/verification/spec.md").read_text(encoding="utf-8")
    for marker in (
        "REQ-VERIFY-4056",
        "SCENARIO-VERIFY-4056",
        "EvalPlus HumanEval+/MBPP+ hidden tests",
        "offarc_power_evalplus_run.py",
        "exp4056_offarc_power_evalplus_build.py",
        "stable corpus/model/k-keyed checkpoint",
    ):
        assert marker in spec


def test_runner_reuses_exp4045_candidate_code_and_writes_stable_checkpoint(tmp_path: Path) -> None:
    # REQ-VERIFY-4056: candidate pools from Exp 4045 are re-scored against EvalPlus hidden tests.
    legacy_checkpoint = tmp_path / "experiment_4045.checkpoint.json"
    stable_checkpoint = tmp_path / "offarc_power_evalplus_gemma12b_k2.checkpoint.json"
    output_path = tmp_path / "raw.json"
    code = "def add(a, b):\n    return a + b\n"
    legacy_checkpoint.write_text(
        json.dumps(
            {
                "completed_task_ids": ["HumanEval/0"],
                "ordered_task_ids": ["HumanEval/0"],
                "k_candidates_per_task": 2,
                "evaluations_by_task": {
                    "HumanEval/0": [
                        {
                            "task_id": "HumanEval/0",
                            "draw_index": 0,
                            "status": "ok",
                            "code": code,
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

    calls: list[int] = []

    def sampler(_task: runner.CodeTask, draw_index: int) -> runner.GeneratedCandidate:
        calls.append(draw_index)
        return _candidate("def add(a, b):\n    return 2\n", draw_index)

    artifact = runner.run(
        output_path=output_path,
        checkpoint_path=stable_checkpoint,
        legacy_checkpoint_path=legacy_checkpoint,
        n_tasks=1,
        k=2,
        mode="smoke",
        preconditions_checked=[_check("all", True)],
        task_loader=lambda limit: ([_task("HumanEval/0")][:limit], []),
        sampler=sampler,
        executor=_executor,
    )

    runner.validate_raw_artifact(artifact, require_full=False)
    checkpoint = json.loads(stable_checkpoint.read_text(encoding="utf-8"))
    rows = checkpoint["evaluations_by_task"]["HumanEval/0"]
    assert artifact["evaluation_corpus"] == runner.EVALUATION_CORPUS
    assert artifact["accumulated_n"] == 1
    assert artifact["resumed_from_n"] == 1
    assert calls == [1]
    assert rows[0]["code"] == code
    assert rows[0]["hidden_passes"] == [True]
    assert rows[1]["hidden_passes"] == [False]


def test_build_artifact_requires_bare_fields_and_headroom_flag(tmp_path: Path) -> None:
    # SCENARIO-VERIFY-4056: build artifacts expose bare fields for collector gates.
    artifact = build.build_artifact(
        honest_verdict=build.SUCCESS_VERDICT,
        runner_ready=True,
        smoke_passed=True,
        smoke_oracle_headroom_present=False,
        preconditions_checked=[_check("all", True)],
        resumed_from_n=22,
        launched_pid=1234,
        duration_s=1.2,
        stable_checkpoint_path=tmp_path / "stable.json",
        smoke_output_path=tmp_path / "smoke.json",
        raw_output_path=tmp_path / "raw.json",
        log_path=tmp_path / "run.log",
    )

    build.validate_build_artifact(artifact)
    assert artifact["runner_ready"] is True
    assert artifact["smoke_passed"] is True
    assert artifact["smoke_oracle_headroom_present"] is False
    assert artifact["needs_harder_corpus_livecodebench"] is True
    assert artifact["evaluation_corpus"] == runner.EVALUATION_CORPUS
    assert artifact["resumed_from_n"] == 22
    assert artifact["launched_pid"] == 1234


def test_run_build_blocks_before_smoke_when_evalplus_missing(tmp_path: Path) -> None:
    # REQ-VERIFY-4056: missing EvalPlus resources write blocked_<resource> and do not launch.
    launched: list[bool] = []

    artifact = build.run_build(
        output_path=tmp_path / "build.json",
        smoke_output_path=tmp_path / "smoke.json",
        raw_output_path=tmp_path / "raw.json",
        log_path=tmp_path / "run.log",
        stable_checkpoint_path=tmp_path / "stable.json",
        precondition_checker=lambda: [
            _check("local_gguf_cached", True),
            _check("llama_cpp_importable", True),
            _check("evalplus_humaneval_plus_loadable", False),
            _check("evalplus_mbpp_plus_loadable", True),
            _check("restricted_exec_importable", True),
        ],
        smoke_runner=lambda **_: pytest.fail("smoke should not run"),
        launcher=lambda **_: launched.append(True) or 99,
    )

    assert artifact["honest_verdict"] == "blocked_evalplus_not_cached"
    assert artifact["runner_ready"] is False
    assert artifact["smoke_passed"] is False
    assert artifact["launched_pid"] == 0
    assert launched == []


def test_run_build_launches_even_when_smoke_has_no_headroom(tmp_path: Path) -> None:
    # SCENARIO-VERIFY-4056: smoke oracle saturation is recorded but does not block the run.
    smoke = runner.build_raw_artifact(
        tasks=[_task("HumanEval/0", hidden_expected=3)],
        evaluations_by_task={
            "HumanEval/0": [
                runner.evaluate_candidate(
                    _task("HumanEval/0"),
                    _candidate("def add(a, b):\n    return a + b\n", 0),
                    executor=_executor,
                )
            ]
        },
        preconditions_checked=[_check("all", True)],
        model_specs={"local_generator": "fixture"},
        k=1,
        started_s=0.0,
        ended_s=0.1,
        mode="smoke",
        accumulated_n=1,
        resumed_from_n=0,
        stable_checkpoint_path=tmp_path / "stable.json",
    )

    artifact = build.run_build(
        output_path=tmp_path / "build.json",
        smoke_output_path=tmp_path / "smoke.json",
        raw_output_path=tmp_path / "raw.json",
        log_path=tmp_path / "run.log",
        stable_checkpoint_path=tmp_path / "stable.json",
        precondition_checker=lambda: [_check("all", True)],
        smoke_runner=lambda **_: smoke,
        launcher=lambda **_: 4321,
        resumed_counter=lambda _path: 22,
    )

    assert artifact["honest_verdict"] == build.SUCCESS_VERDICT
    assert artifact["runner_ready"] is True
    assert artifact["smoke_passed"] is True
    assert artifact["smoke_oracle_headroom_present"] is False
    assert artifact["needs_harder_corpus_livecodebench"] is True
    assert artifact["launched_pid"] == 4321


def test_runner_validation_and_checkpoint_edge_cases(tmp_path: Path) -> None:
    # REQ-VERIFY-4056: checkpoint and schema helpers fail closed without fabricating state.
    assert runner.parse_assertion_test("assert add(1, 2) == 3") is not None
    assert runner.parse_humaneval_candidate_assert("assert candidate(1) == 1", "f") is not None
    assert runner.extract_code_block("```python\ndef add(a, b):\n    return a + b\n```", "add")
    assert runner.blocker_from_preconditions([_check("all", True)]) is None
    assert runner.blocker_from_preconditions([_check("mystery", False)]) == "blocked_resource"
    assert runner.count_checkpoint_completed(tmp_path / "missing.json") == 0
    bad = tmp_path / "bad.json"
    bad.write_text("{", encoding="utf-8")
    assert runner.count_checkpoint_completed(bad) == 0
    evals = tmp_path / "evals.json"
    evals.write_text(json.dumps({"evaluations_by_task": {"a": [], "b": []}}), encoding="utf-8")
    assert runner.count_checkpoint_completed(evals) == 2
    assert runner._task_aliases("Mbpp/2") == ["Mbpp/2", "mbpp-2"]
    assert runner._task_aliases("mbpp-2") == ["mbpp-2", "Mbpp/2"]
    assert runner._input_to_args((1, 2)) == (1, 2)
    assert runner._input_to_args(3) == (3,)
    assert runner._canonical_expected("def f(x):\n    raise ValueError('x')\n", "f", (1,)) == (
        False,
        None,
    )

    artifact = runner.build_raw_artifact(
        tasks=[_task("HumanEval/0")],
        evaluations_by_task={"HumanEval/0": []},
        preconditions_checked=[],
        model_specs={},
        k=0,
        started_s=0.0,
        ended_s=1.0,
        mode="full",
        accumulated_n=0,
        resumed_from_n=0,
        stable_checkpoint_path=tmp_path / "stable.json",
    )
    with pytest.raises(ValueError, match="full EvalPlus run"):
        runner.validate_raw_artifact(artifact, require_full=True)
    missing = dict(artifact)
    missing.pop("candidate_pool")
    with pytest.raises(ValueError, match="missing required raw field"):
        runner.validate_raw_artifact(missing, require_full=False)
    bad_verdict = dict(artifact)
    bad_verdict["honest_verdict"] = "not-terminal"
    with pytest.raises(ValueError, match="terminal prefix"):
        runner.validate_raw_artifact(bad_verdict, require_full=False)


def test_evalplus_task_normalization_and_loaders(monkeypatch: pytest.MonkeyPatch) -> None:
    # REQ-VERIFY-4056: EvalPlus base inputs become visible tests and plus inputs hidden tests.
    row = {
        "entry_point": "add",
        "prompt": "def add(a, b):\n",
        "canonical_solution": "    return a + b\n",
        "base_input": [[1, 1], [2, 2]],
        "plus_input": [[1, 2], [3, 4]],
    }
    task, reason = runner._task_from_evalplus_row(
        row, task_id="HumanEval/fixture", corpus="evalplus_humaneval"
    )
    assert reason is None
    assert task is not None
    assert task.visible_tests[0].expected == 2
    assert task.hidden_tests[1].expected == 7

    mbpp_row = dict(row)
    mbpp_row["canonical_solution"] = "def add(a, b):\n    return a + b\n"
    task, reason = runner._task_from_evalplus_row(
        mbpp_row, task_id="Mbpp/fixture", corpus="evalplus_mbpp"
    )
    assert reason is None
    assert task is not None
    assert task.corpus == "evalplus_mbpp"

    missing, reason = runner._task_from_evalplus_row({}, task_id="bad", corpus="evalplus_mbpp")
    assert missing is None
    assert reason == "missing_entry_point"
    no_tests, reason = runner._task_from_evalplus_row(
        {"entry_point": "f", "canonical_solution": "def f():\n    return 1\n"},
        task_id="bad2",
        corpus="evalplus_mbpp",
    )
    assert no_tests is None
    assert reason == "could_not_build_visible_or_hidden_tests"

    monkeypatch.setattr(evalplus.data, "get_human_eval_plus", lambda: {"HumanEval/0": row})
    monkeypatch.setattr(evalplus.data, "get_mbpp_plus", lambda: {"Mbpp/2": mbpp_row})
    human_tasks, human_skipped = runner.load_humaneval_plus_tasks(limit=1)
    mbpp_tasks, mbpp_skipped = runner.load_mbpp_plus_tasks(limit=1)
    assert len(human_tasks) == 1 and human_skipped == []
    assert len(mbpp_tasks) == 1 and mbpp_skipped == []
    assert runner.load_mbpp_plus_tasks(limit=0) == ([], [])

    monkeypatch.setattr(
        runner,
        "load_humaneval_plus_tasks",
        lambda limit: ([_task("HumanEval/0")], []),
    )
    monkeypatch.setattr(
        runner,
        "load_mbpp_plus_tasks",
        lambda limit: ([_task("Mbpp/2")], []),
    )
    tasks, skipped = runner.load_code_tasks(limit=2)
    assert [task.task_id for task in tasks] == ["HumanEval/0", "Mbpp/2"]
    assert skipped == []


def test_runner_resume_edge_paths(tmp_path: Path) -> None:
    # REQ-VERIFY-4056: runner handles existing stable checkpoints and malformed legacy inputs.
    task = _task("HumanEval/0")
    stable = tmp_path / "stable.json"
    legacy = tmp_path / "legacy.json"
    output = tmp_path / "raw.json"
    stable_eval = runner.evaluate_candidate(
        task,
        _candidate("def add(a, b):\n    return a + b\n", 0),
        executor=_executor,
    )
    runner._write_checkpoint(
        stable,
        tasks=[task],
        evaluations_by_task={task.task_id: [stable_eval]},
        skipped_tasks=[],
        k=1,
        mode="smoke",
        legacy_checkpoint_path=legacy,
    )
    legacy.write_text("{", encoding="utf-8")
    artifact = runner.run(
        output_path=output,
        checkpoint_path=stable,
        legacy_checkpoint_path=legacy,
        n_tasks=1,
        k=1,
        mode="smoke",
        preconditions_checked=[],
        task_loader=lambda limit: ([task][:limit], []),
        sampler=lambda *_: pytest.fail("stable checkpoint should skip sampler"),
        executor=_executor,
    )
    assert artifact["oracle_passrate"] == 1.0
    assert runner._load_checkpoint(tmp_path / "missing.json") == {}
    bad_checkpoint = tmp_path / "bad_checkpoint.json"
    bad_checkpoint.write_text("{", encoding="utf-8")
    assert runner._load_checkpoint(bad_checkpoint) == {}
    assert runner._load_legacy_candidates(tmp_path / "missing_legacy.json") == {}
    assert runner._load_legacy_candidates(legacy) == {}

    with pytest.raises(RuntimeError, match="no EvalPlus code tasks loaded"):
        runner.run(
            output_path=output,
            checkpoint_path=tmp_path / "empty.json",
            legacy_checkpoint_path=tmp_path / "none.json",
            n_tasks=1,
            k=1,
            mode="smoke",
            task_loader=lambda _limit: ([], []),
            sampler=lambda *_: _candidate("", 0),
            executor=_executor,
        )


def test_build_validation_and_failure_paths(tmp_path: Path) -> None:
    # SCENARIO-VERIFY-4056: smoke and launch failures produce terminal blocked artifacts.
    base_kwargs = dict(
        honest_verdict=build.SUCCESS_VERDICT,
        runner_ready=True,
        smoke_passed=True,
        smoke_oracle_headroom_present=True,
        preconditions_checked=[],
        resumed_from_n=0,
        launched_pid=1,
        duration_s=0.1,
        stable_checkpoint_path=tmp_path / "stable.json",
        smoke_output_path=tmp_path / "smoke.json",
        raw_output_path=tmp_path / "raw.json",
        log_path=tmp_path / "log.txt",
    )
    artifact = build.build_artifact(error="fixture", **base_kwargs)
    assert artifact["error"] == "fixture"
    for field, value, message in (
        ("runner_ready", "true", "bare bool"),
        ("needs_harder_corpus_livecodebench", "false", "bare bool"),
        ("resumed_from_n", True, "bare int"),
        ("evaluation_corpus", "base", "EvalPlus hidden tests"),
        ("arms_implemented", [], "four required arms"),
        ("preconditions_checked", {}, "preconditions_checked must be a list"),
        ("inference_substrate", "cached", "live_llm_inference"),
    ):
        bad = dict(artifact)
        bad[field] = value
        with pytest.raises(ValueError, match=message):
            build.validate_build_artifact(bad)
    missing = dict(artifact)
    missing.pop("runner_ready")
    with pytest.raises(ValueError, match="missing required build field"):
        build.validate_build_artifact(missing)
    bad = dict(artifact)
    bad["honest_verdict"] = "not-terminal"
    with pytest.raises(ValueError, match="terminal prefix"):
        build.validate_build_artifact(bad)
    bad = dict(artifact)
    bad["launched_pid"] = 0
    with pytest.raises(ValueError, match="positive"):
        build.validate_build_artifact(bad)

    smoke_fail = build.run_build(
        output_path=tmp_path / "build-smoke-fail.json",
        smoke_output_path=tmp_path / "smoke.json",
        raw_output_path=tmp_path / "raw.json",
        log_path=tmp_path / "log.txt",
        stable_checkpoint_path=tmp_path / "stable.json",
        precondition_checker=lambda: [_check("all", True)],
        smoke_runner=lambda **_: (_ for _ in ()).throw(RuntimeError("boom")),
        launcher=lambda **_: pytest.fail("launch should not run"),
        resumed_counter=lambda _path: 3,
    )
    assert smoke_fail["honest_verdict"] == "blocked_smoke_failed"
    assert smoke_fail["error"] == "RuntimeError: boom"

    smoke = runner.build_raw_artifact(
        tasks=[_task("HumanEval/0")],
        evaluations_by_task={"HumanEval/0": []},
        preconditions_checked=[],
        model_specs={},
        k=1,
        started_s=0.0,
        ended_s=0.1,
        mode="smoke",
        accumulated_n=1,
        resumed_from_n=0,
        stable_checkpoint_path=tmp_path / "stable.json",
    )
    launch_fail = build.run_build(
        output_path=tmp_path / "build-launch-fail.json",
        smoke_output_path=tmp_path / "smoke.json",
        raw_output_path=tmp_path / "raw.json",
        log_path=tmp_path / "log.txt",
        stable_checkpoint_path=tmp_path / "stable.json",
        precondition_checker=lambda: [_check("all", True)],
        smoke_runner=lambda **_: smoke,
        launcher=lambda **_: (_ for _ in ()).throw(OSError("no launch")),
        resumed_counter=lambda _path: 4,
    )
    assert launch_fail["honest_verdict"] == "blocked_launch_failed"
    assert launch_fail["error"] == "OSError: no launch"


def test_runner_remaining_defensive_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # REQ-VERIFY-4056: defensive runner paths stay bounded and explicit.
    artifact = runner.build_raw_artifact(
        tasks=[_task("HumanEval/0")],
        evaluations_by_task={"HumanEval/0": []},
        preconditions_checked=[],
        model_specs={},
        k=1,
        started_s=0.0,
        ended_s=0.1,
        mode="smoke",
        accumulated_n=1,
        resumed_from_n=0,
        stable_checkpoint_path=tmp_path / "stable.json",
    )
    for field, value, message in (
        ("n_tasks", True, "bare int"),
        ("armA_vote_passrate", "0", "bare float"),
        ("evaluation_corpus", "base", "EvalPlus hidden tests"),
        ("arms_implemented", [], "four required arms"),
        ("model_specs", [], "model_specs must be an object"),
        ("preconditions_checked", {}, "preconditions_checked must be a list"),
        ("per_task", {}, "per_task must be a list"),
        ("candidate_pool", [], "candidate_pool must be an object"),
        ("reproducibility_checksum", "", "non-empty"),
        ("inference_substrate", "cached", "live_llm_inference"),
    ):
        bad = dict(artifact)
        bad[field] = value
        with pytest.raises(ValueError, match=message):
            runner.validate_raw_artifact(bad, require_full=False)

    empty_payload = tmp_path / "empty_payload.json"
    empty_payload.write_text(json.dumps({}), encoding="utf-8")
    assert runner.count_checkpoint_completed(empty_payload) == 0

    bad_row = {"entry_point": "missing", "base_input": [[1]], "plus_input": [[2]]}
    good_human = {
        "entry_point": "add",
        "prompt": "def add(a, b):\n",
        "canonical_solution": "    return a + b\n",
        "base_input": [[1, 1], [2, 2]],
        "plus_input": [[1, 2]],
    }
    good_mbpp = dict(good_human)
    good_mbpp["canonical_solution"] = "def add(a, b):\n    return a + b\n"
    monkeypatch.setattr(
        evalplus.data,
        "get_human_eval_plus",
        lambda: {"bad": bad_row, "HumanEval/0": good_human},
    )
    monkeypatch.setattr(
        evalplus.data,
        "get_mbpp_plus",
        lambda: {"bad": bad_row, "Mbpp/2": good_mbpp},
    )
    human_tasks, human_skipped = runner.load_humaneval_plus_tasks(limit=1)
    mbpp_tasks, mbpp_skipped = runner.load_mbpp_plus_tasks(limit=1)
    assert len(human_tasks) == 1 and human_skipped[0]["task_id"] == "bad"
    assert len(mbpp_tasks) == 1 and mbpp_skipped[0]["task_id"] == "bad"

    assert runner._verdict({"oracle_passrate": 1.0}, mode="full").endswith("oracle_saturated")
    assert runner._verdict(
        {
            "oracle_passrate": 0.5,
            "armB_demofit_passrate": 0.8,
            "armA_vote_passrate": 0.1,
            "bootstrap_ci95": [1.0, 2.0],
            "armC_symbolic_partition_passrate": 0.1,
            "bootstrap_ci95_armC_vs_armA": [0.0, 0.0],
        },
        mode="full",
    ).endswith("demofit_beats_vote_ci_excl0")
    assert runner._verdict(
        {
            "oracle_passrate": 0.5,
            "armB_demofit_passrate": 0.1,
            "armA_vote_passrate": 0.1,
            "bootstrap_ci95": [0.0, 0.0],
            "armC_symbolic_partition_passrate": 0.8,
            "bootstrap_ci95_armC_vs_armA": [1.0, 2.0],
        },
        mode="full",
    ).endswith("sep_beats_vote_ci_excl0")

    current = [
        runner.evaluate_candidate(
            _task("HumanEval/0"),
            _candidate("def add(a, b):\n    return a + b\n", 0),
            executor=_executor,
        )
    ]
    legacy = {
        "HumanEval/0": [
            _candidate("def add(a, b):\n    return a + b\n", 0),
            _candidate("def add(a, b):\n    return a + b\n", 1),
        ]
    }
    assert runner._extend_from_legacy(
        task=_task("HumanEval/0"),
        current=list(current),
        legacy_by_task=legacy,
        k=1,
        executor=_executor,
    )
    assert (
        len(
            runner._extend_from_legacy(
                task=_task("HumanEval/0"),
                current=list(current),
                legacy_by_task=legacy,
                k=2,
                executor=_executor,
            )
        )
        == 2
    )
    assert runner._extend_from_sampler(
        task=_task("HumanEval/0"),
        current=list(current),
        sampler=lambda *_: pytest.fail("sampler should break before call"),
        executor=_executor,
        k=1,
    )
    assert runner._legacy_candidates_for_task("HumanEval/missing", legacy) == []

    legacy_with_empty = tmp_path / "legacy_empty.json"
    legacy_with_empty.write_text(
        json.dumps(
            {
                "evaluations_by_task": {
                    "HumanEval/0": [
                        {"draw_index": 0, "code": ""},
                        {"draw_index": 1, "code": "def add(a, b):\n    return a + b\n"},
                    ]
                }
            }
        ),
        encoding="utf-8",
    )
    assert len(runner._load_legacy_candidates(legacy_with_empty)["HumanEval/0"]) == 1
    assert (
        runner._tests_from_inputs(
            "def f(x):\n    raise ValueError('x')\n", func_name="f", inputs=[1], hidden=True
        )
        == []
    )
