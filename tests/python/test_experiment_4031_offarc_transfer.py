"""Tests for Exp 4031/4032 OFF-ARC execution-verifier transfer.

Spec refs: REQ-VERIFY-4031, SCENARIO-VERIFY-4031,
REQ-VERIFY-4032, SCENARIO-VERIFY-4032.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

import exp4031_offarc_transfer_build as build
import offarc_exec_verifier_transfer_run as runner


def _check(resource: str, available: bool) -> dict[str, Any]:
    return {"resource": resource, "available": available}


def _task(task_id: str = "MBPP:1") -> runner.CodeTask:
    return runner.CodeTask(
        task_id=task_id,
        corpus="mbpp",
        prompt="Write a function to add two numbers.",
        func_name="add",
        visible_tests=[
            runner.CodeTest("assert add(1, 2) == 3", "add", (1, 2), 3),
            runner.CodeTest("assert add(4, 5) == 9", "add", (4, 5), 9),
        ],
        hidden_tests=[runner.CodeTest("assert add(2, 7) == 9", "add", (2, 7), 9)],
    )


def _eval(
    task_id: str,
    draw_index: int,
    *,
    visible_passes: list[bool],
    hidden_passes: list[bool],
    visible_outputs: list[Any],
    truncated: bool = False,
) -> runner.CandidateEvaluation:
    return runner.CandidateEvaluation(
        task_id=task_id,
        draw_index=draw_index,
        status="ok",
        code=f"def add(a, b):\n    return a + b  # {task_id}-{draw_index}\n",
        visible_passes=visible_passes,
        hidden_passes=hidden_passes,
        visible_outputs=visible_outputs,
        hidden_outputs=["hidden"],
        generation_seconds=0.1,
        truncated=truncated,
        error=None,
    )


def _synthetic_evaluations() -> tuple[
    list[runner.CodeTask], dict[str, list[runner.CandidateEvaluation]]
]:
    tasks = [_task("MBPP:1"), _task("MBPP:2")]
    evaluations = {
        "MBPP:1": [
            _eval(
                "MBPP:1",
                0,
                visible_passes=[False, False],
                hidden_passes=[False],
                visible_outputs=[0, 0],
            ),
            _eval(
                "MBPP:1",
                1,
                visible_passes=[True, True],
                hidden_passes=[True],
                visible_outputs=[3, 9],
            ),
            _eval(
                "MBPP:1",
                2,
                visible_passes=[False, False],
                hidden_passes=[False],
                visible_outputs=[0, 0],
            ),
        ],
        "MBPP:2": [
            _eval(
                "MBPP:2",
                0,
                visible_passes=[True, True],
                hidden_passes=[True],
                visible_outputs=[3, 9],
            ),
            _eval(
                "MBPP:2",
                1,
                visible_passes=[False, True],
                hidden_passes=[False],
                visible_outputs=[2, 9],
            ),
            _eval(
                "MBPP:2",
                2,
                visible_passes=[False, False],
                hidden_passes=[False],
                visible_outputs=[0, 0],
                truncated=True,
            ),
        ],
    }
    return tasks, evaluations


def test_req_4031_4032_spec_declared() -> None:
    # REQ-VERIFY-4031 / REQ-VERIFY-4032: OpenSpec declares build and run contracts first.
    spec = Path("openspec/capabilities/verification/spec.md").read_text(encoding="utf-8")
    for marker in (
        "REQ-VERIFY-4031",
        "SCENARIO-VERIFY-4031",
        "REQ-VERIFY-4032",
        "SCENARIO-VERIFY-4032",
        "offarc_exec_verifier_transfer_run.py",
        "armAplusplus_aces_passrate",
    ):
        assert marker in spec


def test_parse_assertion_test_extracts_function_args_and_expected() -> None:
    # REQ-VERIFY-4032: visible and hidden tests are parsed as exact-output demos.
    parsed = runner.parse_assertion_test("assert add(2, 3) == 5")
    assert parsed == runner.CodeTest("assert add(2, 3) == 5", "add", (2, 3), 5)
    assert runner.parse_assertion_test("assert add(1, b=2) == 3") is None
    assert runner.parse_assertion_test("assert add(1) > 0") is None
    assert runner.parse_assertion_test("assert add(x) == 3") is None
    assert runner.parse_assertion_test("assert (") is None
    assert runner.parse_assertion_test("print('not an assert')") is None


def test_extract_code_block_prefers_python_fence_and_function_name() -> None:
    # SCENARIO-VERIFY-4032: generated text is normalized to the candidate program body.
    text = "notes\n```python\ndef add(a, b):\n    return a + b\n```\ntrailing"
    assert runner.extract_code_block(text, "add") == "def add(a, b):\n    return a + b"
    unfenced = "Here is code:\ndef add(a, b):\n    return a - b\n"
    assert runner.extract_code_block(unfenced, "add").startswith("def add")
    assert runner.extract_code_block("no function here", "add") == ""


def test_evaluate_candidate_uses_injected_executor_for_visible_and_hidden_tests() -> None:
    # REQ-VERIFY-4032: candidates are executed through the sandbox-compatible executor hook.
    calls: list[tuple[str, tuple[Any, ...]]] = []

    def executor(
        code: str, func_name: str, args: tuple[Any, ...], timeout: float
    ) -> tuple[Any, Exception | None]:
        del code, timeout
        calls.append((func_name, args))
        return sum(args), None

    candidate = runner.GeneratedCandidate(
        draw_index=0,
        raw_text="```python\ndef add(a, b):\n    return a + b\n```",
        code="def add(a, b):\n    return a + b\n",
        generation_seconds=0.1,
        finish_reason="stop",
        truncated=False,
    )
    evaluated = runner.evaluate_candidate(_task(), candidate, executor=executor)
    assert evaluated.visible_passes == [True, True]
    assert evaluated.hidden_passes == [True]
    assert calls == [("add", (1, 2)), ("add", (4, 5)), ("add", (2, 7))]


def test_evaluate_candidate_records_no_code_and_executor_errors() -> None:
    # REQ-VERIFY-4032: malformed candidates and execution errors remain in the shared pool as failures.
    no_code = runner.GeneratedCandidate(0, "text", "", 0.1, "stop", False)
    evaluated = runner.evaluate_candidate(_task(), no_code, executor=lambda *args: (None, None))
    assert evaluated.status == "no_code"
    assert evaluated.visible_passes == [False, False]

    def executor(*args: Any) -> tuple[Any, Exception | None]:
        del args
        return None, RuntimeError("boom")

    candidate = runner.GeneratedCandidate(
        1, "raw", "def add(a, b):\n    return a + b\n", 0.1, "stop", False
    )
    evaluated = runner.evaluate_candidate(_task(), candidate, executor=executor)
    assert evaluated.visible_passes == [False, False]
    assert evaluated.hidden_passes == [False]
    assert "RuntimeError" in (evaluated.error or "")


def test_aces_degenerate_inputs_are_bounded() -> None:
    # REQ-VERIFY-4032: ACES remains deterministic when visible tests are absent, singleton, or tied.
    assert runner._stable_repr({1, 2}) == repr({1, 2})
    assert runner.aces_leave_one_out_scores([]) == {}
    no_tests = [_eval("T", 0, visible_passes=[], hidden_passes=[False], visible_outputs=[])]
    assert runner.aces_leave_one_out_scores(no_tests) == {0: 0.0}
    one_test = [_eval("T", 1, visible_passes=[True], hidden_passes=[True], visible_outputs=[1])]
    assert runner.aces_leave_one_out_scores(one_test) == {1: 1.0}
    tied = [
        _eval("T", 0, visible_passes=[True, False], hidden_passes=[False], visible_outputs=[1, 0]),
        _eval("T", 1, visible_passes=[False, True], hidden_passes=[False], visible_outputs=[0, 1]),
    ]
    assert runner.aces_leave_one_out_scores(tied) == {0: 0.5, 1: 0.5}


def test_score_evaluated_tasks_measures_vote_aces_demofit_and_oracle() -> None:
    # SCENARIO-VERIFY-4032: all arms use one pool and hidden pass@1/pass@2 are measured after selection.
    tasks, evaluations = _synthetic_evaluations()
    scored = runner.score_evaluated_tasks(tasks, evaluations, seed=4032)
    assert scored["armA_vote_passrate"] == pytest.approx(0.5)
    assert scored["armB_demofit_passrate"] == pytest.approx(1.0)
    assert scored["armAplusplus_aces_passrate"] == pytest.approx(1.0)
    assert scored["delta_pp"] == pytest.approx(50.0)
    assert scored["oracle_passrate"] == pytest.approx(1.0)
    assert scored["truncation_rate"] == pytest.approx(1 / 6)
    assert scored["bootstrap_ci95"][0] >= 0.0
    assert scored["per_task"][0]["armB_demo_perfect_count"] == 1


def test_score_evaluated_tasks_records_missing_gap_and_no_demofit() -> None:
    task = _task("MBPP:gap")
    evaluations = {
        "MBPP:gap": [
            _eval(
                "MBPP:gap",
                0,
                visible_passes=[False, False],
                hidden_passes=[True],
                visible_outputs=[1, 1],
            ),
            _eval(
                "MBPP:gap",
                1,
                visible_passes=[False, True],
                hidden_passes=[False],
                visible_outputs=[2, 9],
            ),
        ]
    }
    scored = runner.score_evaluated_tasks([task], evaluations, seed=1)
    assert scored["missing_verifier_gaps"] == ["MBPP:gap"]
    assert scored["armB_demofit_passrate"] == 0.0


def test_full_verdict_branches_cover_success_no_headroom_and_no_transfer() -> None:
    assert runner._verdict(
        {
            "armB_demofit_passrate": 1.0,
            "armA_vote_passrate": 0.0,
            "bootstrap_ci95": [1.0, 100.0],
            "oracle_passrate": 1.0,
        },
        mode="full",
    ).startswith("success:")
    assert runner._verdict(
        {
            "armB_demofit_passrate": 0.5,
            "armA_vote_passrate": 0.5,
            "bootstrap_ci95": [0.0, 0.0],
            "oracle_passrate": 0.5,
        },
        mode="full",
    ).startswith("complete: exec_verifier_transfer_uninformative")
    assert runner._verdict(
        {
            "armB_demofit_passrate": 0.0,
            "armA_vote_passrate": 0.5,
            "bootstrap_ci95": [-50.0, 0.0],
            "oracle_passrate": 1.0,
        },
        mode="full",
    ).startswith("complete: exec_verifier_no_transfer")
    assert runner._bootstrap_ci_pp([], seed=1) == [0.0, 0.0]


def test_build_raw_artifact_validates_required_fields_and_checksum() -> None:
    # REQ-VERIFY-4032: raw artifact contains required metrics and reproducibility checksum.
    tasks, evaluations = _synthetic_evaluations()
    artifact = runner.build_raw_artifact(
        tasks=tasks,
        evaluations_by_task=evaluations,
        preconditions_checked=[
            _check("local_gguf_cached", True),
            _check("llama_cpp_importable", True),
            _check("code_corpus_loadable", True),
            _check("restricted_exec_importable", True),
        ],
        model_specs={"generator_model": "gemma-4-12B", "verifier": "sandboxed demo-fit"},
        k=3,
        started_s=10.0,
        ended_s=12.0,
        mode="smoke",
    )
    runner.validate_raw_artifact(artifact, require_full=False)
    assert artifact["honest_verdict"].startswith(("success:", "complete:"))
    assert artifact["reproducibility_checksum"]

    poisoned = dict(artifact)
    poisoned["runner_ready"] = True
    with pytest.raises(ValueError, match="unexpected raw artifact field"):
        runner.validate_raw_artifact(poisoned, require_full=False)


def test_raw_artifact_full_mode_requires_n_and_k_floor() -> None:
    tasks, evaluations = _synthetic_evaluations()
    artifact = runner.build_raw_artifact(
        tasks=tasks,
        evaluations_by_task=evaluations,
        preconditions_checked=[_check("all", True)],
        model_specs={"generator_model": "fixture"},
        k=3,
        started_s=0.0,
        ended_s=1.0,
        mode="full",
    )
    with pytest.raises(ValueError, match="full run must include"):
        runner.validate_raw_artifact(artifact, require_full=True)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("honest_verdict", "done", "terminal prefix"),
        ("n_tasks", 2.5, "n_tasks must be a bare int"),
        ("armA_vote_passrate", "0.0", "armA_vote_passrate must be a bare float"),
        ("bootstrap_ci95", [0.0], "two-element list"),
        ("bootstrap_ci95", [0.0, "bad"], "values must be numeric"),
        ("model_specs", [], "model_specs must be an object"),
        ("reproducibility_checksum", "", "non-empty string"),
        ("preconditions_checked", {}, "preconditions_checked must be a list"),
        ("missing_verifier_gaps", {}, "missing_verifier_gaps must be a list"),
        ("inference_substrate", "cached", "inference_substrate must be live_llm_inference"),
    ],
)
def test_validate_raw_artifact_rejects_schema_poison(field: str, value: Any, message: str) -> None:
    tasks, evaluations = _synthetic_evaluations()
    artifact = runner.build_raw_artifact(
        tasks=tasks,
        evaluations_by_task=evaluations,
        preconditions_checked=[_check("all", True)],
        model_specs={"generator_model": "fixture"},
        k=3,
        started_s=0.0,
        ended_s=1.0,
        mode="smoke",
    )
    artifact[field] = value
    with pytest.raises(ValueError, match=message):
        runner.validate_raw_artifact(artifact, require_full=False)


def test_validate_raw_artifact_rejects_missing_required_field() -> None:
    with pytest.raises(ValueError, match="missing required raw field"):
        runner.validate_raw_artifact({}, require_full=False)


def test_runner_write_json_creates_parent(tmp_path: Path) -> None:
    out = tmp_path / "nested" / "artifact.json"
    runner._write_json(out, {"ok": True})
    assert out.exists()


def test_precondition_blocker_uses_first_missing_resource() -> None:
    # REQ-VERIFY-4031: missing resources block before inference or launch.
    checks = [
        _check("local_gguf_cached", True),
        _check("llama_cpp_importable", False),
        _check("code_corpus_loadable", False),
    ]
    assert runner.blocker_from_preconditions(checks) == "blocked_llama_cpp_unavailable"
    assert runner.blocker_from_preconditions([_check("local_gguf_cached", True)]) is None


def test_build_artifact_blocked_path_does_not_smoke_or_launch(tmp_path: Path) -> None:
    # REQ-VERIFY-4031: blocked preconditions write the build artifact and stop.
    def boom(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("must not be called")

    artifact = build.run_build(
        output_path=tmp_path / "build.json",
        precondition_checker=lambda: [_check("local_gguf_cached", False)],
        smoke_runner=boom,
        launcher=boom,
    )
    assert artifact["honest_verdict"] == "blocked_local_gguf_not_cached"
    assert artifact["runner_ready"] is False
    assert artifact["smoke_passed"] is False
    assert artifact["launched_pid"] == 0
    assert (tmp_path / "build.json").exists()


def test_build_artifact_smoke_failure_blocks_launch(tmp_path: Path) -> None:
    # REQ-VERIFY-4031: a failed two-task smoke prevents launching the full run.
    def smoke_runner(**kwargs: Any) -> dict[str, Any]:
        del kwargs
        raise RuntimeError("smoke failed")

    def launcher(**kwargs: Any) -> int:
        del kwargs
        raise AssertionError("launch should not run")

    artifact = build.run_build(
        output_path=tmp_path / "build.json",
        precondition_checker=lambda: [_check("local_gguf_cached", True)],
        smoke_runner=smoke_runner,
        launcher=launcher,
    )
    assert artifact["honest_verdict"] == "blocked_smoke_failed"
    assert artifact["runner_ready"] is True
    assert artifact["smoke_passed"] is False
    assert "smoke failed" in artifact["error"]


def test_build_artifact_launch_failure_is_recorded(tmp_path: Path) -> None:
    # REQ-VERIFY-4031: launch failures are honest blocked artifacts after smoke passes.
    def smoke_runner(**kwargs: Any) -> dict[str, Any]:
        tasks, evaluations = _synthetic_evaluations()
        return runner.build_raw_artifact(
            tasks=tasks,
            evaluations_by_task=evaluations,
            preconditions_checked=kwargs["preconditions_checked"],
            model_specs={"generator_model": "fixture"},
            k=3,
            started_s=0.0,
            ended_s=1.0,
            mode="smoke",
        )

    def launcher(**kwargs: Any) -> int:
        del kwargs
        raise OSError("no nohup")

    artifact = build.run_build(
        output_path=tmp_path / "build.json",
        precondition_checker=lambda: [_check("local_gguf_cached", True)],
        smoke_runner=smoke_runner,
        launcher=launcher,
    )
    assert artifact["honest_verdict"] == "blocked_launch_failed"
    assert artifact["smoke_passed"] is True
    assert "no nohup" in artifact["error"]


def test_build_artifact_success_path_smokes_then_launches(tmp_path: Path) -> None:
    # SCENARIO-VERIFY-4031: a valid smoke artifact gates the background launch.
    launched: list[dict[str, Any]] = []

    def smoke_runner(**kwargs: Any) -> dict[str, Any]:
        tasks, evaluations = _synthetic_evaluations()
        artifact = runner.build_raw_artifact(
            tasks=tasks,
            evaluations_by_task=evaluations,
            preconditions_checked=kwargs["preconditions_checked"],
            model_specs={"generator_model": "fixture"},
            k=3,
            started_s=0.0,
            ended_s=1.0,
            mode="smoke",
        )
        kwargs["output_path"].write_text("{}", encoding="utf-8")
        return artifact

    def launcher(**kwargs: Any) -> int:
        launched.append(kwargs)
        return 4321

    artifact = build.run_build(
        output_path=tmp_path / "build.json",
        smoke_output_path=tmp_path / "smoke.json",
        full_output_path=tmp_path / "raw.json",
        log_path=tmp_path / "run.log",
        precondition_checker=lambda: [
            _check("local_gguf_cached", True),
            _check("llama_cpp_importable", True),
            _check("code_corpus_loadable", True),
            _check("restricted_exec_importable", True),
        ],
        smoke_runner=smoke_runner,
        launcher=launcher,
    )
    build.validate_build_artifact(artifact)
    assert artifact["honest_verdict"] == "success: offarc_transfer_runner_built_smoked_launched"
    assert artifact["runner_ready"] is True
    assert artifact["smoke_passed"] is True
    assert artifact["launched_pid"] == 4321
    assert launched and launched[0]["n_tasks"] >= 40 and launched[0]["k"] >= 8


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("runner_ready", {"value": True}, "runner_ready must be a bare bool"),
        ("smoke_passed", "yes", "smoke_passed must be a bare bool"),
        ("launched_pid", True, "launched_pid must be a bare int"),
        ("preconditions_checked", {}, "preconditions_checked must be a list"),
        ("inference_substrate", "cached", "inference_substrate must be live_llm_inference"),
    ],
)
def test_validate_build_artifact_rejects_schema_poison(
    field: str, value: Any, message: str
) -> None:
    artifact = build.build_artifact(
        honest_verdict="blocked_local_gguf_not_cached",
        runner_ready=False,
        smoke_passed=False,
        launched_pid=0,
        preconditions_checked=[_check("local_gguf_cached", False)],
        duration_s=0.1,
    )
    artifact[field] = value
    with pytest.raises(ValueError, match=message):
        build.validate_build_artifact(artifact)


def test_validate_build_artifact_rejects_missing_bad_verdict_and_missing_pid() -> None:
    with pytest.raises(ValueError, match="missing required build field"):
        build.validate_build_artifact({})
    artifact = build.build_artifact(
        honest_verdict="blocked_local_gguf_not_cached",
        runner_ready=False,
        smoke_passed=False,
        launched_pid=0,
        preconditions_checked=[_check("local_gguf_cached", False)],
        duration_s=0.1,
        error="recorded",
    )
    assert artifact["error"] == "recorded"
    artifact["honest_verdict"] = "done"
    with pytest.raises(ValueError, match="terminal prefix"):
        build.validate_build_artifact(artifact)
    artifact["honest_verdict"] = "success: ok"
    artifact["smoke_passed"] = True
    artifact["launched_pid"] = 0
    with pytest.raises(ValueError, match="positive after a successful launch"):
        build.validate_build_artifact(artifact)
