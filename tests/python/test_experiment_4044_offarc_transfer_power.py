"""Tests for Exp 4044/4045 OFF-ARC full-power transfer.

Spec refs: REQ-VERIFY-4044, SCENARIO-VERIFY-4044,
REQ-VERIFY-4045, SCENARIO-VERIFY-4045.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

import exp4045_offarc_transfer_power_collect as collect
import exp4044_offarc_transfer_power_build as build
import offarc_transfer_power_run as runner


def _check(resource: str, available: bool) -> dict[str, Any]:
    return {"resource": resource, "available": available}


def _task(task_id: str = "HumanEval/fixture") -> runner.CodeTask:
    return runner.CodeTask(
        task_id=task_id,
        corpus="humaneval",
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
    fingerprint_outputs: list[Any] | None = None,
    code: str | None = None,
    truncated: bool = False,
) -> runner.CandidateEvaluation:
    return runner.CandidateEvaluation(
        task_id=task_id,
        draw_index=draw_index,
        status="ok",
        code=code or f"def add(a, b):\n    return a + b  # {task_id}-{draw_index}\n",
        visible_passes=visible_passes,
        hidden_passes=hidden_passes,
        visible_outputs=visible_outputs,
        hidden_outputs=["hidden"],
        fingerprint_outputs=fingerprint_outputs or visible_outputs,
        generation_seconds=0.1,
        truncated=truncated,
        error=None,
    )


def _synthetic_evaluations() -> tuple[
    list[runner.CodeTask], dict[str, list[runner.CandidateEvaluation]]
]:
    tasks = [_task("HumanEval/1"), _task("mbpp-1")]
    evaluations = {
        "HumanEval/1": [
            _eval(
                "HumanEval/1",
                0,
                visible_passes=[False, False],
                hidden_passes=[False],
                visible_outputs=[0, 0],
                fingerprint_outputs=[10, 10],
            ),
            _eval(
                "HumanEval/1",
                1,
                visible_passes=[True, True],
                hidden_passes=[True],
                visible_outputs=[3, 9],
                fingerprint_outputs=[8, 12],
            ),
            _eval(
                "HumanEval/1",
                2,
                visible_passes=[True, True],
                hidden_passes=[False],
                visible_outputs=[3, 9],
                fingerprint_outputs=[8, 12],
            ),
        ],
        "mbpp-1": [
            _eval(
                "mbpp-1",
                0,
                visible_passes=[True, True],
                hidden_passes=[False],
                visible_outputs=[3, 9],
                fingerprint_outputs=[1, 1],
                truncated=True,
            ),
            _eval(
                "mbpp-1",
                1,
                visible_passes=[True, True],
                hidden_passes=[True],
                visible_outputs=[3, 9],
                fingerprint_outputs=[7, 7],
            ),
            _eval(
                "mbpp-1",
                2,
                visible_passes=[True, True],
                hidden_passes=[True],
                visible_outputs=[3, 9],
                fingerprint_outputs=[7, 7],
            ),
        ],
    }
    return tasks, evaluations


def _collect_raw(rows: list[dict[str, bool]]) -> dict[str, Any]:
    per_task = []
    for index, row in enumerate(rows):
        per_task.append(
            {
                "task_id": f"T{index}",
                "corpus": "fixture",
                "func_name": "f",
                "n_candidates": 8,
                "n_visible_tests": 2,
                "n_hidden_tests": 1,
                "armA_vote_pass1": row["a"],
                "armAplusplus_aces_pass1": row["app"],
                "armB_demofit_pass1": row["b"],
                "armC_symbolic_partition_pass1": row["c"],
                "oracle_hidden_pass": row["oracle"],
            }
        )

    def rate(key: str) -> float:
        return round(sum(1 for row in rows if row[key]) / max(1, len(rows)), 6)

    return {
        "experiment": "experiment_4045_offarc_transfer_power",
        "schema": "carnot.experiment_4045_offarc_transfer_power_raw.v1",
        "honest_verdict": "success: fixture_raw",
        "corpus": "humaneval_plus_mbpp",
        "n_tasks": len(rows),
        "armA_vote_passrate": rate("a"),
        "armAplusplus_aces_passrate": rate("app"),
        "armB_demofit_passrate": rate("b"),
        "armC_symbolic_partition_passrate": rate("c"),
        "oracle_passrate": rate("oracle"),
        "model_specs": {"local_generator": "fixture", "verifier": "fixture-pool"},
        "random_seed": 4045,
        "reproducibility_checksum": "raw-fixture",
        "missing_verifier_gaps": [],
        "per_task": per_task,
    }


def test_req_4044_4045_spec_declared() -> None:
    # REQ-VERIFY-4044 / REQ-VERIFY-4045: OpenSpec declares build and four-arm run contracts.
    spec = Path("openspec/capabilities/verification/spec.md").read_text(encoding="utf-8")
    for marker in (
        "REQ-VERIFY-4044",
        "SCENARIO-VERIFY-4044",
        "REQ-VERIFY-4045",
        "SCENARIO-VERIFY-4045",
        "offarc_transfer_power_run.py",
        "exp4045_offarc_transfer_power_collect.py",
        "10,000-resample",
        "armC",
    ):
        assert marker in spec


def test_collect_demofit_ci_excluding_zero_answers_headline_gate() -> None:
    # REQ-VERIFY-4045: the final collector reports the bare demo-fit CI gate.
    raw = _collect_raw(
        [
            {"a": False, "app": False, "b": True, "c": False, "oracle": True},
            {"a": False, "app": False, "b": True, "c": False, "oracle": True},
            {"a": False, "app": False, "b": True, "c": False, "oracle": True},
        ]
    )
    artifact = collect.build_final_artifact(
        raw,
        raw_artifact_present=True,
        partial_reason=None,
        n_bootstrap=400,
        powered_task_floor=3,
    )
    collect.validate_final_artifact(artifact)
    assert artifact["honest_verdict"] == "complete: offarc_demofit_transfers_to_code_ci_excl0_n3"
    assert artifact["demofit_delta_pp"] == pytest.approx(100.0)
    assert artifact["demofit_bootstrap_ci95"][0] > 0.0
    assert artifact["demofit_ci_excludes_zero"] is True
    assert artifact["inference_substrate"] == collect.INFERENCE_SUBSTRATE


def test_collect_stronger_arm_only_verdict_when_demofit_touches_zero() -> None:
    # REQ-VERIFY-4045: stronger arms contextualize a shared demo-fit miss.
    raw = _collect_raw(
        [
            {"a": False, "app": True, "b": False, "c": True, "oracle": True},
            {"a": False, "app": True, "b": False, "c": True, "oracle": True},
            {"a": False, "app": False, "b": False, "c": True, "oracle": True},
            {"a": False, "app": False, "b": False, "c": True, "oracle": True},
        ]
    )
    artifact = collect.build_final_artifact(
        raw,
        raw_artifact_present=True,
        partial_reason=None,
        n_bootstrap=400,
        powered_task_floor=4,
    )
    assert artifact["demofit_ci_excludes_zero"] is False
    assert artifact["best_arm"] == "armC_symbolic"
    assert artifact["best_arm_ci_excludes_zero"] is True
    assert artifact["honest_verdict"] == "complete: offarc_demofit_touches0_symbolic_partition_excl0"
    assert "GAP-CODE-EXEC-DEMOFIT" in artifact["missing_verifier_gaps"]


def test_collect_no_headroom_is_uninformative_not_failure() -> None:
    # REQ-VERIFY-4045: oracle parity with vote blocks a false negative claim.
    raw = _collect_raw(
        [
            {"a": True, "app": True, "b": True, "c": True, "oracle": True},
            {"a": True, "app": True, "b": True, "c": True, "oracle": True},
            {"a": False, "app": False, "b": False, "c": False, "oracle": False},
        ]
    )
    artifact = collect.build_final_artifact(
        raw,
        raw_artifact_present=True,
        partial_reason=None,
        n_bootstrap=200,
        powered_task_floor=3,
    )
    assert artifact["oracle_headroom"] is False
    assert artifact["honest_verdict"] == "complete: offarc_transfer_uninformative_no_oracle_headroom"


def test_collect_aces_only_and_small_magnitude_verdicts() -> None:
    # REQ-VERIFY-4045: non-demo stronger-arm and all-touch-zero cases are distinct.
    aces_raw = _collect_raw(
        [
            {"a": False, "app": True, "b": False, "c": False, "oracle": True},
            {"a": False, "app": True, "b": False, "c": False, "oracle": True},
            {"a": False, "app": True, "b": False, "c": False, "oracle": True},
        ]
    )
    aces = collect.build_final_artifact(
        aces_raw,
        raw_artifact_present=True,
        partial_reason=None,
        n_bootstrap=200,
        powered_task_floor=3,
    )
    assert aces["honest_verdict"] == "complete: offarc_demofit_touches0_aces_excl0"

    small_raw = _collect_raw(
        [
            {"a": False, "app": False, "b": False, "c": False, "oracle": True},
            {"a": False, "app": False, "b": False, "c": False, "oracle": True},
            {"a": False, "app": False, "b": False, "c": False, "oracle": True},
        ]
    )
    small = collect.build_final_artifact(
        small_raw,
        raw_artifact_present=True,
        partial_reason=None,
        n_bootstrap=200,
        powered_task_floor=3,
    )
    assert small["honest_verdict"] == "complete: offarc_transfer_small_magnitude_all_arms_touch0"


def test_collect_negative_ci_is_honest_negative() -> None:
    # SCENARIO-VERIFY-4045: an excluding-zero CI in the wrong direction is not transfer.
    raw = _collect_raw(
        [
            {"a": True, "app": False, "b": False, "c": False, "oracle": True},
            {"a": True, "app": False, "b": False, "c": False, "oracle": True},
            {"a": True, "app": False, "b": False, "c": False, "oracle": True},
        ]
    )
    artifact = collect.build_final_artifact(
        raw,
        raw_artifact_present=True,
        partial_reason=None,
        n_bootstrap=200,
        powered_task_floor=3,
    )
    assert artifact["demofit_ci_excludes_zero"] is True
    assert artifact["demofit_bootstrap_ci95"][1] < 0.0
    assert artifact["honest_verdict"] == "complete: offarc_demofit_negative_ci_excl0_n3"


def test_collect_raw_ready_path_validates_raw_and_tails_log(tmp_path: Path) -> None:
    # REQ-VERIFY-4045: a landed raw artifact is preferred over checkpoint fallback.
    build_path = tmp_path / "build.json"
    raw_path = tmp_path / "raw.json"
    output_path = tmp_path / "final.json"
    log_path = tmp_path / "offarc.log"
    tasks, evaluations = _synthetic_evaluations()
    raw = runner.build_raw_artifact(
        tasks=tasks,
        evaluations_by_task=evaluations,
        preconditions_checked=[_check("all", True)],
        model_specs={"generator_model": "fixture", "verifier": "fixture"},
        k=8,
        started_s=0.0,
        ended_s=2.0,
        mode="smoke",
    )
    raw_path.write_text(__import__("json").dumps(raw), encoding="utf-8")
    log_path.write_text("old\nnew\n", encoding="utf-8")
    build_path.write_text(
        __import__("json").dumps(
            {
                "honest_verdict": "success: launched",
                "runner_ready": True,
                "smoke_passed": True,
                "launched_pid": 123,
                "preconditions_checked": [_check("all", True)],
                "full_raw_path": str(raw_path),
                "log_path": str(log_path),
            }
        ),
        encoding="utf-8",
    )
    artifact = collect.run_collect(
        output_path=output_path,
        build_path=build_path,
        raw_path=raw_path,
        checkpoint_path=tmp_path / "checkpoint.json",
        poll_budget_s=0.0,
        n_bootstrap=50,
    )
    assert artifact["raw_artifact_present"] is True
    assert artifact["log_tail"] == "old\nnew"


def test_collect_poll_for_raw_reads_log_and_uses_sleeper(tmp_path: Path) -> None:
    # REQ-VERIFY-4045: bounded polling checks the log while waiting for raw output.
    raw_path = tmp_path / "raw.json"
    log_path = tmp_path / "offarc.log"
    log_path.write_text("boot\nprogress\n", encoding="utf-8")
    slept: list[float] = []

    def sleeper(seconds: float) -> None:
        slept.append(seconds)
        raw_path.write_text("{}", encoding="utf-8")

    assert collect.poll_for_raw(
        raw_path,
        log_path=log_path,
        poll_budget_s=10.0,
        poll_interval_s=1.0,
        sleeper=sleeper,
    )
    assert slept == [pytest.approx(1.0)]


def test_collect_partial_checkpoint_writes_partial_artifact(tmp_path: Path) -> None:
    # REQ-VERIFY-4045: incomplete background runs are summarized from checkpoint data only.
    build_path = tmp_path / "build.json"
    raw_path = tmp_path / "missing_raw.json"
    checkpoint_path = tmp_path / "checkpoint.json"
    output_path = tmp_path / "final.json"
    log_path = tmp_path / "offarc.log"
    tasks, evaluations = _synthetic_evaluations()
    runner._write_checkpoint(
        checkpoint_path,
        tasks=tasks,
        evaluations_by_task=evaluations,
        skipped_tasks=[],
        k=8,
        mode="full",
    )
    build_path.write_text(
        __import__("json").dumps(
            {
                "honest_verdict": "success: launched",
                "runner_ready": True,
                "smoke_passed": True,
                "launched_pid": 123,
                "preconditions_checked": [_check("all", True)],
                "full_raw_path": str(raw_path),
                "log_path": str(log_path),
            }
        ),
        encoding="utf-8",
    )
    artifact = collect.run_collect(
        output_path=output_path,
        build_path=build_path,
        raw_path=raw_path,
        checkpoint_path=checkpoint_path,
        log_path=log_path,
        poll_budget_s=0.0,
        poll_interval_s=60.0,
        task_loader=lambda: tasks,
        n_bootstrap=100,
    )
    assert artifact["honest_verdict"] == "complete: offarc_power_run_incomplete_partial_2_tasks"
    assert artifact["raw_artifact_present"] is False
    assert artifact["n_tasks"] == 2
    assert output_path.exists()


def test_collect_checkpoint_empty_and_fallback_task_paths(tmp_path: Path) -> None:
    # REQ-VERIFY-4045: missing or sparse checkpoint state becomes honest partial evidence.
    missing_raw = collect.raw_from_checkpoint(
        checkpoint_path=tmp_path / "missing.json",
        build={"preconditions_checked": [_check("all", True)]},
        task_loader=lambda: [],
    )
    assert missing_raw["n_tasks"] == 0
    assert missing_raw["per_task"] == []

    empty_checkpoint = tmp_path / "empty.json"
    empty_checkpoint.write_text(
        __import__("json").dumps({"evaluations_by_task": {}, "ordered_task_ids": []}),
        encoding="utf-8",
    )
    empty_raw = collect.raw_from_checkpoint(
        checkpoint_path=empty_checkpoint,
        build={"preconditions_checked": []},
        task_loader=lambda: [],
    )
    assert empty_raw["n_tasks"] == 0

    fallback_checkpoint = tmp_path / "fallback.json"
    fallback_checkpoint.write_text(
        __import__("json").dumps(
            {
                "evaluations_by_task": {
                    "HumanEval/fallback": [
                        _eval(
                            "HumanEval/fallback",
                            0,
                            visible_passes=[True, True],
                            hidden_passes=[True],
                            visible_outputs=[1, 1],
                        ).__dict__
                    ]
                },
                "ordered_task_ids": ["HumanEval/fallback"],
            }
        ),
        encoding="utf-8",
    )
    fallback_raw = collect.raw_from_checkpoint(
        checkpoint_path=fallback_checkpoint,
        build={"preconditions_checked": []},
        task_loader=lambda: [],
    )
    assert fallback_raw["n_tasks"] == 1
    assert fallback_raw["per_task"][0]["corpus"] == "humaneval"


def test_collect_blocks_when_build_runner_not_ready(tmp_path: Path) -> None:
    # REQ-VERIFY-4045: the collector stops if Exp 4044 did not launch a ready runner.
    build_path = tmp_path / "build.json"
    output_path = tmp_path / "final.json"
    build_path.write_text(
        __import__("json").dumps(
            {
                "honest_verdict": "blocked_local_gguf_not_cached",
                "runner_ready": False,
                "smoke_passed": False,
                "launched_pid": 0,
                "preconditions_checked": [_check("local_gguf_cached", False)],
            }
        ),
        encoding="utf-8",
    )
    artifact = collect.run_collect(
        output_path=output_path,
        build_path=build_path,
        raw_path=tmp_path / "raw.json",
        checkpoint_path=tmp_path / "checkpoint.json",
        poll_budget_s=0.0,
    )
    assert artifact["honest_verdict"] == "blocked_build_runner_not_ready"
    assert artifact["runner_ready"] is False
    assert output_path.exists()


def test_collect_validation_edge_cases() -> None:
    # REQ-VERIFY-4045: validation fails closed on malformed final and blocked artifacts.
    assert collect.bootstrap_delta_ci95([], seed=1) == [0.0, 0.0]
    with pytest.raises(ValueError, match="runner_ready=false"):
        collect.validate_final_artifact(
            {"honest_verdict": "blocked_build_runner_not_ready", "runner_ready": True}
        )

    artifact = collect.build_final_artifact(
        _collect_raw(
            [{"a": False, "app": False, "b": True, "c": False, "oracle": True}]
        ),
        raw_artifact_present=True,
        partial_reason=None,
        n_bootstrap=50,
        powered_task_floor=1,
    )
    missing = dict(artifact)
    missing.pop("corpus")
    with pytest.raises(ValueError, match="missing required final field"):
        collect.validate_final_artifact(missing)

    bad_verdict = dict(artifact)
    bad_verdict["honest_verdict"] = "success: not-final"
    with pytest.raises(ValueError, match="complete: terminal prefix"):
        collect.validate_final_artifact(bad_verdict)

    bad_float = dict(artifact)
    bad_float["armA_vote_passrate"] = "0.0"
    with pytest.raises(ValueError, match="armA_vote_passrate must be a bare float"):
        collect.validate_final_artifact(bad_float)

    bad_gaps = dict(artifact)
    bad_gaps["missing_verifier_gaps"] = "gap"
    with pytest.raises(ValueError, match="missing_verifier_gaps must be a list"):
        collect.validate_final_artifact(bad_gaps)

    bad_checksum = dict(artifact)
    bad_checksum["reproducibility_checksum"] = ""
    with pytest.raises(ValueError, match="reproducibility_checksum must be non-empty"):
        collect.validate_final_artifact(bad_checksum)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("demofit_ci_excludes_zero", "false", "demofit_ci_excludes_zero must be a bare bool"),
        ("best_arm_ci_excludes_zero", 1, "best_arm_ci_excludes_zero must be a bare bool"),
        ("n_tasks", 1.5, "n_tasks must be a bare int"),
        ("demofit_bootstrap_ci95", [0.0], "demofit_bootstrap_ci95 must be a two-element list"),
        ("model_specs", [], "model_specs must be an object"),
        ("inference_substrate", "live_llm_inference", "verifier-scoring substrate"),
    ],
)
def test_validate_final_artifact_rejects_schema_poison(
    field: str, value: Any, message: str
) -> None:
    # REQ-VERIFY-4045: final collector fields stay bare and verifier-scoring only.
    artifact = collect.build_final_artifact(
        _collect_raw(
            [{"a": False, "app": False, "b": True, "c": False, "oracle": True}]
        ),
        raw_artifact_present=True,
        partial_reason=None,
        n_bootstrap=50,
        powered_task_floor=1,
    )
    artifact[field] = value
    with pytest.raises(ValueError, match=message):
        collect.validate_final_artifact(artifact)


def test_public_test_parsing_handles_mbpp_and_humaneval_candidate_asserts() -> None:
    # REQ-VERIFY-4045: visible/hidden tests are normalized to exact-output demos.
    mbpp = runner.parse_assertion_test("assert add(2, 3) == 5")
    assert mbpp == runner.CodeTest("assert add(2, 3) == 5", "add", (2, 3), 5)

    hidden = runner.parse_humaneval_candidate_assert(
        "assert candidate(['a', 'ab'], 'a') == ['a', 'ab']",
        func_name="filter_by_prefix",
    )
    assert hidden == runner.CodeTest(
        "assert filter_by_prefix(['a', 'ab'], 'a') == ['a', 'ab']",
        "filter_by_prefix",
        (["a", "ab"], "a"),
        ["a", "ab"],
    )
    assert runner.parse_humaneval_candidate_assert("assert candidate(x) == 1", "f") is None


def test_parser_and_code_extraction_edge_cases_are_bounded() -> None:
    # REQ-VERIFY-4045: unsupported public tests fail closed instead of becoming demos.
    assert runner.parse_assertion_test("assert (") is None
    assert runner.parse_assertion_test("print('x')") is None
    assert runner.parse_assertion_test("assert add(1) > 0") is None
    assert runner.parse_assertion_test("assert 5 == add(2, 3)") == runner.CodeTest(
        "assert 5 == add(2, 3)", "add", (2, 3), 5
    )
    assert runner.parse_assertion_test("assert add(1, b=2) == 3") is None
    assert runner.parse_assertion_test("assert add(x) == 3") is None
    assert runner.parse_assertion_test("assert add(1) == x") is None

    assert runner.parse_humaneval_candidate_assert("assert (", "f") is None
    assert runner.parse_humaneval_candidate_assert("print('x')", "f") is None
    assert runner.parse_humaneval_candidate_assert("assert candidate(1)", "f") == runner.CodeTest(
        "assert f(1,) == True", "f", (1,), True
    )
    assert runner.parse_humaneval_candidate_assert(
        "assert not candidate(1)", "f"
    ) == runner.CodeTest("assert f(1,) == False", "f", (1,), False)
    assert runner.parse_humaneval_candidate_assert("assert candidate(1) > 0", "f") is None
    assert runner.parse_humaneval_candidate_assert(
        "assert 2 == candidate(1)", "f"
    ) == runner.CodeTest("assert f(1,) == 2", "f", (1,), 2)
    assert runner.parse_humaneval_candidate_assert("assert candidate(x) == 1", "f") is None
    assert runner.parse_humaneval_candidate_assert("assert candidate(1) == x", "f") is None

    text = "notes\n```python\ndef add(a, b):\n    return a + b\n```"
    assert runner.extract_code_block(text, "add") == "def add(a, b):\n    return a + b"
    assert runner.extract_code_block("prefix\ndef add(a, b):\n    return a - b", "add").startswith(
        "def add"
    )
    assert runner.extract_code_block("no matching function", "add") == ""
    assert runner._stable_repr({1, 2}) == repr({1, 2})


def test_evaluate_candidate_records_behavioral_fingerprint_with_shared_executor() -> None:
    # REQ-VERIFY-4045: Arm C fingerprints use the same sandbox-compatible execution path.
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
    evaluated = runner.evaluate_candidate(
        _task(),
        candidate,
        executor=executor,
        fingerprint_tests=[runner.CodeTest("assert add(10, 11) == 21", "add", (10, 11), 21)],
    )
    assert evaluated.visible_passes == [True, True]
    assert evaluated.hidden_passes == [True]
    assert evaluated.fingerprint_outputs == [21]
    assert calls[-1] == ("add", (10, 11))


def test_evaluate_candidate_records_no_code_and_executor_errors() -> None:
    # REQ-VERIFY-4045: malformed candidates and sandbox errors stay in the shared pool as failures.
    no_code = runner.GeneratedCandidate(0, "text", "", 0.1, "stop", False)
    evaluated = runner.evaluate_candidate(
        _task(),
        no_code,
        executor=lambda *args: (None, None),
        fingerprint_tests=[runner.CodeTest("probe", "add", (1, 1), None)],
    )
    assert evaluated.status == "no_code"
    assert evaluated.fingerprint_outputs == ["no_code"]

    def executor(*args: Any) -> tuple[Any, Exception | None]:
        del args
        return None, RuntimeError("boom")

    candidate = runner.GeneratedCandidate(
        1, "raw", "def add(a, b):\n    return a + b\n", 0.1, "stop", False
    )
    evaluated = runner.evaluate_candidate(
        _task(),
        candidate,
        executor=executor,
        fingerprint_tests=[runner.CodeTest("probe", "add", (1, 1), None)],
    )
    assert evaluated.visible_passes == [False, False]
    assert evaluated.hidden_passes == [False]
    assert evaluated.fingerprint_outputs[0]["error"].startswith("RuntimeError")
    assert "RuntimeError" in (evaluated.error or "")


def test_aces_and_partition_degenerate_inputs_are_bounded() -> None:
    # REQ-VERIFY-4045: selection arms are deterministic on empty and degenerate pools.
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
    assert runner._rank_demofit(tied) == []
    ranked, meta = runner._rank_symbolic_partition(tied)
    assert ranked == []
    assert meta["partition_count"] == 0
    assert runner._bootstrap_ci_pp([], seed=1) == [0.0, 0.0]


def test_score_evaluated_tasks_measures_four_arms_and_oracle() -> None:
    # SCENARIO-VERIFY-4045: all four arms share one pool and hidden scoring happens after selection.
    tasks, evaluations = _synthetic_evaluations()
    scored = runner.score_evaluated_tasks(tasks, evaluations, seed=4045)
    assert scored["armA_vote_passrate"] == pytest.approx(0.5)
    assert scored["armAplusplus_aces_passrate"] == pytest.approx(1.0)
    assert scored["armB_demofit_passrate"] == pytest.approx(0.5)
    assert scored["armC_symbolic_partition_passrate"] == pytest.approx(1.0)
    assert scored["oracle_passrate"] == pytest.approx(1.0)
    assert scored["truncation_rate"] == pytest.approx(1 / 6)
    assert scored["per_task"][1]["armC_partition_count"] == 2
    assert scored["per_task"][1]["armC_dominant_partition_size"] == 2


def test_score_evaluated_tasks_records_missing_gap_when_oracle_has_headroom() -> None:
    # SCENARIO-VERIFY-4045: oracle-only hits are preserved as missing verifier gaps.
    task = _task("HumanEval/gap")
    evaluations = {
        "HumanEval/gap": [
            _eval(
                "HumanEval/gap",
                0,
                visible_passes=[False, False],
                hidden_passes=[True],
                visible_outputs=[1, 1],
            )
        ]
    }
    scored = runner.score_evaluated_tasks([task], evaluations, seed=1)
    assert scored["missing_verifier_gaps"] == ["HumanEval/gap"]


def test_build_raw_artifact_validates_required_4045_fields_and_checksum() -> None:
    # REQ-VERIFY-4045: raw artifact contains the four-arm metrics and reproducibility checksum.
    tasks, evaluations = _synthetic_evaluations()
    artifact = runner.build_raw_artifact(
        tasks=tasks,
        evaluations_by_task=evaluations,
        preconditions_checked=[
            _check("local_gguf_cached", True),
            _check("llama_cpp_importable", True),
            _check("humaneval_corpus_loadable", True),
            _check("mbpp_corpus_loadable", True),
            _check("restricted_exec_importable", True),
        ],
        model_specs={"generator_model": "gemma-4-12B", "verifier": "sandboxed demo-fit"},
        k=3,
        started_s=10.0,
        ended_s=12.0,
        mode="smoke",
        skipped_tasks=[{"task_id": "HumanEval/skip", "reason": "unsupported_non_exact"}],
    )
    runner.validate_raw_artifact(artifact, require_full=False)
    assert artifact["arms_implemented"] == runner.ARMS_IMPLEMENTED
    assert artifact["reproducibility_checksum"]
    assert artifact["skipped_tasks"][0]["reason"] == "unsupported_non_exact"


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("honest_verdict", "done", "terminal prefix"),
        ("n_tasks", 2.5, "n_tasks must be a bare int"),
        ("armA_vote_passrate", "0.0", "armA_vote_passrate must be a bare float"),
        ("bootstrap_ci95", [0.0], "bootstrap_ci95 must be a two-element list"),
        ("bootstrap_ci95", [0.0, "bad"], "bootstrap_ci95 values must be numeric"),
        ("arms_implemented", ["armA"], "four required arms"),
        ("model_specs", [], "model_specs must be an object"),
        ("reproducibility_checksum", "", "non-empty string"),
        ("preconditions_checked", {}, "preconditions_checked must be a list"),
        ("candidate_pool", [], "candidate_pool must be an object"),
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


def test_validate_raw_artifact_rejects_missing_and_runner_ready() -> None:
    with pytest.raises(ValueError, match="missing required raw field"):
        runner.validate_raw_artifact({}, require_full=False)
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
    artifact["runner_ready"] = True
    with pytest.raises(ValueError, match="unexpected raw artifact field"):
        runner.validate_raw_artifact(artifact, require_full=False)


def test_full_verdict_branches_cover_success_no_headroom_and_no_transfer() -> None:
    assert runner._verdict(
        {
            "armB_demofit_passrate": 1.0,
            "armA_vote_passrate": 0.0,
            "bootstrap_ci95": [1.0, 100.0],
            "armC_symbolic_partition_passrate": 0.0,
            "bootstrap_ci95_armC_vs_armA": [0.0, 0.0],
            "oracle_passrate": 1.0,
        },
        mode="full",
    ).startswith("success: offarc_power_demofit")
    assert runner._verdict(
        {
            "armB_demofit_passrate": 0.0,
            "armA_vote_passrate": 0.0,
            "bootstrap_ci95": [0.0, 0.0],
            "armC_symbolic_partition_passrate": 1.0,
            "bootstrap_ci95_armC_vs_armA": [1.0, 100.0],
            "oracle_passrate": 1.0,
        },
        mode="full",
    ).startswith("success: offarc_power_sep")
    assert runner._verdict(
        {
            "armB_demofit_passrate": 0.5,
            "armA_vote_passrate": 0.5,
            "bootstrap_ci95": [0.0, 0.0],
            "armC_symbolic_partition_passrate": 0.5,
            "bootstrap_ci95_armC_vs_armA": [0.0, 0.0],
            "oracle_passrate": 0.5,
        },
        mode="full",
    ).startswith("complete: offarc_power_uninformative")
    assert runner._verdict(
        {
            "armB_demofit_passrate": 0.0,
            "armA_vote_passrate": 0.5,
            "bootstrap_ci95": [-50.0, 0.0],
            "armC_symbolic_partition_passrate": 0.5,
            "bootstrap_ci95_armC_vs_armA": [0.0, 0.0],
            "oracle_passrate": 1.0,
        },
        mode="full",
    ).startswith("complete: offarc_power_no_ci")


def test_manifest_parsers_and_fingerprint_mutation_helpers(tmp_path: Path) -> None:
    # REQ-VERIFY-4045: cached corpus rows become exact-output tasks with explicit skip accounting.
    he_path = tmp_path / "humaneval.jsonl"
    he_rows = [
        {
            "stable_id": "HumanEval/ok",
            "entry_point": "add",
            "prompt": ">>> add(1, 2)\n3\n",
            "tests": "def check(candidate):\n    assert candidate(1, 2) == 3\n    assert candidate(2, 3) == 5\n",
        },
        {
            "stable_id": "HumanEval/doc",
            "entry_point": "neg",
            "prompt": ">>> neg(1)\n-1\n",
            "tests": "def check(candidate):\n    assert candidate(2) == -2\n",
        },
        {
            "stable_id": "HumanEval/skip",
            "entry_point": "f",
            "prompt": "def f(x): pass",
            "tests": "def check(candidate):\n    assert candidate(x) == 1\n",
        },
    ]
    he_path.write_text(
        "\n".join(__import__("json").dumps(row) for row in he_rows), encoding="utf-8"
    )
    tasks, skipped = runner.load_humaneval_tasks(limit=10, manifest_path=he_path)
    assert [task.task_id for task in tasks] == ["HumanEval/ok", "HumanEval/doc"]
    assert skipped[0]["task_id"] == "HumanEval/skip"

    mbpp_path = tmp_path / "mbpp.jsonl"
    mbpp_path.write_text(
        __import__("json").dumps(
            {
                "stable_id": "mbpp-ok",
                "prompt": "add",
                "tests": [
                    "assert add(1, 2) == 3",
                    "assert add(2, 3) == 5",
                    "assert add(3, 4) == 7",
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    assert runner.load_mbpp_tasks(limit=1, manifest_path=mbpp_path)[0].task_id == "mbpp-ok"
    assert runner.load_mbpp_tasks(limit=0, manifest_path=mbpp_path) == []

    task = _task()
    probes = runner.build_fingerprint_tests(task, max_tests=1)
    assert len(probes) == 1
    assert runner._mutate_public_arg(True) is False
    assert runner._mutate_public_arg(1) == 2
    assert runner._mutate_public_arg(1.0) == 1.5
    assert runner._mutate_public_arg("x") == "xa"
    assert runner._mutate_public_arg([1, 2]) == [2, 1]
    assert runner._mutate_public_arg([]) == []
    assert runner._mutate_public_arg((1, 2)) == (2, 1)
    assert runner._mutate_public_arg(()) == ()
    assert runner._mutate_public_arg({"a": 1, "b": 2}) == {"b": 2, "a": 1}
    sentinel = object()
    assert runner._mutate_public_arg(sentinel) is sentinel

    no_probe_task = runner.CodeTask(
        task_id="stable",
        corpus="fixture",
        prompt="noop",
        func_name="identity",
        visible_tests=[
            runner.CodeTest("assert identity(None) == None", "identity", (sentinel,), None)
        ],
        hidden_tests=[],
    )
    assert runner.build_fingerprint_tests(no_probe_task) == []


def test_humaneval_parser_edge_cases_are_non_promoting() -> None:
    # REQ-VERIFY-4045: malformed HumanEval snippets and doctests are skipped.
    assert runner._parse_humaneval_tests("def check(:", "f") == []
    assert runner._parse_docstring_examples(">>> f(1)\n", "f") == []
    assert runner._parse_docstring_examples(">>> x\n1\n", "f") == []
    prompt = "\n".join(
        [
            ">>>",
            ">>> print('not a call')",
            "x",
            ">>> other(1)",
            "1",
            ">>> f(x)",
            "1",
            ">>> f(1)",
            "not_literal(",
        ]
    )
    assert runner._parse_docstring_examples(prompt, "f") == []


def test_checkpoint_json_round_trip(tmp_path: Path) -> None:
    # REQ-VERIFY-4045: checkpoints persist enough per-task state to resume.
    path = tmp_path / "checkpoint.json"
    tasks, evaluations = _synthetic_evaluations()
    runner._write_checkpoint(
        path,
        tasks=tasks,
        evaluations_by_task=evaluations,
        skipped_tasks=[{"task_id": "skip"}],
        k=3,
        mode="smoke",
    )
    loaded = runner._load_checkpoint(path)
    assert sorted(loaded) == ["HumanEval/1", "mbpp-1"]
    assert loaded["HumanEval/1"][0].draw_index == 0
    bad = tmp_path / "bad.json"
    bad.write_text("{", encoding="utf-8")
    assert runner._load_checkpoint(bad) == {}
    assert runner._load_checkpoint(tmp_path / "missing.json") == {}


def test_raw_artifact_full_mode_requires_power_floor() -> None:
    # REQ-VERIFY-4045: the full run gate targets at least 160 tasks and k>=8.
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


def test_precondition_blocker_uses_first_missing_power_resource() -> None:
    # REQ-VERIFY-4044: missing resources block before smoke, inference, or launch.
    checks = [
        _check("local_gguf_cached", True),
        _check("llama_cpp_importable", True),
        _check("humaneval_corpus_loadable", False),
        _check("mbpp_corpus_loadable", False),
    ]
    assert runner.blocker_from_preconditions(checks) == "blocked_humaneval_corpus_unavailable"
    assert runner.blocker_from_preconditions([_check("local_gguf_cached", True)]) is None


def test_build_artifact_blocked_path_does_not_smoke_or_launch(tmp_path: Path) -> None:
    # REQ-VERIFY-4044: blocked preconditions write the build artifact and stop.
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
    assert artifact["arms_implemented"] == runner.ARMS_IMPLEMENTED


def test_build_artifact_smoke_failure_blocks_launch(tmp_path: Path) -> None:
    # REQ-VERIFY-4044: a failed two-task smoke prevents launching the full run.
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
    # REQ-VERIFY-4044: launch failures are honest blocked artifacts after smoke passes.
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
    # SCENARIO-VERIFY-4044: a valid smoke artifact gates the background full-power launch.
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
        return 4045

    artifact = build.run_build(
        output_path=tmp_path / "build.json",
        smoke_output_path=tmp_path / "smoke.json",
        full_output_path=tmp_path / "raw.json",
        log_path=tmp_path / "run.log",
        precondition_checker=lambda: [
            _check("local_gguf_cached", True),
            _check("llama_cpp_importable", True),
            _check("humaneval_corpus_loadable", True),
            _check("mbpp_corpus_loadable", True),
            _check("restricted_exec_importable", True),
        ],
        smoke_runner=smoke_runner,
        launcher=launcher,
    )
    build.validate_build_artifact(artifact)
    assert (
        artifact["honest_verdict"]
        == "success: offarc_power_runner_built_smoked_launched_humaneval_mbpp"
    )
    assert artifact["runner_ready"] is True
    assert artifact["smoke_passed"] is True
    assert artifact["launched_pid"] == 4045
    assert launched and launched[0]["n_tasks"] >= 160 and launched[0]["k"] >= 8


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("runner_ready", {"value": True}, "runner_ready must be a bare bool"),
        ("smoke_passed", "yes", "smoke_passed must be a bare bool"),
        ("launched_pid", True, "launched_pid must be a bare int"),
        ("arms_implemented", "armA", "arms_implemented must be a list"),
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
    artifact["launched_pid"] = 0
    with pytest.raises(ValueError, match="positive after a successful launch"):
        build.validate_build_artifact(artifact)
    artifact["honest_verdict"] = "blocked_x"
    artifact["arms_implemented"] = ["armA"]
    with pytest.raises(ValueError, match="four required arms"):
        build.validate_build_artifact(artifact)
