"""Tests for Exp 1536 SATQuest CNF verifier benchmark.

Spec: REQ-BENCH-1536, SCENARIO-BENCH-1536.
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from typing import Any

import pytest

from carnot.eval import satquest_cnf_verifier_benchmark as exp


def test_req_bench_1536_instances_are_balanced_and_reproducibly_labeled() -> None:
    """REQ-BENCH-1536: bounded CNFs include SAT and UNSAT deterministic labels."""

    first = exp.build_cnf_instances(run_date="20260508")
    second = exp.build_cnf_instances(run_date="20260508")

    assert first == second
    assert len(first) >= 6
    assert {case.oracle.is_satisfiable for case in first} == {True, False}
    assert max(instance.n_vars for instance in first) <= exp.MAX_EXHAUSTIVE_VARS
    assert all(instance.oracle.backend for instance in first)
    assert all(instance.oracle.checked_assignments > 0 for instance in first)


def test_req_bench_1536_exact_solver_checks_assignments_and_unsat() -> None:
    """REQ-BENCH-1536: the local solver fallback is authoritative for bounded CNFs."""

    sat = exp.solve_cnf_exact(2, ((1, 2), (-1, 2)))
    unsat = exp.solve_cnf_exact(1, ((1,), (-1,)))

    assert sat.is_satisfiable is True
    assert sat.satisfying_assignment is not None
    assert exp.assignment_satisfies(((1, 2), (-1, 2)), sat.satisfying_assignment)
    assert unsat.is_satisfiable is False
    assert unsat.satisfying_assignment is None
    assert unsat.checked_assignments == 2
    with pytest.raises(ValueError, match="bounded exhaustive solver"):
        exp.solve_cnf_exact(exp.MAX_EXHAUSTIVE_VARS + 1, ((1,),))


def test_req_bench_1536_pysat_and_fallback_branches_are_deterministic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-BENCH-1536: PySAT preference and exact fallback expose the same labels."""

    class FakeSolver:
        def __init__(self, bootstrap_with: list[list[int]]) -> None:
            self.bootstrap_with = bootstrap_with

        def __enter__(self) -> "FakeSolver":
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def solve(self) -> bool:
            return self.bootstrap_with != [[1], [-1]]

        def get_model(self) -> list[int]:
            return [1, -2]

    pysat_module = types.ModuleType("pysat")
    solvers_module = types.ModuleType("pysat.solvers")
    solvers_module.Solver = FakeSolver  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "pysat", pysat_module)
    monkeypatch.setitem(sys.modules, "pysat.solvers", solvers_module)

    sat = exp.solve_cnf_pysat(2, ((1,),))
    unsat = exp.solve_cnf_pysat(1, ((1,), (-1,)))

    assert sat.backend == "pysat"
    assert sat.is_satisfiable is True
    assert sat.satisfying_assignment == (True, False)
    assert unsat.is_satisfiable is False

    monkeypatch.setattr(
        exp,
        "solve_cnf_pysat",
        lambda _n_vars, _clauses: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    assert exp.solve_cnf(1, ((1,),)).backend == "exact_exhaustive_fallback"


def test_scenario_bench_1536_prompts_emit_machine_symbolic_and_narrative_formats() -> None:
    """SCENARIO-BENCH-1536: every prompt format preserves the same CNF oracle label."""

    instances = exp.build_cnf_instances()[:2]
    prompt_cases = exp.build_prompt_cases(instances)

    assert {case.format_name for case in prompt_cases} == {"machine", "symbolic", "narrative"}
    assert len(prompt_cases) == len(instances) * 3
    for instance in instances:
        labels = {
            case.oracle_label
            for case in prompt_cases
            if case.instance.instance_id == instance.instance_id
        }
        assert labels == {instance.oracle.label}

    machine = next(case for case in prompt_cases if case.format_name == "machine")
    symbolic = next(case for case in prompt_cases if case.format_name == "symbolic")
    narrative = next(case for case in prompt_cases if case.format_name == "narrative")
    assert "p cnf" in machine.prompt
    assert "x1" in symbolic.prompt
    assert "Rule 1" in narrative.prompt


def test_req_bench_1536_output_parser_handles_wrappers_candidates_and_junk() -> None:
    """REQ-BENCH-1536: model output parsing is tolerant but schema-bound."""

    payload = {
        "answer": "SAT",
        "assignment": {"x1": True, "x2": False},
        "verifier": {"accept": True},
        "candidate_answers": [
            {"answer": "UNSAT"},
            {"answer": "SAT", "assignment": [True, False]},
        ],
        "repair_hint_answer": {"answer": "UNSAT"},
    }
    parsed = exp.parse_model_answer(f"```json\n{json.dumps(payload)}\n```")

    assert parsed.parse_ok is True
    assert parsed.baseline.label == "SAT"
    assert parsed.baseline.assignment == (True, False)
    assert parsed.model_declared_accept is True
    assert [candidate.label for candidate in parsed.candidates] == ["UNSAT", "SAT"]
    assert parsed.repair_hint.label == "UNSAT"

    missing = exp.parse_model_answer("no JSON")
    malformed = exp.parse_model_answer('{"answer": "maybe"}')
    bad_assignment = exp.parse_model_answer('{"answer": "SAT", "assignment": ["yes"]}')

    assert missing.parse_ok is False
    assert missing.parse_error == "no_json_object"
    assert malformed.parse_error == "answer_not_sat_or_unsat"
    assert bad_assignment.baseline.assignment is None


def test_req_bench_1536_parser_and_energy_edge_branches_are_closed() -> None:
    """REQ-BENCH-1536: malformed assignments and no-candidate rankings fail closed."""

    sat_case = next(case for case in exp.build_prompt_cases() if case.oracle_label == "SAT")

    string_candidate = exp.parse_model_answer('{"answer": true, "candidate_answers": ["UNSAT"]}')
    bad_key = exp.parse_model_answer('{"answer": "SAT", "assignment": {"z1": true}}')
    bad_suffix = exp.parse_model_answer('{"answer": "SAT", "assignment": {"xA": true}}')
    empty_assignment = exp.parse_model_answer('{"answer": "SAT", "assignment": {}}')
    unsupported_assignment = exp.parse_model_answer('{"answer": "SAT", "assignment": 7}')
    non_string_answer = exp.parse_model_answer('{"answer": 3}')
    no_answer = exp._evaluate_candidate(sat_case, exp.CandidateAnswer(None), None)
    missing_assignment = exp._evaluate_candidate(sat_case, exp.CandidateAnswer("SAT"), None)
    invalid_assignment = exp._evaluate_candidate(
        sat_case,
        exp.CandidateAnswer("SAT", tuple(False for _ in range(sat_case.instance.n_vars))),
        None,
    )

    assert string_candidate.baseline.label == "SAT"
    assert string_candidate.candidates[0].label == "UNSAT"
    assert bad_key.baseline.assignment is None
    assert bad_suffix.baseline.assignment is None
    assert empty_assignment.baseline.assignment is None
    assert unsupported_assignment.baseline.assignment is None
    assert non_string_answer.parse_error == "answer_not_sat_or_unsat"
    assert no_answer["classification"] == "no_answer"
    assert missing_assignment["classification"] == "missing_assignment"
    assert invalid_assignment["classification"] == "invalid_assignment"
    assert exp.assignment_satisfies(((3,),), (True,)) is False
    unsat_case = next(case for case in exp.build_prompt_cases() if case.oracle_label == "UNSAT")
    assert exp._candidate_energy(unsat_case, exp.CandidateAnswer("SAT")) > 50.0
    assert exp._energy_rank_candidate(sat_case, ()) == exp.CandidateAnswer(None)


def test_scenario_bench_1536_manifest_rows_count_false_accepts_and_repairs() -> None:
    """SCENARIO-BENCH-1536: wrong answers, false accepts, and repairs are distinct."""

    sat_case = next(case for case in exp.build_prompt_cases() if case.oracle_label == "SAT")
    unsat_case = next(case for case in exp.build_prompt_cases() if case.oracle_label == "UNSAT")
    sat_answer = exp.gold_answer_for_prompt_case(sat_case)

    correct_row = exp.build_manifest_row(
        sat_case,
        {
            "model_hf_id": exp.MANDATED_MODEL_SPECS[0]["hf_id"],
            "model_name": "Qwen3.6-35B-A3B",
            "generation_source": "live_sota_llamacpp",
            "output_text": sat_answer,
            "elapsed_seconds": 0.01,
            "blocker": None,
        },
    )

    false_accept_payload = {
        "answer": "SAT",
        "assignment": {"x1": True},
        "verifier": {"accept": True},
        "candidate_answers": [{"answer": "UNSAT"}],
        "repair_hint_answer": {"answer": "UNSAT"},
    }
    false_accept_row = exp.build_manifest_row(
        unsat_case,
        {
            "model_hf_id": exp.MANDATED_MODEL_SPECS[0]["hf_id"],
            "model_name": "Qwen3.6-35B-A3B",
            "generation_source": "live_sota_llamacpp",
            "output_text": json.dumps(false_accept_payload),
            "elapsed_seconds": 0.01,
            "blocker": None,
        },
    )

    wrong_rejected_payload = {
        "answer": "UNSAT",
        "verifier": {"accept": False},
        "candidate_answers": json.loads(sat_answer)["candidate_answers"],
        "repair_hint_answer": json.loads(sat_answer),
    }
    wrong_rejected_row = exp.build_manifest_row(
        sat_case,
        {
            "model_hf_id": exp.MANDATED_MODEL_SPECS[0]["hf_id"],
            "model_name": "Qwen3.6-35B-A3B",
            "generation_source": "live_sota_llamacpp",
            "output_text": json.dumps(wrong_rejected_payload),
            "elapsed_seconds": 0.01,
            "blocker": None,
        },
    )

    parse_row = exp.build_manifest_row(
        sat_case,
        {
            "model_hf_id": exp.MANDATED_MODEL_SPECS[0]["hf_id"],
            "model_name": "Qwen3.6-35B-A3B",
            "generation_source": "live_sota_llamacpp",
            "output_text": "not-json",
            "elapsed_seconds": 0.01,
            "blocker": None,
        },
    )

    assert correct_row["baseline"]["correct"] is True
    assert false_accept_row["baseline"]["correct"] is False
    assert false_accept_row["verifier"]["self_verifier_false_accept"] is True
    assert false_accept_row["energy_ranked"]["correct"] is True
    assert false_accept_row["repair_hint"]["correct"] is True
    assert wrong_rejected_row["verifier"]["self_verifier_false_accept"] is False
    assert wrong_rejected_row["energy_ranked"]["correct"] is True
    assert parse_row["baseline"]["classification"] == "parse_failure"

    metrics = exp.aggregate_manifest_metrics(
        [correct_row, false_accept_row, wrong_rejected_row, parse_row]
    )
    assert metrics["baseline_accuracy"] == pytest.approx(0.25)
    assert metrics["energy_ranked_accuracy"] == pytest.approx(0.75)
    assert metrics["repair_hint_accuracy"] == pytest.approx(0.75)
    assert metrics["solver_oracle_false_accepts"] == 1
    assert metrics["false_accept_rate"] == pytest.approx(0.25)


def test_req_bench_1536_live_collector_has_injectable_runtime_hooks() -> None:
    """REQ-BENCH-1536: live GGUF collection can be tested without loading a model."""

    class FakeLlama:
        prompts: list[str] = []
        closed = False

        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs

        def __call__(self, prompt: str, **kwargs: Any) -> dict[str, Any]:
            self.prompts.append(prompt)
            case = prompt_cases[len(self.prompts) - 1]
            return {"choices": [{"text": exp.gold_answer_for_prompt_case(case)}]}

        def close(self) -> None:
            self.closed = True

    prompt_cases = exp.build_prompt_cases()[:2]
    spec = {"hf_id": exp.MANDATED_MODEL_SPECS[0]["hf_id"], "gpu": 0}

    ok = exp.collect_live_model_outputs(
        spec,
        prompt_cases,
        resolver=lambda _hf_id: "/tmp/fake.gguf",
        llama_importer=lambda: (True, FakeLlama, None),
        env_preparer=lambda: {},
    )
    missing = exp.collect_live_model_outputs(
        spec,
        prompt_cases,
        resolver=lambda _hf_id: None,
        llama_importer=lambda: (True, FakeLlama, None),
        env_preparer=lambda: {},
    )
    import_failed = exp.collect_live_model_outputs(
        {**spec, "model_path": "/tmp/fake.gguf"},
        prompt_cases,
        llama_importer=lambda: (False, None, "llama_cpp missing"),
        env_preparer=lambda: {},
    )

    assert ok["summary"]["model_used"] is True
    assert len(ok["rows"]) == 2
    assert FakeLlama.prompts == [case.prompt for case in prompt_cases]
    assert missing["summary"]["blocker"] == "model_not_cached"
    assert import_failed["summary"]["blocker"] == "llama_cpp missing"


def test_req_bench_1536_live_collector_reports_load_and_generation_failures() -> None:
    """REQ-BENCH-1536: collector records load errors and per-case blockers."""

    class LoadFails:
        def __init__(self, **_kwargs: Any) -> None:
            raise RuntimeError("load failed")

    class GenerateFails:
        def __init__(self, **_kwargs: Any) -> None:
            pass

        def __call__(self, _prompt: str, **_kwargs: Any) -> dict[str, Any]:
            raise RuntimeError("generation failed")

    prompt_cases = exp.build_prompt_cases()[:1]
    spec = {"hf_id": exp.MANDATED_MODEL_SPECS[0]["hf_id"], "model_path": "/tmp/fake.gguf"}

    load_failed = exp.collect_live_model_outputs(
        spec,
        prompt_cases,
        llama_importer=lambda: (True, LoadFails, None),
        env_preparer=lambda: {},
    )
    generate_failed = exp.collect_live_model_outputs(
        spec,
        prompt_cases,
        llama_importer=lambda: (True, GenerateFails, None),
        env_preparer=lambda: {},
    )

    assert load_failed["summary"]["model_used"] is False
    assert "load failed" in load_failed["summary"]["blocker"]
    assert generate_failed["summary"]["model_used"] is False
    assert generate_failed["rows"][0]["blocker"] == "RuntimeError: generation failed"


def test_scenario_bench_1536_runner_writes_manifest_and_required_artifact(
    tmp_path: Path,
) -> None:
    """SCENARIO-BENCH-1536: runner writes manifest rows and required artifact fields."""

    def fake_collect(spec: dict[str, Any], prompt_cases: list[exp.PromptCase]) -> dict[str, Any]:
        return {
            "summary": {
                "hf_id": spec["hf_id"],
                "model_name": spec.get("name"),
                "model_used": True,
                "blocker": None,
            },
            "rows": [
                {
                    "case_id": case.case_id,
                    "instance_id": case.instance.instance_id,
                    "format_name": case.format_name,
                    "model_hf_id": spec["hf_id"],
                    "model_name": spec.get("name"),
                    "generation_source": "live_sota_llamacpp",
                    "output_text": exp.gold_answer_for_prompt_case(case),
                    "elapsed_seconds": 0.01,
                    "blocker": None,
                }
                for case in prompt_cases
            ],
        }

    output_path = tmp_path / "experiment_1536.json"
    manifest_path = tmp_path / "manifest.jsonl"

    artifact = exp.run_benchmark(
        output_path=output_path,
        manifest_path=manifest_path,
        run_date="20260508",
        collect_model_outputs_fn=fake_collect,
        cached_pair_fn=lambda gpu_indices=(0, 1): [
            {
                "name": "Qwen3.6-35B-A3B",
                "hf_id": exp.MANDATED_MODEL_SPECS[0]["hf_id"],
                "gpu": 0,
                "model_path": "/tmp/qwen.gguf",
            }
        ],
        gpu_probe_fn=lambda: {"gpu_count": 1},
        max_models=1,
        focused_tests_passed=True,
    )
    persisted = json.loads(output_path.read_text(encoding="utf-8"))
    rows = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines()]

    assert artifact == persisted
    assert set(exp.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert artifact["status"] == "complete"
    assert artifact["milestone"] == "20260508"
    assert artifact["satquest_benchmark_ready"] is True
    assert artifact["live_sota_model_inference_used"] is True
    assert artifact["cnf_instances"] == len(exp.build_cnf_instances())
    assert artifact["formats_tested"] == ["machine", "narrative", "symbolic"]
    assert artifact["solver_oracle_used"].startswith(("exact_exhaustive", "pysat"))
    assert artifact["solver_oracle_false_accepts"] == 0
    assert artifact["baseline_accuracy"] == pytest.approx(1.0)
    assert artifact["energy_ranked_accuracy"] == pytest.approx(1.0)
    assert artifact["repair_hint_accuracy"] == pytest.approx(1.0)
    assert artifact["false_accept_rate"] == pytest.approx(0.0)
    assert artifact["focused_tests_passed"] is True
    assert artifact["honest_verdict"].startswith("complete:")
    assert len(rows) == len(exp.build_prompt_cases())


def test_req_bench_1536_runner_blocks_without_live_sota_rows(tmp_path: Path) -> None:
    """REQ-BENCH-1536: missing live SOTA rows is explicit and non-headline."""

    def blocked_collect(spec: dict[str, Any], prompt_cases: list[exp.PromptCase]) -> dict[str, Any]:
        return {
            "summary": {
                "hf_id": spec["hf_id"],
                "model_name": spec.get("name"),
                "model_used": False,
                "blocker": "model_not_cached",
            },
            "rows": [],
        }

    artifact = exp.run_benchmark(
        output_path=tmp_path / "blocked.json",
        manifest_path=tmp_path / "blocked.jsonl",
        collect_model_outputs_fn=blocked_collect,
        cached_pair_fn=lambda gpu_indices=(0, 1): None,
        gpu_probe_fn=lambda: {"gpu_count": 0},
        max_models=1,
    )

    assert artifact["status"] == "blocked"
    assert artifact["satquest_benchmark_ready"] is False
    assert artifact["live_sota_model_inference_used"] is False
    assert artifact["honest_verdict"].startswith("complete_blocked:")
    assert artifact["blockers"] == ["model_not_cached"]
    assert exp._collect_blockers(
        [{"blocker": "not_attempted_runtime_budget"}, {"blocker": "model_not_cached"}],
        "resolver boom",
    ) == ["cached_sota_pair_error:resolver boom", "model_not_cached"]


def test_req_bench_1536_main_uses_all_models_flag(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """REQ-BENCH-1536: CLI exposes the readiness and accuracy fields."""

    seen: dict[str, int] = {}

    def fake_run(*, max_models: int, focused_tests_passed: bool) -> dict[str, Any]:
        seen["max_models"] = max_models
        assert focused_tests_passed is False
        return {
            "satquest_benchmark_ready": True,
            "cnf_instances": 6,
            "formats_tested": ["machine", "narrative", "symbolic"],
            "baseline_accuracy": 0.5,
            "energy_ranked_accuracy": 0.75,
            "repair_hint_accuracy": 0.875,
            "solver_oracle_false_accepts": 0,
        }

    monkeypatch.setenv("CARNOT_SATQUEST_1536_MAX_MODELS", "1")
    monkeypatch.setattr(exp, "run_benchmark", fake_run)

    assert exp.main(["--all-models"]) == 0
    assert seen["max_models"] == len(exp.MANDATED_MODEL_SPECS)
    assert "ready=True" in capsys.readouterr().out
