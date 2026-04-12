"""Spec: REQ-VERIFY-025, REQ-VERIFY-026, REQ-VERIFY-027,
SCENARIO-VERIFY-025, SCENARIO-VERIFY-026, SCENARIO-VERIFY-027.
"""

from __future__ import annotations

import importlib.util
import json
import runpy
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest


def load_module():
    repo_root = Path(__file__).resolve().parents[2]
    module_path = repo_root / "scripts" / "experiment_218_live_dual_model_suite.py"
    spec = importlib.util.spec_from_file_location(
        "experiment_218_live_dual_model_suite", module_path
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    return repo


def sample_records() -> list[dict[str, object]]:
    return [
        {"case_id": "case-a", "prompt": "Alpha", "task_slice": "instruction_surface_only"},
        {"case_id": "case-b", "prompt": "Beta", "task_slice": "code_typed_properties"},
        {"case_id": "case-c", "prompt": "Gamma", "task_slice": "live_gsm8k_semantic_failure"},
    ]


# REQ-VERIFY-025
def test_build_parser_and_supported_models_are_fixed():
    module = load_module()

    parser = module.build_parser()
    benchmark_action = next(action for action in parser._actions if action.dest == "benchmark")

    assert list(benchmark_action.choices) == [
        "gsm8k_semantic",
        "humaneval_property",
        "constraint_ir",
    ]
    assert [model["name"] for model in module.MODEL_SPECS] == [
        "Qwen3.5-0.8B",
        "Gemma4-E4B-it",
    ]
    assert [model["hf_id"] for model in module.MODEL_SPECS] == [
        "Qwen/Qwen3.5-0.8B",
        "google/gemma-4-E4B-it",
    ]

    args = parser.parse_args(["--benchmark", "gsm8k_semantic"])
    assert args.sample_seed == module.DEFAULT_SAMPLE_SEED
    assert args.sample_size == module.BENCHMARK_SPECS["gsm8k_semantic"]["default_sample_size"]
    assert args.output == module.default_output_path("gsm8k_semantic")


# REQ-VERIFY-025, SCENARIO-VERIFY-026
def test_build_cohort_manifest_is_deterministic_and_reuses_prompt_seed_across_modes():
    module = load_module()

    first = module.build_cohort_manifest(sample_records(), sample_size=2, sample_seed=11)
    second = module.build_cohort_manifest(sample_records(), sample_size=2, sample_seed=11)

    assert first == second
    assert [case["sample_position"] for case in first] == [1, 2]
    assert {case["case_id"] for case in first}.issubset({"case-a", "case-b", "case-c"})
    for case in first:
        prompt_seeds = case["prompt_seeds"]
        assert prompt_seeds["baseline"] == prompt_seeds["verify_only"]
        assert prompt_seeds["verify_only"] == prompt_seeds["verify_repair"]


# REQ-VERIFY-025
def test_recommended_response_mode_uses_exp213_policy_defaults():
    module = load_module()

    assert module.recommended_response_mode("live_gsm8k_semantic_failure") == "structured_json"
    assert module.recommended_response_mode("code_typed_properties") == "answer_only_terse"
    assert module.recommended_response_mode("unknown-task-slice") == "answer_only_terse"


# REQ-VERIFY-025
def test_repo_override_policy_fallback_checkpoint_edges_and_utc_now(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    module = load_module()
    repo = make_repo(tmp_path)
    monkeypatch.setenv("CARNOT_REPO_ROOT", str(repo))

    assert module.get_repo_root() == repo.resolve()
    assert module.load_monitorability_policy(repo / "results" / "missing.json") == {}

    missing = module.load_checkpoint(tmp_path / "missing.json", ["case-a"])
    assert missing == {"case_ids": ["case-a"], "results_by_case": {}}

    mismatch_path = tmp_path / "mismatch.json"
    mismatch_path.write_text(
        json.dumps({"case_ids": ["other-case"], "results_by_case": {"other-case": {}}}),
        encoding="utf-8",
    )
    mismatch = module.load_checkpoint(mismatch_path, ["case-a"])
    assert mismatch == {"case_ids": ["case-a"], "results_by_case": {}}

    invalid_results_path = tmp_path / "invalid-results.json"
    invalid_results_path.write_text(
        json.dumps({"case_ids": ["case-a"], "results_by_case": []}),
        encoding="utf-8",
    )
    invalid = module.load_checkpoint(invalid_results_path, ["case-a"])
    assert invalid == {"case_ids": ["case-a"], "results_by_case": {}}

    assert "T" in module.utc_now()


# SCENARIO-VERIFY-025
def test_run_mode_resumes_only_pending_cases_and_preserves_order(tmp_path: Path):
    module = load_module()
    checkpoint_dir = tmp_path / "checkpoints"
    cases = module.build_cohort_manifest(sample_records(), sample_size=2, sample_seed=7)
    checkpoint_path = module.checkpoint_path(
        checkpoint_dir,
        benchmark="constraint_ir",
        model_name="Qwen3.5-0.8B",
        mode="baseline",
    )

    module.save_checkpoint(
        checkpoint_path,
        {
            "benchmark": "constraint_ir",
            "model_name": "Qwen3.5-0.8B",
            "mode": "baseline",
            "case_ids": [case["case_id"] for case in cases],
            "results_by_case": {
                cases[0]["case_id"]: {
                    "case_id": cases[0]["case_id"],
                    "mode": "baseline",
                    "value": "cached",
                }
            },
        },
    )

    executed: list[str] = []

    def execute_case(case: dict[str, object]) -> dict[str, object]:
        executed.append(str(case["case_id"]))
        return {
            "case_id": case["case_id"],
            "mode": "baseline",
            "value": f"fresh-{case['case_id']}",
        }

    results = module.run_mode(
        benchmark="constraint_ir",
        model_name="Qwen3.5-0.8B",
        mode="baseline",
        cases=cases,
        checkpoint_dir=checkpoint_dir,
        execute_case=execute_case,
    )

    assert executed == [cases[1]["case_id"]]
    assert [result["case_id"] for result in results] == [case["case_id"] for case in cases]
    assert results[0]["value"] == "cached"
    assert results[1]["value"] == f"fresh-{cases[1]['case_id']}"


# REQ-VERIFY-026, SCENARIO-VERIFY-026
def test_build_artifact_payload_has_stable_schema_and_order():
    module = load_module()
    cohort = module.build_cohort_manifest(sample_records(), sample_size=2, sample_seed=5)
    paired_runs = [
        {
            "benchmark": "constraint_ir",
            "mode": "baseline",
            "model_name": "Qwen3.5-0.8B",
            "model_hf_id": "Qwen/Qwen3.5-0.8B",
            "summary": {"n_cases": 2},
            "cases": [{"case_id": cohort[0]["case_id"]}, {"case_id": cohort[1]["case_id"]}],
        },
        {
            "benchmark": "constraint_ir",
            "mode": "verify_only",
            "model_name": "Qwen3.5-0.8B",
            "model_hf_id": "Qwen/Qwen3.5-0.8B",
            "summary": {"n_cases": 2},
            "cases": [{"case_id": cohort[0]["case_id"]}, {"case_id": cohort[1]["case_id"]}],
        },
    ]

    payload = module.build_artifact_payload(
        benchmark="constraint_ir",
        output_path=Path("results/experiment_219_results.json"),
        cohort=cohort,
        paired_runs=paired_runs,
        statistics={"Qwen3.5-0.8B": {"baseline": {"n_cases": 2}}},
        sample_seed=5,
        sample_size=2,
        started_at="2026-04-12T12:00:00Z",
        finished_at="2026-04-12T12:01:00Z",
        runtime_seconds=60.0,
        checkpoint_dir=Path("results/checkpoints/experiment_218"),
        max_repairs=3,
        policy_path=Path("results/monitorability_policy_213.json"),
        inference_mode="live_gpu",
    )

    assert payload["experiment"] == 219
    assert payload["run_date"] == "20260412"
    assert payload["schema"]["artifact"] == "carnot.live_dual_model_suite.v1"
    assert payload["schema"]["benchmark_case_schema"] == "constraint_ir.v1"
    assert payload["metadata"]["output_path"] == "results/experiment_219_results.json"
    assert payload["metadata"]["checkpoint_dir"] == "results/checkpoints/experiment_218"
    assert payload["metadata"]["max_repairs"] == 3
    assert payload["metadata"]["policy_source"] == "results/monitorability_policy_213.json"
    assert payload["metadata"]["inference_mode"] == "live_gpu"
    assert payload["cohort"]["case_ids"] == [case["case_id"] for case in cohort]
    assert payload["paired_runs"] == paired_runs
    assert (
        payload["cohort"]["cases"][0]["prompt_seeds"]["baseline"]
        == payload["cohort"]["cases"][0]["prompt_seeds"]["verify_repair"]
    )


# REQ-VERIFY-027
def test_artifact_experiment_id_follows_output_filename():
    module = load_module()

    assert module.artifact_experiment_id(Path("results/experiment_219_results.json")) == 219
    assert module.artifact_experiment_id(Path("results/experiment_220_results.json")) == 220
    assert module.artifact_experiment_id(Path("results/custom.json")) == 218


# REQ-VERIFY-027
def test_extract_final_number_ignores_punctuation_only_matches():
    module = load_module()

    assert module._extract_final_number("Result: ,,,") is None
    assert module._extract_final_number("Answer: -12") == -12


# REQ-VERIFY-027, SCENARIO-VERIFY-027
def test_summarize_gsm8k_runs_reports_semantic_metrics_parse_coverage_and_overhead():
    module = load_module()

    baseline_runs = [
        {
            "correct": True,
            "latency_seconds": 1.2,
            "prompt_tokens": 18,
            "response_tokens": 7,
            "total_tokens": 25,
            "typed_reasoning_parse_status": "direct_json",
        },
        {
            "correct": False,
            "latency_seconds": 1.8,
            "prompt_tokens": 20,
            "response_tokens": 9,
            "total_tokens": 29,
            "typed_reasoning_parse_status": "fallback_text",
        },
    ]
    verify_only_runs = [
        {
            "accepted_correct": True,
            "flagged": False,
            "correct": True,
            "semantic_violation_count": 0,
            "typed_reasoning_parse_status": "direct_json",
            "latency_seconds": 0.4,
            "total_tokens": 0,
        },
        {
            "accepted_correct": False,
            "flagged": True,
            "correct": False,
            "semantic_violation_count": 2,
            "typed_reasoning_parse_status": "unavailable",
            "latency_seconds": 0.6,
            "total_tokens": 0,
        },
    ]
    verify_repair_runs = [
        {
            "initial_correct": True,
            "correct": True,
            "repaired": False,
            "n_repairs": 0,
            "latency_seconds": 0.0,
            "total_tokens": 0,
        },
        {
            "initial_correct": False,
            "correct": True,
            "repaired": True,
            "n_repairs": 2,
            "latency_seconds": 2.5,
            "total_tokens": 34,
        },
    ]

    summary = module._summarize_runs(
        "gsm8k_semantic",
        baseline_runs,
        verify_only_runs,
        verify_repair_runs,
    )

    assert summary["baseline"]["accuracy"] == 0.5
    assert summary["baseline"]["mean_total_tokens"] == 27.0
    assert summary["verify_only"]["accuracy"] == 0.5
    assert summary["verify_only"]["n_wrong_answers"] == 1
    assert summary["verify_only"]["n_wrong_detected"] == 1
    assert summary["verify_only"]["semantic_violation_count"] == 2
    assert summary["verify_only"]["parse_coverage"] == 0.5
    assert summary["verify_only"]["false_positives"] == 0
    assert summary["verify_only"]["mean_additional_latency_seconds"] == 0.5
    assert summary["verify_repair"]["accuracy"] == 1.0
    assert summary["verify_repair"]["n_repaired"] == 1
    assert summary["verify_repair"]["repair_yield"] == 1.0
    assert summary["verify_repair"]["mean_additional_tokens"] == 17.0
    assert summary["paired_deltas"]["repair_minus_baseline"] == 0.5
    assert summary["paired_deltas"]["verify_only_minus_baseline"] == 0.0


@dataclass
class _FakeSerializable:
    payload: dict[str, object]

    def to_dict(self) -> dict[str, object]:
        return dict(self.payload)


@dataclass
class _FakeDataclassPayload:
    label: str


# REQ-VERIFY-027, SCENARIO-VERIFY-027
def test_serialize_verification_result_preserves_semantic_trace_artifacts():
    module = load_module()

    verification = SimpleNamespace(
        verified=False,
        energy=1.25,
        constraints=[],
        violations=[],
        certificate={"n_constraints": 2, "n_violations": 1},
        typed_reasoning=_FakeSerializable(
            {
                "provenance": {
                    "extraction_method": "direct_json",
                    "parser_version": "20260412",
                }
            }
        ),
        semantic_grounding=_FakeSerializable(
            {
                "violations": [
                    {
                        "violation_type": "answer_target_mismatch",
                        "description": "Answered the wrong target.",
                    }
                ]
            }
        ),
    )

    serialized = module._serialize_verification_result(verification)

    assert serialized["verified"] is False
    assert serialized["energy"] == 1.25
    assert serialized["certificate"] == {"n_constraints": 2, "n_violations": 1}
    assert serialized["typed_reasoning"]["provenance"]["extraction_method"] == "direct_json"
    assert serialized["semantic_grounding"]["violations"][0]["violation_type"] == (
        "answer_target_mismatch"
    )


# REQ-VERIFY-027
def test_helper_serializers_cover_env_token_and_generation_branches(
    monkeypatch: pytest.MonkeyPatch,
):
    module = load_module()

    monkeypatch.delenv("CARNOT_FORCE_LIVE", raising=False)
    monkeypatch.delenv("CARNOT_FORCE_CPU", raising=False)
    assert module.live_inference_mode() == "simulated"

    monkeypatch.setenv("CARNOT_FORCE_LIVE", "1")
    monkeypatch.setenv("CARNOT_FORCE_CPU", "0")
    assert module.live_inference_mode() == "live_gpu"

    monkeypatch.setenv("CARNOT_FORCE_CPU", "1")
    assert module.live_inference_mode() == "live_cpu"

    class FakeTokenizer:
        def __call__(self, text: str, **_: object) -> dict[str, list[int]]:
            return {"input_ids": list(range(len(text.split())))}

    class RaisingTokenizer:
        def __call__(self, text: str, **_: object) -> dict[str, list[int]]:
            raise RuntimeError(text)

    assert module._token_count(None, "") == 0
    assert module._token_count(None, "alpha beta") == 2
    assert module._token_count(FakeTokenizer(), "alpha beta gamma") == 3
    assert module._token_count(RaisingTokenizer(), "alpha beta") == 2
    assert module._round_mean([]) == 0.0
    assert module._round_mean([1.0, 2.0]) == 1.5

    typed_with_attr = SimpleNamespace(provenance=SimpleNamespace(extraction_method="direct_json"))
    assert module._typed_reasoning_parse_status(None) == "unavailable"
    assert module._typed_reasoning_parse_status(typed_with_attr) == "direct_json"
    assert module._typed_reasoning_parse_status(_FakeSerializable({})) == "parsed"

    assert module._serialize_jsonable(("alpha", 2)) == ["alpha", 2]
    assert module._serialize_jsonable(_FakeDataclassPayload(label="beta")) == {"label": "beta"}
    assert module._serialize_jsonable(object()).startswith("<object object at ")
    assert module._serialize_constraint_result({"kind": "cached"}) == {"kind": "cached"}
    constraint = SimpleNamespace(
        constraint_type="semantic_grounding",
        description="needs review",
        metadata={"satisfied": False},
    )
    assert module._serialize_constraint_result(constraint) == {
        "constraint_type": "semantic_grounding",
        "description": "needs review",
        "metadata": {"satisfied": False},
    }

    trace = module._build_generation_trace(
        tokenizer=FakeTokenizer(),
        attempts=[
            module._serialize_generation_attempt(
                prompt="prompt one",
                response="answer one",
                tokenizer=FakeTokenizer(),
                valid=True,
                error="schema",
            )
        ],
        fallback_record={"prompt": "retry prompt", "response": "retry answer"},
    )

    assert trace["fallback_used"] is True
    assert trace["attempts"][0]["valid"] is True
    assert trace["attempts"][0]["error"] == "schema"
    assert trace["total_tokens"] == 8


# REQ-VERIFY-026
def test_write_artifact_creates_parent_directories_and_trailing_newline(tmp_path: Path):
    module = load_module()
    output_path = tmp_path / "nested" / "artifact.json"
    payload = {"experiment": 218}

    module.write_artifact(output_path, payload)

    assert output_path.exists()
    assert output_path.read_text(encoding="utf-8").endswith("\n")
    assert json.loads(output_path.read_text(encoding="utf-8")) == payload


# REQ-VERIFY-025
def test_cli_help_entrypoint_exits_zero(monkeypatch: pytest.MonkeyPatch):
    module_path = (
        Path(__file__).resolve().parents[2] / "scripts" / "experiment_218_live_dual_model_suite.py"
    )
    monkeypatch.setattr("sys.argv", [str(module_path), "--help"])

    with pytest.raises(SystemExit) as excinfo:
        runpy.run_path(str(module_path), run_name="__main__")

    assert excinfo.value.code == 0


# REQ-VERIFY-026
def test_main_writes_payload_and_returns_zero(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
):
    module = load_module()
    repo = make_repo(tmp_path)
    output_path = repo / "results" / "exp218.json"
    captured: dict[str, object] = {}

    monkeypatch.setenv("CARNOT_REPO_ROOT", str(repo))
    monkeypatch.setattr(module, "_run_live_benchmark", lambda args: {"experiment": 218})
    monkeypatch.setattr(
        module,
        "write_artifact",
        lambda path, payload: captured.update({"path": path, "payload": payload}),
    )

    result = module.main(["--benchmark", "gsm8k_semantic", "--output", str(output_path)])

    assert result == 0
    assert captured["path"] == output_path
    assert captured["payload"] == {"experiment": 218}
    assert f"Saved {output_path}" in capsys.readouterr().out
