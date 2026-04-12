"""Spec: REQ-VERIFY-025, REQ-VERIFY-026, SCENARIO-VERIFY-025, SCENARIO-VERIFY-026."""

from __future__ import annotations

import importlib.util
import json
import runpy
from pathlib import Path

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
        output_path=Path("results/experiment_221_results.json"),
        cohort=cohort,
        paired_runs=paired_runs,
        statistics={"Qwen3.5-0.8B": {"baseline": {"n_cases": 2}}},
        sample_seed=5,
        sample_size=2,
        started_at="2026-04-12T12:00:00Z",
        finished_at="2026-04-12T12:01:00Z",
        runtime_seconds=60.0,
    )

    assert payload["experiment"] == 218
    assert payload["run_date"] == "20260412"
    assert payload["schema"]["artifact"] == "carnot.live_dual_model_suite.v1"
    assert payload["schema"]["benchmark_case_schema"] == "constraint_ir.v1"
    assert payload["metadata"]["output_path"] == "results/experiment_221_results.json"
    assert payload["cohort"]["case_ids"] == [case["case_id"] for case in cohort]
    assert payload["paired_runs"] == paired_runs
    assert (
        payload["cohort"]["cases"][0]["prompt_seeds"]["baseline"]
        == payload["cohort"]["cases"][0]["prompt_seeds"]["verify_repair"]
    )


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
