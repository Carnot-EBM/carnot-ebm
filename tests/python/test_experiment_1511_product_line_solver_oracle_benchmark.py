"""Tests for Exp 1511 product-line solver oracle benchmark.

Spec: REQ-BENCH-1511, SCENARIO-BENCH-1511.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from carnot.eval import product_line_solver_oracle_benchmark as exp


def test_req_bench_1511_parses_feature_model_blueprints() -> None:
    """REQ-BENCH-1511: semi-formal blueprints parse into product-line constraints."""

    case = exp.build_feature_model_cases()[0]
    parsed = exp.parse_blueprint(case.blueprint_text)

    assert parsed.case_id == case.case_id
    assert parsed.model.model_id == "RetailCheckout"
    assert "Checkout" in parsed.model.mandatory
    assert "Coupons" in parsed.model.optional
    assert ("Coupons", "Loyalty") in parsed.model.requires
    assert ("CryptoPay", "Coupons") in parsed.model.excludes
    assert parsed.operation.kind == "max_value"
    assert parsed.operation.budget == 12
    assert "Store" in parsed.operation.include
    assert case.prompt
    assert "selected_features" in case.prompt


def test_req_bench_1511_blueprint_parser_rejects_malformed_inputs() -> None:
    """REQ-BENCH-1511: malformed feature models fail closed before oracle use."""

    with pytest.raises(ValueError, match="missing CASE"):
        exp.parse_blueprint("MODEL: Broken\n")
    with pytest.raises(ValueError, match="missing MODEL"):
        exp.parse_blueprint(
            "\nCASE: c\nFEATURES:\n- Root mandatory cost=0 value=0\nOPERATION: min_cost include=Root\n"
        )
    with pytest.raises(ValueError, match="feature before FEATURES"):
        exp.parse_blueprint("CASE: c\nMODEL: M\n- Root mandatory cost=0 value=0")
    with pytest.raises(ValueError, match="unknown feature kind"):
        exp.parse_blueprint(
            "CASE: c\nMODEL: M\nFEATURES:\n- Root sometimes cost=0 value=0\n"
        )
    with pytest.raises(ValueError, match="unsupported operation"):
        exp.parse_blueprint(
            "CASE: c\nMODEL: M\nFEATURES:\n"
            "- Root mandatory cost=0 value=0\nOPERATION: count include=Root\n"
        )


def test_req_bench_1511_oracle_checks_feasibility_and_optimality() -> None:
    """REQ-BENCH-1511: exhaustive oracle separates feasible and optimal answers."""

    case = exp.build_feature_model_cases()[0]
    oracle = exp.solve_case(case)
    optimal = set(oracle.optimal_features)
    infeasible = {"Store", "Catalog", "Checkout", "Coupons"}
    feasible_but_suboptimal = set(case.model.mandatory)

    assert oracle.feasible_exists is True
    assert oracle.feasible_count > 0
    assert exp.is_selection_feasible(case.model, optimal).ok is True
    assert exp.selection_satisfies_operation(case, optimal).ok is True
    assert exp.evaluate_selection(case, optimal).classification == "oracle_agreement"

    impossible = exp.ProductLineCase(
        case_id="impossible",
        model=case.model,
        operation=exp.AnalysisOperation("min_cost", frozenset({"NotAFeature"})),
        blueprint_text=case.blueprint_text,
        prompt="",
    )
    assert exp.solve_case(impossible).feasible_exists is False

    infeasible_result = exp.evaluate_selection(case, infeasible)
    assert infeasible_result.classification == "infeasible"
    assert "requires:Coupons->Loyalty" in infeasible_result.reasons

    unknown_result = exp.is_selection_feasible(case.model, {"Store", "Bogus"})
    assert "unknown:Bogus" in unknown_result.reasons
    missing_result = exp.is_selection_feasible(case.model, {"Store"})
    assert "missing_mandatory:Catalog,Checkout" in missing_result.reasons

    suboptimal_result = exp.evaluate_selection(case, feasible_but_suboptimal)
    assert suboptimal_result.classification == "wrong_or_suboptimal"
    assert suboptimal_result.feasible is True
    assert suboptimal_result.oracle_agrees is False


def test_req_bench_1511_output_parser_handles_wrappers_and_rejects_junk() -> None:
    """REQ-BENCH-1511: model JSON extraction is tolerant but schema validation is strict."""

    case = exp.build_feature_model_cases()[1]
    answer = exp.compliant_answer_for_case(case)
    wrapped = f"<think>solved</think>\n```json\n{answer}\n```"

    parsed = exp.parse_model_answer(wrapped)
    assert parsed.parse_ok is True
    assert parsed.selected_features == tuple(exp.solve_case(case).optimal_features)
    assert parsed.model_declared_accept is True

    missing = exp.parse_model_answer("no JSON here")
    assert missing.parse_ok is False
    assert missing.parse_error == "no_json_object"

    malformed = exp.parse_model_answer('{"selected_features": "Store"}')
    assert malformed.parse_ok is False
    assert malformed.parse_error == "selected_features_not_list"

    bad_numeric = exp.parse_model_answer(
        '{"selected_features": ["Drone"], "objective_cost": "n/a", "objective_value": true}'
    )
    assert bad_numeric.objective_cost is None
    assert bad_numeric.objective_value is None


def test_scenario_bench_1511_evaluator_classifies_failures_and_false_accepts() -> None:
    """SCENARIO-BENCH-1511: parsing, feasibility, suboptimality, and false accepts are distinct."""

    case = exp.build_feature_model_cases()[2]
    correct_row = exp.build_manifest_row(
        case,
        {
            "model_hf_id": exp.MANDATED_MODEL_SPECS[0]["hf_id"],
            "model_name": "Qwen3.6-35B-A3B",
            "generation_source": "live_sota_llamacpp",
            "output_text": exp.compliant_answer_for_case(case),
            "elapsed_seconds": 0.01,
            "blocker": None,
        },
    )

    infeasible_payload = json.loads(exp.compliant_answer_for_case(case))
    infeasible_payload["selected_features"] = ["Clinic", "Portal", "Scheduling", "Payments"]
    infeasible_payload["verifier"] = {"accept": True}
    infeasible_row = exp.build_manifest_row(
        case,
        {
            "model_hf_id": exp.MANDATED_MODEL_SPECS[0]["hf_id"],
            "model_name": "Qwen3.6-35B-A3B",
            "generation_source": "live_sota_llamacpp",
            "output_text": json.dumps(infeasible_payload),
            "elapsed_seconds": 0.01,
            "blocker": None,
        },
    )

    suboptimal_payload = json.loads(exp.compliant_answer_for_case(case))
    suboptimal_payload["selected_features"] = sorted(
        case.model.mandatory | case.operation.include | {"SMS"}
    )
    suboptimal_payload["verifier"] = {"accept": False}
    suboptimal_row = exp.build_manifest_row(
        case,
        {
            "model_hf_id": exp.MANDATED_MODEL_SPECS[0]["hf_id"],
            "model_name": "Qwen3.6-35B-A3B",
            "generation_source": "live_sota_llamacpp",
            "output_text": json.dumps(suboptimal_payload),
            "elapsed_seconds": 0.01,
            "blocker": None,
        },
    )

    parse_row = exp.build_manifest_row(
        case,
        {
            "model_hf_id": exp.MANDATED_MODEL_SPECS[0]["hf_id"],
            "model_name": "Qwen3.6-35B-A3B",
            "generation_source": "live_sota_llamacpp",
            "output_text": "not-json",
            "elapsed_seconds": 0.01,
            "blocker": None,
        },
    )

    assert correct_row["oracle_result"]["classification"] == "oracle_agreement"
    assert correct_row["verifier_result"]["accepted"] is True
    assert infeasible_row["oracle_result"]["classification"] == "infeasible"
    assert infeasible_row["verifier_result"]["self_verifier_false_accept"] is True
    assert suboptimal_row["oracle_result"]["classification"] == "wrong_or_suboptimal"
    assert suboptimal_row["verifier_result"]["self_verifier_false_accept"] is False
    assert parse_row["oracle_result"]["classification"] == "parse_failure"

    metrics = exp.aggregate_manifest_metrics(
        [correct_row, infeasible_row, suboptimal_row, parse_row]
    )
    assert metrics["parse_rate"] == pytest.approx(0.75)
    assert metrics["feasibility_rate"] == pytest.approx(2 / 3)
    assert metrics["oracle_agreement_rate"] == pytest.approx(0.25)
    assert metrics["verifier_false_accept_rate"] == pytest.approx(1 / 3)


def test_req_bench_1511_live_collector_has_injectable_runtime_hooks() -> None:
    """REQ-BENCH-1511: live SOTA collection can be tested without loading a GGUF."""

    class FakeLlama:
        prompts: list[str] = []
        closed = False

        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs

        def __call__(self, prompt: str, **kwargs: Any) -> dict[str, Any]:
            self.prompts.append(prompt)
            case = exp.build_feature_model_cases()[len(self.prompts) - 1]
            return {"choices": [{"text": exp.compliant_answer_for_case(case)}]}

        def close(self) -> None:
            self.closed = True

    cases = exp.build_feature_model_cases()[:2]
    spec = {"hf_id": exp.MANDATED_MODEL_SPECS[0]["hf_id"], "gpu": 0}

    ok = exp.collect_live_model_outputs(
        spec,
        cases,
        resolver=lambda _hf_id: "/tmp/fake.gguf",
        llama_importer=lambda: (True, FakeLlama, None),
        env_preparer=lambda: {},
    )
    missing = exp.collect_live_model_outputs(
        spec,
        cases,
        resolver=lambda _hf_id: None,
        llama_importer=lambda: (True, FakeLlama, None),
        env_preparer=lambda: {},
    )
    import_failed = exp.collect_live_model_outputs(
        {**spec, "model_path": "/tmp/fake.gguf"},
        cases,
        llama_importer=lambda: (False, None, "llama_cpp missing"),
        env_preparer=lambda: {},
    )

    assert ok["summary"]["model_used"] is True
    assert len(ok["rows"]) == 2
    assert ok["rows"][0]["generation_source"] == "live_sota_llamacpp"
    assert FakeLlama.prompts == [case.prompt for case in cases]
    assert missing["summary"]["blocker"] == "model_not_cached"
    assert import_failed["summary"]["blocker"] == "llama_cpp missing"


def test_req_bench_1511_live_collector_reports_load_and_generation_failures() -> None:
    """REQ-BENCH-1511: collector records load errors and per-case generation blockers."""

    class LoadFails:
        def __init__(self, **_kwargs: Any) -> None:
            raise RuntimeError("load failed")

    class GenerateFails:
        def __init__(self, **_kwargs: Any) -> None:
            pass

        def __call__(self, _prompt: str, **_kwargs: Any) -> dict[str, Any]:
            raise RuntimeError("generation failed")

    cases = exp.build_feature_model_cases()[:1]
    spec = {"hf_id": exp.MANDATED_MODEL_SPECS[0]["hf_id"], "model_path": "/tmp/fake.gguf"}

    load_failed = exp.collect_live_model_outputs(
        spec,
        cases,
        llama_importer=lambda: (True, LoadFails, None),
        env_preparer=lambda: {},
    )
    generate_failed = exp.collect_live_model_outputs(
        spec,
        cases,
        llama_importer=lambda: (True, GenerateFails, None),
        env_preparer=lambda: {},
    )

    assert load_failed["summary"]["model_used"] is False
    assert "load failed" in load_failed["summary"]["blocker"]
    assert generate_failed["summary"]["model_used"] is False
    assert generate_failed["rows"][0]["blocker"] == "RuntimeError: generation failed"


def test_scenario_bench_1511_runner_writes_manifest_and_artifact(tmp_path: Path) -> None:
    """SCENARIO-BENCH-1511: runner persists JSONL rows and required artifact fields."""

    def fake_collect(spec: dict[str, Any], cases: list[exp.ProductLineCase]) -> dict[str, Any]:
        rows: list[dict[str, Any]] = []
        for case in cases:
            rows.append(
                {
                    "case_id": case.case_id,
                    "model_hf_id": spec["hf_id"],
                    "model_name": spec.get("name"),
                    "generation_source": "live_sota_llamacpp",
                    "output_text": exp.compliant_answer_for_case(case),
                    "elapsed_seconds": 0.01,
                    "blocker": None,
                }
            )
        rows.append(
            {
                "case_id": "unknown",
                "model_hf_id": spec["hf_id"],
                "model_name": spec.get("name"),
                "generation_source": "live_sota_llamacpp",
                "output_text": "{}",
                "elapsed_seconds": 0.01,
                "blocker": None,
            }
        )
        return {
            "summary": {
                "hf_id": spec["hf_id"],
                "model_name": spec.get("name"),
                "model_used": True,
                "blocker": None,
            },
            "rows": rows,
        }

    output_path = tmp_path / "experiment_1511.json"
    manifest_path = tmp_path / "manifest.jsonl"

    artifact = exp.run_benchmark(
        output_path=output_path,
        manifest_path=manifest_path,
        run_date="20260508",
        collect_model_outputs_fn=fake_collect,
        cached_pair_fn=lambda gpu_indices=(0, 1): [  # noqa: ARG005
            {
                "name": "Qwen3.6-35B-A3B",
                "hf_id": exp.MANDATED_MODEL_SPECS[0]["hf_id"],
                "gpu": 0,
                "model_path": "/tmp/qwen.gguf",
            }
        ],
        gpu_probe_fn=lambda: {"gpu_count": 1},
        max_models=1,
        tests_run=["focused pytest"],
    )
    persisted = json.loads(output_path.read_text(encoding="utf-8"))
    rows = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines()]

    assert artifact == persisted
    assert set(exp.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert artifact["status"] == "complete"
    assert artifact["product_line_benchmark_ready"] is True
    assert artifact["live_sota_model_inference_used"] is True
    assert artifact["solver_oracle_ready"] is True
    assert artifact["feature_models_defined"] == len(exp.build_feature_model_cases())
    assert artifact["cases_attempted"] == len(exp.build_feature_model_cases())
    assert artifact["parse_rate"] == pytest.approx(1.0)
    assert artifact["feasibility_rate"] == pytest.approx(1.0)
    assert artifact["oracle_agreement_rate"] == pytest.approx(1.0)
    assert artifact["verifier_false_accept_rate"] == pytest.approx(0.0)
    assert artifact["models_used"] == [exp.MANDATED_MODEL_SPECS[0]["hf_id"]]
    assert artifact["benchmark_manifest_path"].endswith("manifest.jsonl")
    assert artifact["honest_verdict"].startswith("complete:")
    assert len(rows) == len(exp.build_feature_model_cases())
    assert rows[0]["oracle_result"]["classification"] == "oracle_agreement"
    assert artifact["model_attempts"][1]["blocker"] == "not_attempted_runtime_budget"


def test_req_bench_1511_runner_blocks_without_live_sota_rows(tmp_path: Path) -> None:
    """REQ-BENCH-1511: missing mandated live inference produces an honest blocker."""

    def blocked_collect(spec: dict[str, Any], cases: list[exp.ProductLineCase]) -> dict[str, Any]:
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
        cached_pair_fn=lambda gpu_indices=(0, 1): None,  # noqa: ARG005
        gpu_probe_fn=lambda: {"gpu_count": 0},
        max_models=1,
    )

    assert artifact["status"] == "blocked"
    assert artifact["product_line_benchmark_ready"] is False
    assert artifact["live_sota_model_inference_used"] is False
    assert artifact["honest_verdict"].startswith("complete_blocked:")
    assert artifact["blockers"] == ["model_not_cached"]


def test_req_bench_1511_main_uses_all_models_flag(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """REQ-BENCH-1511: CLI exposes the artifact readiness metrics."""

    seen: dict[str, int] = {}

    def fake_run(*, max_models: int) -> dict[str, Any]:
        seen["max_models"] = max_models
        return {
            "product_line_benchmark_ready": True,
            "feature_models_defined": 6,
            "models_used": ["m"],
            "parse_rate": 0.5,
            "feasibility_rate": 0.25,
            "verifier_false_accept_rate": 0.0,
        }

    monkeypatch.setenv("CARNOT_PRODUCT_LINE_1511_MAX_MODELS", "1")
    monkeypatch.setattr(exp, "run_benchmark", fake_run)

    assert exp.main(["--all-models"]) == 0
    assert seen["max_models"] == len(exp.MANDATED_MODEL_SPECS)
    assert "ready=True" in capsys.readouterr().out
