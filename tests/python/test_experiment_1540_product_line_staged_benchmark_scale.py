"""Tests for Exp1540 product-line staged benchmark scale-up.

Spec: REQ-BENCH-1540, SCENARIO-BENCH-1540.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from carnot.eval import product_line_solver_oracle_benchmark as exp1511
from carnot.eval import product_line_staged_benchmark_scale as exp


def test_req_bench_1540_builds_reproducible_scaled_case_pack() -> None:
    """REQ-BENCH-1540: the staged pack reaches the 40-case scale gate."""

    cases = exp.build_staged_product_line_cases(target_count=40)
    repeated = exp.build_staged_product_line_cases(target_count=40)

    assert len(cases) >= 40
    assert [case.case_id for case in cases] == [case.case_id for case in repeated]
    assert len({case.case_id for case in cases}) == len(cases)
    assert all(exp1511.solve_case(case).feasible_exists for case in cases)
    assert exp.oracle_label_snapshot(cases[:8]) == exp.oracle_label_snapshot(repeated[:8])
    assert all(case.case_id in case.prompt for case in cases[:8])


def test_scenario_bench_1540_staged_feedback_rows_preserve_oracle_labels() -> None:
    """SCENARIO-BENCH-1540: syntax, feature, solver, and policy stages are audited."""

    cases = exp.build_staged_product_line_cases(target_count=8)
    seed_rows = exp.build_staged_seed_rows(cases)
    rows = [
        exp.evaluate_staged_case(case, seed_row)
        for case, seed_row in zip(cases, seed_rows, strict=True)
    ]
    metrics = exp.summarize_staged_rows(rows)

    assert {row["seed_mode"] for row in rows} >= {
        "syntax_failure",
        "feature_model_repair",
        "solver_repair",
        "automata_guided_oracle",
    }
    assert all(
        [stage["stage"] for stage in row["stages"]]
        == [
            "syntax_parse_feedback",
            "feature_model_consistency_feedback",
            "solver_feasibility_feedback",
            "policy_compliance_feedback",
        ]
        for row in rows
    )
    assert all(row["oracle_label"] == exp.oracle_label_for_case(case) for row, case in zip(rows, cases, strict=True))
    assert all(row["policy_result"]["accepted"] is True for row in rows)
    assert all(row["policy_result"]["false_accept"] is False for row in rows)
    assert metrics["syntax_stage_pass_rate"] == pytest.approx(1.0)
    assert metrics["feature_model_stage_pass_rate"] == pytest.approx(1.0)
    assert metrics["feasibility_stage_pass_rate"] == pytest.approx(1.0)
    assert metrics["oracle_agreement_rate"] == pytest.approx(1.0)
    assert metrics["false_accept_rate"] == pytest.approx(0.0)


def test_req_bench_1540_automata_json_is_below_oracle_authority() -> None:
    """REQ-BENCH-1540: automata-constrained JSON parses but oracle still labels it."""

    case = exp.build_staged_product_line_cases(target_count=1)[0]
    dfa = exp.compile_product_line_answer_dfa(case)
    generated = dfa.generate()
    parsed = exp1511.parse_model_answer(generated)
    row = exp.evaluate_staged_case(
        case,
        {
            "seed_mode": "automata_guided_oracle",
            "model_output": generated,
            "generation_source": "automata_guided_abs_dfa",
            "model_hf_id": exp.MANDATED_MODEL_SPECS[0],
            "model_name": "Qwen3.6-35B-A3B",
        },
    )

    assert dfa.accepts(generated) is True
    assert parsed.parse_ok is True
    assert row["automata_constraints_used"] is True
    assert row["oracle_result"]["oracle_agrees"] is True
    assert row["policy_result"]["rule"] == "accept iff deterministic oracle_agrees"


def test_req_bench_1540_readiness_gate_retires_false_accepts() -> None:
    """REQ-BENCH-1540: any final false accept retires the branch."""

    cases = exp.build_staged_product_line_cases(target_count=4)
    rows = [
        exp.evaluate_staged_case(case, seed_row)
        for case, seed_row in zip(cases, exp.build_staged_seed_rows(cases), strict=True)
    ]
    rows[0]["policy_result"]["false_accept"] = True
    metrics = exp.summarize_staged_rows(rows)
    decision = exp.decide_scale_readiness(
        cases_total=40,
        metrics=metrics,
        live_sota_model_inference_used=True,
        focused_tests_passed=True,
        blockers=[],
    )

    assert metrics["false_accept_rate"] > 0.0
    assert decision["product_line_scale_ready"] is False
    assert decision["branch_retired"] is True
    assert "false_accept_rate" in decision["retirement_reason"]


def test_req_bench_1540_edge_gates_report_blockers(tmp_path: Path) -> None:
    """REQ-BENCH-1540: empty packs and blocked live provenance fail honestly."""

    good_metrics = {
        "syntax_stage_pass_rate": 1.0,
        "feature_model_stage_pass_rate": 1.0,
        "feasibility_stage_pass_rate": 1.0,
        "oracle_agreement_rate": 1.0,
        "false_accept_rate": 0.0,
    }
    stage_failed = {**good_metrics, "oracle_agreement_rate": 0.5}

    assert exp.build_staged_product_line_cases(target_count=0) == []
    assert exp.summarize_staged_rows([])["false_accept_rate"] == pytest.approx(0.0)
    assert "scaled corpus below" in exp.decide_scale_readiness(
        cases_total=3,
        metrics=good_metrics,
        live_sota_model_inference_used=True,
        focused_tests_passed=True,
        blockers=[],
    )["retirement_reason"]
    assert "mandated live SOTA" in exp.decide_scale_readiness(
        cases_total=40,
        metrics=good_metrics,
        live_sota_model_inference_used=False,
        focused_tests_passed=True,
        blockers=[],
    )["retirement_reason"]
    assert "staged validators failed" in exp.decide_scale_readiness(
        cases_total=40,
        metrics=stage_failed,
        live_sota_model_inference_used=True,
        focused_tests_passed=True,
        blockers=[],
    )["retirement_reason"]
    assert exp.decide_scale_readiness(
        cases_total=40,
        metrics=good_metrics,
        live_sota_model_inference_used=True,
        focused_tests_passed=True,
        blockers=["manual_blocker"],
    )["retirement_reason"] == "manual_blocker"
    assert exp._honest_verdict(False, False, "") == (  # noqa: SLF001
        "complete_blocked: product-line staged benchmark incomplete"
    )

    cases = exp.build_staged_product_line_cases(target_count=2)
    seed_rows = exp.build_staged_seed_rows(cases)
    merged = exp._merge_live_rows(  # noqa: SLF001 - covered to exercise provenance merge gates.
        cases,
        seed_rows,
        [
            {"case_id": cases[0].case_id, "blocker": "generation_failed"},
            {"case_id": "unknown", "blocker": None},
        ],
    )
    assert merged == seed_rows

    def cache_error(**_kwargs: Any) -> None:
        raise RuntimeError("cache probe failed")

    blocked = exp.run_benchmark(
        output_path=tmp_path / "blocked.json",
        manifest_path=tmp_path / "blocked.jsonl",
        target_count=4,
        focused_tests_passed=False,
        cached_pair_fn=cache_error,
        gpu_probe_fn=lambda: {"gpu_count": 0},
    )

    assert blocked["branch_retired"] is True
    assert blocked["focused_tests_passed"] is False
    assert "focused_tests_not_passed" in blocked["blockers"]
    assert any(str(blocker).startswith("cached_sota_pair_error:") for blocker in blocked["blockers"])


def test_scenario_bench_1540_runner_writes_manifest_and_artifact(tmp_path: Path) -> None:
    """SCENARIO-BENCH-1540: runner writes required fields with live SOTA provenance."""

    output_path = tmp_path / "experiment_1540.json"
    manifest_path = tmp_path / "product_line_staged_1540.jsonl"

    def fake_live_collector(
        cases: list[exp1511.ProductLineCase],
        model_spec: dict[str, Any],
        prompt_limit: int,
    ) -> dict[str, Any]:
        return {
            "models_used": [model_spec["hf_id"]],
            "rows": [
                {
                    "case_id": case.case_id,
                    "model_hf_id": model_spec["hf_id"],
                    "model_name": model_spec["name"],
                    "generation_source": "live_sota_llamacpp",
                    "model_output": exp1511.compliant_answer_for_case(case),
                    "elapsed_seconds": 0.01,
                    "blocker": None,
                }
                for case in cases[:prompt_limit]
            ],
            "blockers": [],
        }

    artifact = exp.run_benchmark(
        output_path=output_path,
        manifest_path=manifest_path,
        target_count=40,
        focused_tests_passed=True,
        live_prompt_limit=2,
        live_collector_fn=fake_live_collector,
        cached_pair_fn=lambda **_: [
            {
                "name": "Qwen3.6-35B-A3B",
                "hf_id": exp.MANDATED_MODEL_SPECS[0],
                "gpu": 0,
                "model_path": "/models/qwen.gguf",
            }
        ],
        gpu_probe_fn=lambda: {"cuda_available": True, "gpu_count": 1},
    )
    persisted = json.loads(output_path.read_text(encoding="utf-8"))
    rows = [
        json.loads(line)
        for line in manifest_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    assert artifact == persisted
    assert set(exp.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert artifact["status"] == "complete"
    assert artifact["product_line_scale_ready"] is True
    assert artifact["branch_retired"] is False
    assert artifact["live_sota_model_inference_used"] is True
    assert artifact["cases_total"] >= 40
    assert artifact["oracle_agreement_rate"] == pytest.approx(1.0)
    assert artifact["false_accept_rate"] == pytest.approx(0.0)
    assert artifact["focused_tests_passed"] is True
    assert artifact["honest_verdict"].startswith("complete:")
    assert len(rows) == artifact["cases_total"]
    assert rows[0]["case_id"].startswith("plc-1540-")
