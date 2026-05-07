"""Tests for Exp 1487 V_1 pairwise self-verification.

Spec: REQ-VERIFY-1487, SCENARIO-VERIFY-1487.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from carnot.eval import cctu_executable_constraint_microbenchmark as cctu
from carnot.eval import v1_pairwise_self_verification_vs_energy as exp


def _manifest_row(
    case: cctu.BenchmarkCase,
    output_text: str,
    *,
    hf_id: str = "unsloth/Qwen3.6-35B-A3B-GGUF",
) -> dict[str, Any]:
    validation = cctu.validate_transcript(case, output_text)
    return {
        "case_id": case.case_id,
        "family": case.family,
        "prompt": case.prompt,
        "model_hf_id": hf_id,
        "model_name": "Qwen3.6-35B-A3B",
        "generation_source": "live_sota_llamacpp",
        "blocker": None,
        "model_output": output_text,
        "validator_result": validation["validator_result"],
        "verifier_result": validation["verifier_result"],
    }


def _write_exp1486_inputs(
    tmp_path: Path,
    rows: list[dict[str, Any]],
) -> tuple[Path, Path]:
    artifact_path = tmp_path / "experiment_1486.json"
    manifest_path = tmp_path / "manifest_1486.jsonl"
    artifact_path.write_text(
        json.dumps(
            {
                "status": "complete",
                "live_sota_model_inference_used": True,
                "executable_constraint_benchmark_ready": True,
                "benchmark_cases": 20,
                "model_specs": [spec["hf_id"] for spec in cctu.MANDATED_MODEL_SPECS],
                "models_used": ["unsloth/Qwen3.6-35B-A3B-GGUF"],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    manifest_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    return artifact_path, manifest_path


def test_req_verify_1487_writes_in_progress_artifact(tmp_path: Path) -> None:
    """REQ-VERIFY-1487: startup artifact is durable before row loading."""

    out_path = tmp_path / "experiment_1487.json"

    artifact = exp.write_in_progress_artifact(out_path)

    assert json.loads(out_path.read_text(encoding="utf-8")) == artifact
    assert exp.REQUIRED_ARTIFACT_FIELDS <= set(artifact)
    assert artifact["status"] == "in_progress"
    assert artifact["pairwise_verification_complete"] is False
    assert artifact["candidate_pairs_evaluated"] == 0
    assert artifact["diagnostic_path"] == "results/v1_pairwise_verification_1487.json"


def test_req_verify_1487_constructs_valid_invalid_pairs() -> None:
    """REQ-VERIFY-1487: Exp 1486 rows become valid/invalid answer pairs."""

    cases = cctu.build_benchmark_cases()[:3]
    bad_payload = json.loads(cctu.compliant_transcript_for_case(cases[0]))
    bad_payload["final_answer"] = "not the answer"
    rows = [
        _manifest_row(cases[0], json.dumps(bad_payload)),
        _manifest_row(cases[0], json.dumps(bad_payload)),
        _manifest_row(cases[1], "no json here"),
        _manifest_row(cases[2], cctu.compliant_transcript_for_case(cases[2])),
        {"case_id": "unknown", "model_output": "ignored"},
    ]

    pairs = exp.construct_candidate_pairs(rows, cases)

    assert [pair.case_id for pair in pairs] == [case.case_id for case in cases]
    assert [pair.correct_label for pair in pairs] == ["A", "B", "A"]
    assert {pair.invalid_source for pair in pairs} == {
        "exp1486_model_output",
        "synthetic_invalid_from_valid_output",
    }
    for pair in pairs:
        assert exp.carnot_energy(pair.answer_by_label[pair.correct_label], pair.case) == 0
        wrong_label = "B" if pair.correct_label == "A" else "A"
        assert exp.carnot_energy(pair.answer_by_label[wrong_label], pair.case) > 0
        assert exp.energy_decision(pair) == pair.correct_label
        assert exp.beaver_style_decision(pair) == pair.correct_label


def test_req_verify_1487_loads_only_complete_live_exp1486_inputs(tmp_path: Path) -> None:
    """REQ-VERIFY-1487: source Exp 1486 artifact must be complete, ready, and live."""

    manifest_path = tmp_path / "manifest.jsonl"
    manifest_path.write_text("{}\n", encoding="utf-8")
    artifact_path = tmp_path / "artifact.json"
    base = {
        "status": "complete",
        "executable_constraint_benchmark_ready": True,
        "live_sota_model_inference_used": True,
    }

    artifact_path.write_text(json.dumps(base) + "\n", encoding="utf-8")
    artifact, rows = exp.load_exp1486_rows(artifact_path, manifest_path)
    assert artifact == base
    assert rows == [{}]

    for field, value, message in [
        ("status", "blocked", "complete"),
        ("executable_constraint_benchmark_ready", False, "ready"),
        ("live_sota_model_inference_used", False, "live SOTA"),
    ]:
        bad = dict(base)
        bad[field] = value
        artifact_path.write_text(json.dumps(bad) + "\n", encoding="utf-8")
        with pytest.raises(ValueError, match=message):
            exp.load_exp1486_rows(artifact_path, manifest_path)


def test_scenario_verify_1487_run_writes_diagnostic_and_terminal_artifact(
    tmp_path: Path,
) -> None:
    """SCENARIO-VERIFY-1487: run compares pairwise, energy, and superficial baselines."""

    cases = cctu.build_benchmark_cases()[:4]
    rows = []
    for case in cases:
        bad_payload = json.loads(cctu.compliant_transcript_for_case(case))
        bad_payload["tool_result"] = {"value": "wrong"}
        rows.append(_manifest_row(case, json.dumps(bad_payload)))
    artifact_path, manifest_path = _write_exp1486_inputs(tmp_path, rows)
    out_path = tmp_path / "experiment_1487.json"
    diagnostic_path = tmp_path / "diagnostic_1487.json"

    def fake_collector(_spec: dict[str, Any], pairs: list[exp.CandidatePair]) -> dict[str, Any]:
        decisions = [
            {
                "pair_id": pair.pair_id,
                "choice": pair.correct_label
                if index < 2
                else ("B" if pair.correct_label == "A" else "A"),
                "raw_output": pair.correct_label,
                "blocker": None,
            }
            for index, pair in enumerate(pairs)
        ]
        return {
            "summary": {
                "hf_id": "unsloth/Qwen3.6-35B-A3B-GGUF",
                "model_used": True,
                "blocker": None,
            },
            "decisions": decisions,
        }

    artifact = exp.run_evaluation(
        output_path=out_path,
        diagnostic_path=diagnostic_path,
        exp1486_artifact_path=artifact_path,
        exp1486_manifest_path=manifest_path,
        tests_run=[
            "pytest tests/python/test_experiment_1487_v1_pairwise_self_verification_vs_energy.py -q"
        ],
        collect_pairwise_choices_fn=fake_collector,
    )

    diagnostic = json.loads(diagnostic_path.read_text(encoding="utf-8"))
    assert json.loads(out_path.read_text(encoding="utf-8")) == artifact
    assert exp.REQUIRED_ARTIFACT_FIELDS <= set(artifact)
    assert artifact["status"] == "complete"
    assert artifact["live_sota_model_inference_used"] is True
    assert artifact["pairwise_verification_complete"] is True
    assert artifact["benchmark_cases_loaded"] == 4
    assert artifact["candidate_pairs_evaluated"] == 4
    assert artifact["pairwise_accuracy"] == 0.5
    assert artifact["energy_ranking_accuracy"] == 1.0
    assert artifact["random_baseline_accuracy"] == 0.5
    assert artifact["pairwise_delta_over_energy"] == -0.5
    assert artifact["improvement_allowed"] is False
    assert len(diagnostic["pairs"]) == 4
    assert diagnostic["model_attempts"][0]["model_used"] is True
    assert diagnostic["baseline_accuracies"]["beaver_style_ranking"] == 1.0


def test_req_verify_1487_run_records_blocked_pairwise_attempt_when_no_model_runs(
    tmp_path: Path,
) -> None:
    """REQ-VERIFY-1487: unavailable verifier models keep the verdict honest."""

    case = cctu.build_benchmark_cases()[0]
    bad_payload = json.loads(cctu.compliant_transcript_for_case(case))
    bad_payload["final_answer"] = "wrong"
    artifact_path, manifest_path = _write_exp1486_inputs(
        tmp_path,
        [_manifest_row(case, json.dumps(bad_payload))],
    )

    artifact = exp.run_evaluation(
        output_path=tmp_path / "experiment_1487.json",
        diagnostic_path=tmp_path / "diagnostic_1487.json",
        exp1486_artifact_path=artifact_path,
        exp1486_manifest_path=manifest_path,
        max_models=0,
    )

    assert artifact["live_sota_model_inference_used"] is False
    assert artifact["pairwise_verification_complete"] is False
    assert artifact["model_attempts"][0]["blocker"] == "not_attempted_runtime_budget"
    assert artifact["honest_verdict"].startswith("blocked:")


def test_req_verify_1487_improvement_gate_requires_energy_and_superficial_delta() -> None:
    """REQ-VERIFY-1487: improvement is blocked by energy or superficial ties."""

    assert exp.improvement_allowed(
        pairwise_accuracy=0.8,
        energy_ranking_accuracy=0.7,
        superficial_baseline_accuracy=0.6,
    )
    assert not exp.improvement_allowed(
        pairwise_accuracy=0.8,
        energy_ranking_accuracy=0.8,
        superficial_baseline_accuracy=0.6,
    )
    assert not exp.improvement_allowed(
        pairwise_accuracy=0.8,
        energy_ranking_accuracy=0.7,
        superficial_baseline_accuracy=0.8,
    )


def test_req_verify_1487_terminal_artifact_validation_and_verdict_edges(
    tmp_path: Path,
) -> None:
    """REQ-VERIFY-1487: terminal schema and honest verdict gates are strict."""

    metrics = {
        "candidate_pairs_evaluated": 1,
        "pairwise_accuracy": 0.8,
        "energy_ranking_accuracy": 0.7,
        "beaver_style_ranking_accuracy": 0.7,
        "random_baseline_accuracy": 0.5,
        "response_length_accuracy": 0.6,
        "format_validity_accuracy": 0.6,
        "superficial_baseline_accuracy": 0.6,
        "pairwise_delta_over_energy": 0.1,
        "improvement_allowed": True,
    }
    live = [{"hf_id": "unsloth/Qwen3.6-35B-A3B-GGUF", "model_used": True}]

    improved = exp.build_terminal_artifact(
        benchmark_cases_loaded=1,
        metrics=metrics,
        model_attempts=live,
        diagnostic_path=tmp_path / "diagnostic.json",
    )
    assert improved["honest_verdict"].startswith("complete: pairwise verifier improved")

    blocked = exp.build_terminal_artifact(
        benchmark_cases_loaded=1,
        metrics={**metrics, "improvement_allowed": True},
        model_attempts=[{"hf_id": "unsloth/Qwen3.6-35B-A3B-GGUF", "model_used": False}],
        diagnostic_path=tmp_path / "diagnostic.json",
    )
    assert blocked["honest_verdict"].startswith("blocked:")

    superficial = exp.build_terminal_artifact(
        benchmark_cases_loaded=1,
        metrics={
            **metrics,
            "superficial_baseline_accuracy": 0.8,
            "improvement_allowed": False,
        },
        model_attempts=live,
        diagnostic_path=tmp_path / "diagnostic.json",
    )
    assert superficial["honest_verdict"].endswith("superficial baseline")

    for mutation, message in [
        (lambda item: item.pop("status"), "missing required fields"),
        (lambda item: item.__setitem__("status", "in_progress"), "status"),
        (lambda item: item.__setitem__("candidate_pairs_evaluated", 0), "positive"),
        (lambda item: item.__setitem__("improvement_allowed", False), "strict gate"),
    ]:
        bad = dict(improved)
        mutation(bad)
        with pytest.raises(ValueError, match=message):
            exp.validate_terminal_artifact(bad)


def test_req_verify_1487_pairwise_choice_parser_is_bounded() -> None:
    """REQ-VERIFY-1487: local verifier outputs parse only bounded A/B decisions."""

    assert exp.parse_pairwise_choice("A") == "A"
    assert exp.parse_pairwise_choice("Answer: B because it is valid") == "B"
    assert exp.parse_pairwise_choice('{"choice": "A"}') == "A"
    assert exp.parse_pairwise_choice('{"winner": "B"}') == "B"
    assert exp.parse_pairwise_choice('{"answer": "b"}') == "B"
    assert exp.parse_pairwise_choice("choice = A") == "A"
    assert exp.parse_pairwise_choice("neither candidate is acceptable") is None


def test_req_verify_1487_live_pairwise_collector_is_injectable() -> None:
    """REQ-VERIFY-1487: live GGUF calls can be tested through injected hooks."""

    class FakeLlama:
        closed = False

        def __init__(self, **_kwargs: Any) -> None:
            self.calls = 0

        def __call__(self, _prompt: str, **_kwargs: Any) -> dict[str, Any]:
            self.calls += 1
            if self.calls == 2:
                raise RuntimeError("generation failed")
            return {"choices": [{"text": '{"choice": "A"}'}]}

        def close(self) -> None:
            FakeLlama.closed = True

    class LoadFails:
        def __init__(self, **_kwargs: Any) -> None:
            raise RuntimeError("load failed")

    cases = cctu.build_benchmark_cases()[:2]
    rows = [_manifest_row(case, "invalid") for case in cases]
    pairs = exp.construct_candidate_pairs(rows, cases)
    spec = {"hf_id": "unsloth/Qwen3.6-35B-A3B-GGUF", "name": "Qwen", "gpu": 0}

    missing = exp.collect_live_pairwise_choices(
        spec,
        pairs,
        resolver=lambda _hf_id: None,
        env_preparer=lambda: {},
    )
    import_failed = exp.collect_live_pairwise_choices(
        spec,
        pairs,
        resolver=lambda _hf_id: "/tmp/model.gguf",
        llama_importer=lambda: (False, None, "llama_cpp missing"),
        env_preparer=lambda: {},
    )
    load_failed = exp.collect_live_pairwise_choices(
        {**spec, "model_path": "/tmp/model.gguf"},
        pairs,
        llama_importer=lambda: (True, LoadFails, None),
        env_preparer=lambda: {},
    )
    ok = exp.collect_live_pairwise_choices(
        {**spec, "model_path": "/tmp/model.gguf"},
        pairs,
        llama_importer=lambda: (True, FakeLlama, None),
        env_preparer=lambda: {},
    )

    assert missing["summary"]["blocker"] == "model_not_cached"
    assert import_failed["summary"]["blocker"] == "llama_cpp missing"
    assert "load failed" in load_failed["summary"]["blocker"]
    assert ok["summary"]["model_used"] is True
    assert ok["decisions"][0]["choice"] == "A"
    assert ok["decisions"][1]["blocker"] == "RuntimeError: generation failed"
    assert FakeLlama.closed is True


def test_req_verify_1487_tie_and_private_edge_cases() -> None:
    """REQ-VERIFY-1487: tie handling uses half-credit rather than overclaiming."""

    case = cctu.build_benchmark_cases()[0]
    valid = cctu.compliant_transcript_for_case(case)
    pair = exp.CandidatePair(
        pair_id="tie",
        case_id=case.case_id,
        case=case,
        answer_a=valid,
        answer_b=valid,
        correct_label="A",
        invalid_source="test",
        source_model_hf_id=None,
    )

    assert exp.energy_decision(pair) is None
    assert exp.beaver_style_decision(pair) is None
    assert exp._shorter_length_decision(pair) is None
    assert exp._format_validity_decision(pair) is None
    assert (
        exp._format_validity_decision(
            exp.CandidatePair(
                pair_id="format",
                case_id=case.case_id,
                case=case,
                answer_a=valid,
                answer_b="not json",
                correct_label="A",
                invalid_source="test",
                source_model_hf_id=None,
            )
        )
        == "A"
    )
    assert exp._choice_accuracy([]) == 0.0
    scored, metrics = exp.score_pairs([pair], {})
    assert scored[0]["energy_choice"] is None
    assert metrics["pairwise_accuracy"] == 0.0
    assert metrics["energy_ranking_accuracy"] == 0.5
    assert exp._row_is_base_valid({"validator_result": {}}) is False
    assert exp._row_is_base_valid(
        {
            "validator_result": {
                "tool_call_structure_valid": True,
                "tool_result_consistent": True,
                "final_answer_valid": True,
                "verifier_outcome_valid": True,
            }
        }
    )
    assert exp._row_is_base_valid({}) is False


def test_req_verify_1487_main_reports_summary(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """REQ-VERIFY-1487: CLI wrapper reports the terminal pairwise fields."""

    def fake_run_evaluation(*, max_models: int) -> dict[str, Any]:
        assert max_models == len(exp.MANDATED_MODEL_SPECS)
        return {
            "pairwise_verification_complete": True,
            "candidate_pairs_evaluated": 2,
            "pairwise_accuracy": 1.0,
            "energy_ranking_accuracy": 0.5,
            "improvement_allowed": True,
        }

    monkeypatch.setattr(exp, "run_evaluation", fake_run_evaluation)

    assert exp.main(["--all-models"]) == 0
    assert "pairwise_complete=True" in capsys.readouterr().out
