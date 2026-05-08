"""Tests for Exp 1556 ARM/EBT logprob telemetry repair.

Spec: REQ-VERIFY-1556, SCENARIO-VERIFY-1556.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from carnot.verify import arm_ebm_logprob_telemetry_repair as exp


QWEN = "unsloth/Qwen3.6-35B-A3B-GGUF"


def test_req_verify_1556_parses_logprob_and_topk_telemetry() -> None:
    """REQ-VERIFY-1556: token logprobs and top-k alternatives become diagnostics."""

    signals = exp.parse_telemetry_row(
        {
            "case_id": "sat-a",
            "response_text": "SAT",
            "token_logprobs": [-0.1, "-0.3", None, True],
            "top_logprobs": [{"SAT": -0.1, "UNSAT": "-2.4", "skip": False}],
            "topk_alternatives_available": True,
            "blocker": None,
        }
    )

    assert signals["logprob_available"] is True
    assert signals["topk_available"] is True
    assert signals["mean_logprob"] == pytest.approx(-0.2)
    assert signals["semantic_energy"] == pytest.approx(0.2)
    assert signals["topk_position_count"] == 1
    assert signals["telemetry_blockers"] == []


def test_req_verify_1556_soft_signals_never_override_validator() -> None:
    """REQ-VERIFY-1556: soft accept signals cannot override deterministic rejection."""

    rejected_with_confident_soft_signal = {
        "case_id": "reject",
        "deterministic_accept": False,
        "model_declared_accept": True,
        "mean_logprob": -0.001,
        "topk_available": True,
        "soft_signal_accept": True,
    }
    accepted_with_poor_soft_signal = {
        "case_id": "accept",
        "deterministic_accept": True,
        "mean_logprob": -12.0,
        "topk_available": False,
        "soft_signal_accept": False,
    }

    assert exp.final_authority_accept(rejected_with_confident_soft_signal) is False
    assert exp.final_authority_accept(accepted_with_poor_soft_signal) is True


def test_scenario_verify_1556_runner_writes_ready_artifact(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-1556: runner joins live telemetry to deterministic labels."""

    source_manifest = tmp_path / "satquest.jsonl"
    output_path = tmp_path / "results" / "experiment_1556.json"
    telemetry_manifest = tmp_path / "results" / "telemetry.jsonl"
    diagnostic_report = tmp_path / "results" / "diagnostic.jsonl"
    source_rows = [
        _satquest_row("sat-ok-1", correct=True, energy=0.0),
        _satquest_row("sat-ok-2", correct=True, energy=0.2),
        _satquest_row("sat-bad-1", correct=False, energy=5.0),
        _satquest_row("sat-bad-2", correct=False, energy=7.0),
    ]
    _write_jsonl(source_manifest, source_rows)

    artifact = exp.run_experiment(
        project_root=tmp_path,
        run_date="20260508",
        output_path=output_path,
        telemetry_manifest_path=telemetry_manifest,
        diagnostic_report_path=diagnostic_report,
        satquest_manifest_path=source_manifest,
        focused_tests_passed=True,
        case_limit=4,
        telemetry_builder=_telemetry_builder(logprobs=True, topk=True),
    )
    report_rows = _read_jsonl(diagnostic_report)

    assert json.loads(output_path.read_text(encoding="utf-8")) == artifact
    assert set(exp.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert artifact["status"] == "complete"
    assert artifact["milestone"] == "20260508"
    assert artifact["arm_ebm_logprob_telemetry_ready"] is True
    assert artifact["model_specs"] == list(exp.MODEL_SPECS)
    assert artifact["live_sota_model_inference_used"] is True
    assert artifact["logprob_available"] is True
    assert artifact["topk_available"] is True
    assert artifact["telemetry_adapter_path"] == exp.TELEMETRY_ADAPTER_PATH
    assert artifact["diagnostic_cases"] == 4
    assert artifact["energy_label_correlation"] > 0.9
    assert artifact["routing_auc"] == pytest.approx(1.0)
    assert artifact["deterministic_validators_final_authority"] is True
    assert artifact["telemetry_blockers"] == []
    assert artifact["focused_tests_passed"] is True
    assert artifact["honest_verdict"].startswith("complete:")
    assert all(row["soft_signal_overrode_validator"] is False for row in report_rows)
    assert [row["deterministic_final_accept"] for row in report_rows] == [
        row["deterministic_accept"] for row in report_rows
    ]


def test_req_verify_1556_missing_runtime_telemetry_blocks_honestly(tmp_path: Path) -> None:
    """REQ-VERIFY-1556: text-only live responses are not promoted to telemetry readiness."""

    source_manifest = tmp_path / "satquest.jsonl"
    _write_jsonl(
        source_manifest,
        [
            _satquest_row("sat-ok", correct=True, energy=0.0),
            _satquest_row("sat-bad", correct=False, energy=6.0),
        ],
    )

    artifact = exp.run_experiment(
        project_root=tmp_path,
        run_date="20260508",
        output_path=tmp_path / "results" / "experiment_1556.json",
        telemetry_manifest_path=tmp_path / "results" / "telemetry.jsonl",
        diagnostic_report_path=tmp_path / "results" / "diagnostic.jsonl",
        satquest_manifest_path=source_manifest,
        focused_tests_passed=True,
        case_limit=2,
        telemetry_builder=_telemetry_builder(logprobs=False, topk=False),
    )

    assert artifact["status"] == "complete"
    assert artifact["arm_ebm_logprob_telemetry_ready"] is False
    assert artifact["live_sota_model_inference_used"] is True
    assert artifact["logprob_available"] is False
    assert artifact["topk_available"] is False
    assert "token_logprobs_missing" in artifact["telemetry_blockers"]
    assert "topk_logprobs_missing" in artifact["telemetry_blockers"]
    assert artifact["honest_verdict"].startswith("complete:")


def test_req_verify_1556_selects_bounded_balanced_cases() -> None:
    """REQ-VERIFY-1556: SATQuest source selection keeps both validator labels when present."""

    rows = [
        _satquest_row("bad-1", correct=False, energy=9.0),
        _satquest_row("bad-2", correct=False, energy=8.0),
        _satquest_row("bad-3", correct=False, energy=7.0),
        _satquest_row("ok-1", correct=True, energy=0.0),
        _satquest_row("ok-2", correct=True, energy=0.1),
    ]

    selected = exp.select_satquest_rows(rows, limit=4)

    assert len(selected) == 4
    assert {bool((row["baseline"])["correct"]) for row in selected} == {False, True}
    assert [row["case_id"] for row in selected] == ["ok-1", "ok-2", "bad-1", "bad-2"]


def test_req_verify_1556_validate_artifact_rejects_bad_terminal_shape() -> None:
    """REQ-VERIFY-1556: required fields and final-authority invariants are enforced."""

    artifact = {
        "status": "complete",
        "milestone": "20260508",
        "arm_ebm_logprob_telemetry_ready": True,
        "model_specs": list(exp.MODEL_SPECS),
        "live_sota_model_inference_used": True,
        "logprob_available": True,
        "topk_available": True,
        "telemetry_adapter_path": exp.TELEMETRY_ADAPTER_PATH,
        "diagnostic_cases": 1,
        "energy_label_correlation": 1.0,
        "routing_auc": 1.0,
        "deterministic_validators_final_authority": True,
        "telemetry_blockers": [],
        "focused_tests_passed": True,
        "honest_verdict": "complete: ready",
    }

    exp.validate_artifact(artifact)
    with pytest.raises(AssertionError, match="missing required fields"):
        exp.validate_artifact({key: value for key, value in artifact.items() if key != "status"})
    with pytest.raises(AssertionError, match="allowed terminal prefix"):
        exp.validate_artifact(artifact | {"honest_verdict": "blocked: no"})
    with pytest.raises(AssertionError, match="deterministic validators"):
        exp.validate_artifact(artifact | {"deterministic_validators_final_authority": False})
    with pytest.raises(AssertionError, match="focused tests"):
        exp.validate_artifact(artifact | {"focused_tests_passed": False})
    with pytest.raises(AssertionError, match="live SOTA inference"):
        exp.validate_artifact(artifact | {"live_sota_model_inference_used": False})
    with pytest.raises(AssertionError, match="live SOTA logprob and top-k telemetry"):
        exp.validate_artifact(artifact | {"topk_available": False})


def test_req_verify_1556_edge_branches_cover_blockers_and_ties(tmp_path: Path) -> None:
    """REQ-VERIFY-1556: edge cases stay deterministic and honest."""

    only_rejects = [
        _satquest_row("bad-1", correct=False, energy=9.0),
        _satquest_row("bad-2", correct=False, energy=8.0),
    ]
    assert [row["case_id"] for row in exp.select_satquest_rows(only_rejects, limit=1)] == ["bad-1"]

    tied_summary = exp.evaluate_diagnostic_rows(
        [
            _diagnostic_row("ok", accept=True, energy=1.0, routing=1.0),
            _diagnostic_row("bad", accept=False, energy=1.0, routing=1.0),
        ],
        focused_tests_passed=True,
        telemetry_artifact={"live_sota_model_inference_used": True, "blockers": []},
    )
    assert tied_summary["energy_label_correlation"] is None
    assert tied_summary["routing_auc"] == pytest.approx(0.5)

    one_label_summary = exp.evaluate_diagnostic_rows(
        [_diagnostic_row("ok", accept=True, energy=0.0, routing=0.0)],
        focused_tests_passed=True,
        telemetry_artifact={"live_sota_model_inference_used": True, "blockers": []},
    )
    assert one_label_summary["routing_auc"] is None

    artifact = exp.run_experiment(
        project_root=tmp_path,
        run_date="20260508",
        output_path=tmp_path / "results" / "experiment_1556.json",
        telemetry_manifest_path=tmp_path / "results" / "empty_telemetry.jsonl",
        diagnostic_report_path=tmp_path.parent / "outside_1556_diagnostic.jsonl",
        satquest_manifest_path=tmp_path / "missing_satquest.jsonl",
        focused_tests_passed=True,
        case_limit=2,
        telemetry_builder=_telemetry_builder(logprobs=True, topk=True),
    )
    assert artifact["diagnostic_cases"] == 0
    assert "no_diagnostic_cases_loaded" in artifact["telemetry_blockers"]
    assert artifact["diagnostic_report_path"].startswith(str(tmp_path.parent))


def _satquest_row(case_id: str, *, correct: bool, energy: float) -> dict[str, Any]:
    return {
        "case_id": case_id,
        "family": "fixture_satquest",
        "prompt": f"Solve SATQuest fixture {case_id}",
        "model_hf_id": QWEN,
        "model_name": "Qwen3.6-35B-A3B",
        "model_output": "SAT" if correct else "UNSAT",
        "baseline": {
            "answer": "SAT" if correct else "UNSAT",
            "classification": "oracle_agreement" if correct else "wrong_label",
            "correct": correct,
            "energy": energy,
        },
        "solver_oracle": {"label": "SAT" if correct else "UNSAT"},
        "parse_result": {"model_declared_accept": not correct},
    }


def _diagnostic_row(case_id: str, *, accept: bool, energy: float, routing: float) -> dict[str, Any]:
    return {
        "case_id": case_id,
        "deterministic_accept": accept,
        "reject_label": 0 if accept else 1,
        "carnot_energy_score": energy,
        "routing_score": routing,
        "logprob_available": True,
        "topk_available": True,
        "semantic_energy": 0.1,
        "telemetry_blockers": [],
    }


def _telemetry_builder(*, logprobs: bool, topk: bool):
    def build(
        *,
        project_root: Path,
        run_date: str,
        cases: list[Any],
        manifest_path: Path,
    ) -> dict[str, Any]:
        del project_root, run_date
        rows = []
        for index, case in enumerate(cases):
            token_logprobs = [-0.05, -0.2 - index] if logprobs else []
            top_logprobs = [{"SAT": -0.05, "UNSAT": -1.5 - index}] if topk else []
            rows.append(
                {
                    "case_id": case.case_id,
                    "hf_id": QWEN,
                    "model_name": "Qwen3.6-35B-A3B",
                    "response_text": "SAT",
                    "token_logprobs": token_logprobs,
                    "token_logprobs_available": bool(token_logprobs),
                    "top_logprobs": top_logprobs,
                    "topk_alternatives_available": bool(top_logprobs),
                    "topk_position_count": len(top_logprobs),
                    "blocker": None,
                }
            )
        _write_jsonl(manifest_path, rows)
        return {
            "status": "complete",
            "live_sota_model_inference_used": True,
            "models_used": [QWEN],
            "telemetry_cases_completed": len(rows),
            "topk_logprobs_available": bool(topk),
            "logprob_available": bool(logprobs),
            "blockers": [] if logprobs and topk else ["runtime_text_without_full_telemetry"],
        }

    return build


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
