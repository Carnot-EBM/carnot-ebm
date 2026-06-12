"""Tests for Exp 4080 Sudoku verifier-RFT positive control.

Spec refs: REQ-LEARN-4080, SCENARIO-LEARN-4080,
SCENARIO-LEARN-4080-FAIL.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from carnot.agentic import sudoku_exp4080_rft_positive_control as exp4080


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "self-learning" / "spec.md"


def _available_checks() -> list[exp4080.PreconditionCheck]:
    return [
        exp4080.PreconditionCheck("cuda_visible", True, "test"),
        exp4080.PreconditionCheck("trl_peft_trainers", True, "test"),
    ]


def _source_artifact(rows: list[dict[str, float | int]]) -> dict[str, object]:
    return {
        "honest_verdict": "complete: source_sudoku_live_gpu",
        "inference_substrate": "live_gpu_training_energy_self_distillation_plus_gate_retest",
        "per_seed": rows,
        "seeds": [row["seed"] for row in rows],
        "duration_s": 123.0,
    }


def _row(seed: int, rft: float, sft: float) -> dict[str, float | int]:
    return {
        "seed": seed,
        "energy_distilled_greedy": rft,
        "gold_distilled_greedy": sft,
    }


def test_req_learn_4080_spec_declares_positive_control_contract() -> None:
    """REQ-LEARN-4080: OpenSpec declares the Sudoku positive-control fields."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-LEARN-4080" in spec
    assert "SCENARIO-LEARN-4080" in spec
    assert "SCENARIO-LEARN-4080-FAIL" in spec
    assert exp4080.RESULT_FILENAME in spec
    assert "Exp 4078" in spec
    for field in exp4080.REQUIRED_ARTIFACT_FIELDS:
        assert field in spec


def test_scenario_learn_4080_reproduces_beachhead_from_three_seed_artifact(
    tmp_path: Path,
) -> None:
    """SCENARIO-LEARN-4080: three held-out seeds reproduce RFT >= SFT."""

    source_path = tmp_path / "sudoku_source.json"
    source_path.write_text(
        json.dumps(
            _source_artifact(
                [
                    _row(0, 0.0615, 0.0475),
                    _row(1, 0.0625, 0.0475),
                    _row(2, 0.0735, 0.0565),
                ]
            )
        ),
        encoding="utf-8",
    )
    output_path = tmp_path / "artifact.json"

    artifact = exp4080.run_experiment(
        repo_root=tmp_path,
        source_path=source_path,
        output_path=output_path,
        preconditions_checker=lambda **_: _available_checks(),
    )

    assert json.loads(output_path.read_text(encoding="utf-8")) == artifact
    assert artifact["honest_verdict"] == "complete: sudoku_positive_control_rft_ge_sft_reproduced"
    assert artifact["rft_rate"] == pytest.approx(0.065833)
    assert artifact["sft_rate"] == pytest.approx(0.0505)
    assert artifact["n_seeds"] == 3
    assert artifact["reproduces_beachhead"] is True
    assert "pipeline sanity" in artifact["field_principles"]["reproduces_beachhead"]
    assert exp4080.artifact_schema_errors(artifact) == []


def test_scenario_learn_4080_flags_pipeline_suspect_when_rft_under_sft(
    tmp_path: Path,
) -> None:
    """SCENARIO-LEARN-4080-FAIL: RFT<SFT flags the pipeline as suspect."""

    source_path = tmp_path / "sudoku_source.json"
    source_path.write_text(
        json.dumps(
            _source_artifact(
                [
                    _row(0, 0.10, 0.12),
                    _row(1, 0.11, 0.12),
                    _row(2, 0.09, 0.12),
                ]
            )
        ),
        encoding="utf-8",
    )

    artifact = exp4080.run_experiment(
        repo_root=tmp_path,
        source_path=source_path,
        output_path=tmp_path / "artifact.json",
        preconditions_checker=lambda **_: _available_checks(),
    )

    assert artifact["honest_verdict"] == "complete: sudoku_positive_control_FAILED_pipeline_suspect"
    assert artifact["rft_rate"] == pytest.approx(0.10)
    assert artifact["sft_rate"] == pytest.approx(0.12)
    assert artifact["reproduces_beachhead"] is False
    assert exp4080.artifact_schema_errors(artifact) == []


def test_req_learn_4080_blocks_missing_preconditions_without_claim(
    tmp_path: Path,
) -> None:
    """REQ-LEARN-4080-1: missing Exp 4078-style resources block the claim."""

    source_path = tmp_path / "sudoku_source.json"
    source_path.write_text(json.dumps(_source_artifact([_row(0, 1.0, 0.0)])), encoding="utf-8")

    artifact = exp4080.run_experiment(
        repo_root=tmp_path,
        source_path=source_path,
        output_path=tmp_path / "artifact.json",
        preconditions_checker=lambda **_: [
            exp4080.PreconditionCheck("cuda_visible", False, "no gpu"),
            exp4080.PreconditionCheck("trl_peft_trainers", True, "test"),
        ],
    )

    assert artifact["honest_verdict"] == "blocked_cuda_visible"
    assert artifact["rft_rate"] == 0.0
    assert artifact["sft_rate"] == 0.0
    assert artifact["n_seeds"] == 0
    assert artifact["reproduces_beachhead"] is False
    assert exp4080.artifact_schema_errors(artifact) == []


def test_req_learn_4080_source_validation_and_schema_errors(tmp_path: Path) -> None:
    """REQ-LEARN-4080: source rows and terminal schema are defensive."""

    raw_check = type("RawCheck", (), {"resource": "cuda_visible", "available": 1, "detail": "ok"})()
    assert exp4080._coerce_precondition(raw_check) == exp4080.PreconditionCheck(
        "cuda_visible",
        True,
        "ok",
    )

    rows = exp4080.extract_seed_rates(
        _source_artifact([_row(0, 0.2, 0.1), _row(1, 0.3, 0.2), _row(2, 0.4, 0.3)])
    )
    assert [row.seed for row in rows] == [0, 1, 2]
    assert exp4080.mean_rate(row.rft_rate for row in rows) == pytest.approx(0.3)
    assert exp4080.positive_control_reproduces(rows) is True

    with pytest.raises(ValueError, match="at least 3"):
        exp4080.extract_seed_rates(_source_artifact([_row(0, 0.2, 0.1), _row(1, 0.3, 0.2)]))
    with pytest.raises(ValueError, match="per_seed"):
        exp4080.extract_seed_rates({"honest_verdict": "complete: missing_rows"})
    with pytest.raises(ValueError, match="seed row 0 must be an object"):
        exp4080.extract_seed_rates(
            {
                "honest_verdict": "complete: source_sudoku_live_gpu",
                "per_seed": ["bad", _row(1, 0.3, 0.2), _row(2, 0.4, 0.3)],
            }
        )
    with pytest.raises(ValueError, match="seed row 0 missing"):
        exp4080.extract_seed_rates(_source_artifact([{"seed": 0}, _row(1, 0.3, 0.2), _row(2, 0.4, 0.3)]))
    with pytest.raises(ValueError, match="seed row 0 has non-integer seed"):
        exp4080.extract_seed_rates(
            _source_artifact(
                [
                    {"seed": "0", "energy_distilled_greedy": 0.2, "gold_distilled_greedy": 0.2},
                    _row(1, 0.3, 0.2),
                    _row(2, 0.4, 0.3),
                ]
            )
        )
    with pytest.raises(ValueError, match="seed row 0 has non-numeric"):
        exp4080.extract_seed_rates(
            _source_artifact(
                [
                    {"seed": 0, "energy_distilled_greedy": "bad", "gold_distilled_greedy": 0.2},
                    _row(1, 0.3, 0.2),
                    _row(2, 0.4, 0.3),
                ]
            )
        )

    source_path = tmp_path / "not-an-object.json"
    source_path.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="source artifact must be a JSON object"):
        exp4080.load_source_artifact(source_path)

    with pytest.raises(ValueError, match="honest_verdict must be terminal-prefixed"):
        exp4080._base_artifact(
            honest_verdict="bad",
            rft_rate=0.1,
            sft_rate=0.1,
            n_seeds=3,
            reproduces_beachhead=False,
            preconditions_checked=_available_checks(),
            duration_s=0.0,
            extra={},
        )

    errors = exp4080.artifact_schema_errors(
        {
            "honest_verdict": "bad",
            "rft_rate": "0.1",
            "sft_rate": True,
            "n_seeds": False,
            "reproduces_beachhead": "yes",
            "inference_substrate": "",
            "preconditions_checked": [{}],
        }
    )
    assert "honest_verdict must be terminal-prefixed" in errors
    assert "rft_rate must be a bare float" in errors
    assert "sft_rate must be a bare float" in errors
    assert "n_seeds must be a bare int" in errors
    assert "reproduces_beachhead must be a bare bool" in errors
    assert "inference_substrate must be a non-empty string" in errors
    assert "preconditions_checked entries must include resource and available" in errors

    assert "missing required field honest_verdict" in exp4080.artifact_schema_errors({})
