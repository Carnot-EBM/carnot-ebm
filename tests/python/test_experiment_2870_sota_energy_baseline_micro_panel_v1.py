"""Tests for Exp 2870 SOTA energy baseline micro-panel.

Spec: REQ-INFER-SOTA-014,
      SCENARIO-INFER-SOTA-014-001
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pytest

from carnot.reporting import sota_energy_baseline_micro_panel_v1 as mod
from carnot.reporting.sota_energy_baseline_micro_panel_v1 import (
    ExperimentConfig,
    REQUIRED_ARTIFACT_FIELDS,
)


QWEN = "unsloth/Qwen3.6-35B-A3B-GGUF"
GEMMA31 = "unsloth/gemma-4-31B-it-GGUF"
GEMMA26 = "unsloth/gemma-4-26B-A4B-it-GGUF"


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _exp2862_payload(*, ready: bool = True) -> dict[str, Any]:
    return {
        "sota_runtime_ready_v3": ready,
        "selected_python": "/repo/.venv/bin/python",
        "selected_model_hf_id": GEMMA26,
        "selected_model_path": "/cache/gemma-4-26B-A4B-it-UD-Q4_K_M.gguf",
        "model_specs": [
            {"hf_id": QWEN, "legacy_smoke_only": False},
            {"hf_id": GEMMA31, "legacy_smoke_only": False},
            {"hf_id": GEMMA26, "legacy_smoke_only": False},
        ],
        "preconditions_checked": [{"resource": "llama_cpp_gpu_offload", "available": True}],
    }


def _manifest_rows(count: int = 12) -> list[dict[str, Any]]:
    labels = ["SUPPORTS", "REFUTES", "NOT ENOUGH INFO"]
    rows: list[dict[str, Any]] = []
    for index in range(count):
        label_text = labels[index % len(labels)]
        rows.append(
            {
                "dataset": "FEVER",
                "stable_id": f"fever-fixture-{index:02d}",
                "prompt": f"Evidence sentence {index}.",
                "claim": f"Fixture claim {index}.",
                "label_text": label_text,
            }
        )
    return rows


def _fake_panel_runner(*, with_logprobs: bool) -> mod.PanelRunnerFn:
    def run(
        *,
        model_spec: dict[str, Any],
        examples: list[mod.MicroPanelExample],
        selected_python: str,
        env: dict[str, str],
        random_seed: int,
    ) -> list[dict[str, Any]]:
        del selected_python, env, random_seed
        rows: list[dict[str, Any]] = []
        for index, example in enumerate(examples):
            correct = index % 2 == 0
            answer = example.expected_answer if correct else "REFUTES"
            if answer == example.expected_answer and not correct:
                answer = "SUPPORTS"
            row: dict[str, Any] = {
                "example_id": example.example_id,
                "model_hf_id": model_spec["hf_id"],
                "model_path": model_spec["model_path"],
                "response_text": answer,
                "tokens_generated": 1,
                "duration_s": 0.1,
                "logprobs_requested": True,
                "logprobs_available": with_logprobs,
            }
            if with_logprobs:
                prob = 0.92 if correct else 0.12
                row["tokens"] = [answer]
                row["token_logprobs"] = [math.log(prob)]
                row["top_logprobs"] = [{answer: math.log(prob), "other": math.log(1.0 - prob)}]
            rows.append(row)
        return rows

    return run


def test_scenario_infer_sota_014_live_micro_panel_reports_logit_signals(
    tmp_path: Path,
) -> None:
    """SCENARIO-INFER-SOTA-014-001: live rows produce cheap logit AUROCs."""
    _write_json(tmp_path / "results" / mod.EXP2862_FILENAME, _exp2862_payload())
    _write_jsonl(tmp_path / "data" / "eval_manifests" / "fever_20260522.jsonl", _manifest_rows())
    output_path = tmp_path / "results" / mod.OUTPUT_FILENAME

    artifact = mod.run_experiment(
        ExperimentConfig(
            repo_root=tmp_path,
            output_path=output_path,
            run_date="20260522",
            n_examples=10,
            started_at=10.0,
            clock=lambda: 13.5,
        ),
        cached_pair_fn=lambda *, gpu_indices, preferred_quant: [
            {"name": "Gemma4-26B-A4B-it", "hf_id": GEMMA26, "gpu": 0, "model_path": "/cache/gemma.gguf"},
            {"name": "Gemma4-31B-it", "hf_id": GEMMA31, "gpu": 1, "model_path": "/cache/gemma31.gguf"},
        ],
        panel_runner_fn=_fake_panel_runner(with_logprobs=True),
    )

    assert json.loads(output_path.read_text(encoding="utf-8")) == artifact
    assert set(REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert artifact["honest_verdict"] == "micro_panel_complete_no_full_benchmark_claim"
    assert artifact["micro_panel_ready"] is True
    assert artifact["live_model_invoked"] is True
    assert artifact["model_specs"][0]["hf_id"] == GEMMA26
    assert artifact["models_used"] == [GEMMA26]
    assert artifact["n_examples"] == 10
    assert artifact["usable_response_count"] == 10
    assert artifact["first_token_confidence_available"] is True
    assert artifact["spilled_energy_available"] is True
    assert artifact["first_token_confidence_auroc"] == pytest.approx(1.0)
    assert artifact["spilled_energy_auroc"] == pytest.approx(1.0)
    assert "blocked_logprobs_unavailable" not in artifact["blocked_metrics"]
    assert len(artifact["sample_rows"]) == 10
    assert len(artifact["reproducibility_checksum"]) == 64
    assert artifact["field_principles"]["claim_boundary"].startswith("Tiny live micro-panel only")
    assert artifact["duration_s"] == pytest.approx(3.5)


def test_req_infer_sota_014_exp2862_not_ready_blocks_without_live_call(
    tmp_path: Path,
) -> None:
    """REQ-INFER-SOTA-014: Exp 2862 must gate all live micro-panel work."""
    _write_json(tmp_path / "results" / mod.EXP2862_FILENAME, _exp2862_payload(ready=False))
    _write_jsonl(tmp_path / "data" / "eval_manifests" / "fever_20260522.jsonl", _manifest_rows())

    artifact = mod.run_experiment(
        ExperimentConfig(
            repo_root=tmp_path,
            output_path=tmp_path / "results" / mod.OUTPUT_FILENAME,
            started_at=2.0,
            clock=lambda: 2.25,
        ),
        cached_pair_fn=lambda *, gpu_indices, preferred_quant: pytest.fail("pair should not run"),
        panel_runner_fn=lambda **_: pytest.fail("live panel should not run"),
    )

    assert artifact["honest_verdict"] == "blocked_exp2862_sota_runtime_not_ready"
    assert artifact["micro_panel_ready"] is False
    assert artifact["live_model_invoked"] is False
    assert artifact["n_examples"] == 0
    assert artifact["blocked_metrics"] == ["blocked_exp2862_sota_runtime_not_ready"]
    assert artifact["first_token_confidence_auroc"] is None
    assert artifact["spilled_energy_auroc"] is None
    assert artifact["preconditions_checked"][0]["resource"] == "exp2862_sota_runtime_ready_v3"
    assert artifact["duration_s"] == pytest.approx(0.25)


def test_req_infer_sota_014_missing_logprobs_blocks_metrics_but_keeps_generation(
    tmp_path: Path,
) -> None:
    """REQ-INFER-SOTA-014: unavailable logprobs are recorded, not fabricated."""
    _write_json(tmp_path / "results" / mod.EXP2862_FILENAME, _exp2862_payload())
    _write_jsonl(tmp_path / "data" / "eval_manifests" / "fever_20260522.jsonl", _manifest_rows())

    artifact = mod.run_experiment(
        ExperimentConfig(
            repo_root=tmp_path,
            output_path=tmp_path / "results" / mod.OUTPUT_FILENAME,
            n_examples=10,
            started_at=1.0,
            clock=lambda: 2.0,
        ),
        cached_pair_fn=lambda *, gpu_indices, preferred_quant: None,
        panel_runner_fn=_fake_panel_runner(with_logprobs=False),
    )

    assert artifact["micro_panel_ready"] is True
    assert artifact["live_model_invoked"] is True
    assert artifact["model_specs"] == [
        {
            "name": "Gemma4-26B-A4B-it",
            "hf_id": GEMMA26,
            "gpu": 0,
            "model_path": "/cache/gemma-4-26B-A4B-it-UD-Q4_K_M.gguf",
            "source": "exp2862_selected_model_fallback",
        }
    ]
    assert artifact["models_used"] == [GEMMA26]
    assert artifact["usable_response_count"] == 10
    assert artifact["first_token_confidence_available"] is False
    assert artifact["spilled_energy_available"] is False
    assert artifact["first_token_confidence_auroc"] is None
    assert artifact["spilled_energy_auroc"] is None
    assert artifact["blocked_metrics"] == ["blocked_logprobs_unavailable"]
    assert all(row["first_token_confidence"] is None for row in artifact["sample_rows"])


def test_req_infer_sota_014_helpers_are_deterministic(tmp_path: Path) -> None:
    """REQ-INFER-SOTA-014: panel selection, AUROC, and parsing are deterministic."""
    _write_jsonl(
        tmp_path / "data" / "eval_manifests" / "fever_20260522.jsonl",
        [
            *_manifest_rows(15),
            {"stable_id": "unknown-label", "claim": "x", "prompt": "y", "label_text": "MAYBE"},
            {"stable_id": "missing-context", "claim": "x", "label_text": "SUPPORTS"},
        ],
    )

    selected = mod.select_micro_panel(
        tmp_path,
        n_examples=20,
        random_seed=2870,
        manifest_paths=(
            Path("data/eval_manifests/missing.jsonl"),
            Path("data/eval_manifests/fever_20260522.jsonl"),
        ),
    )

    assert len(selected) == 15
    assert selected[0].example_id == "fever-fixture-00"
    assert "Classify the claim" in selected[0].prompt_text()
    assert {example.expected_answer for example in selected} == {"SUPPORTS", "REFUTES", "UNKNOWN"}
    assert mod.compute_auroc([1, 1, 0, 0], [0.9, 0.8, 0.2, 0.1]) == pytest.approx(1.0)
    assert mod.compute_auroc([1, 0], [0.5, 0.5]) == pytest.approx(0.5)
    assert mod.compute_auroc([1, 1], [0.1, 0.2]) is None
    assert mod.compute_auroc([], []) is None
    with pytest.raises(ValueError, match="same length"):
        mod.compute_auroc([1], [0.2, 0.3])
    with pytest.raises(ValueError, match="finite"):
        mod.compute_auroc([1, 0], [0.2, math.inf])
    assert mod.normalize_expected_answer("NOT ENOUGH INFO") == "UNKNOWN"
    assert mod.normalize_expected_answer("maybe") == "MAYBE"
    assert mod.classify_response(" supports\nbecause") == "SUPPORTS"
    assert mod.classify_response("Not enough info.") == "UNKNOWN"
    assert mod.classify_response("") is None
    assert mod.classify_response("I cannot tell") is None
    assert ExperimentConfig(repo_root=tmp_path).resolved_output_path() == (
        tmp_path / "results" / mod.OUTPUT_FILENAME
    )
    assert mod._load_exp2862(tmp_path / "missing.json") == {}
    assert mod._first_token_confidence([" ", "\n"], [math.log(0.4), math.log(0.6)]) == pytest.approx(0.4)


def test_req_infer_sota_014_blocks_missing_specs_and_short_panel(tmp_path: Path) -> None:
    """REQ-INFER-SOTA-014: missing model specs or rows block before live claims."""
    missing_model_exp2862 = _exp2862_payload()
    missing_model_exp2862["selected_model_hf_id"] = ""
    missing_model_exp2862["selected_model_path"] = ""
    _write_json(tmp_path / "results" / mod.EXP2862_FILENAME, missing_model_exp2862)
    _write_jsonl(tmp_path / "data" / "eval_manifests" / "fever_20260522.jsonl", _manifest_rows())

    missing_specs = mod.run_experiment(
        ExperimentConfig(
            repo_root=tmp_path,
            output_path=tmp_path / "results" / "missing-specs.json",
            started_at=3.0,
            clock=lambda: 4.0,
        ),
        cached_pair_fn=lambda *, gpu_indices, preferred_quant: (_ for _ in ()).throw(
            RuntimeError("pair unavailable")
        ),
        panel_runner_fn=lambda **_: pytest.fail("panel should not run"),
    )

    assert missing_specs["honest_verdict"] == "blocked_no_mandated_sota_model_spec"
    assert missing_specs["blocked_metrics"] == ["blocked_no_mandated_sota_model_spec"]
    assert missing_specs["live_model_invoked"] is False

    _write_json(tmp_path / "results" / mod.EXP2862_FILENAME, _exp2862_payload())
    _write_jsonl(
        tmp_path / "data" / "eval_manifests" / "fever_20260522.jsonl",
        _manifest_rows(3),
    )

    short_panel = mod.run_experiment(
        ExperimentConfig(
            repo_root=tmp_path,
            output_path=tmp_path / "results" / "short-panel.json",
            n_examples=10,
            started_at=5.0,
            clock=lambda: 6.0,
        ),
        cached_pair_fn=lambda *, gpu_indices, preferred_quant: None,
        panel_runner_fn=lambda **_: pytest.fail("panel should not run"),
    )

    assert short_panel["honest_verdict"] == "blocked_insufficient_micro_panel_rows"
    assert short_panel["blocked_metrics"] == ["blocked_insufficient_micro_panel_rows"]


def test_req_infer_sota_014_one_class_logprobs_block_only_auroc(tmp_path: Path) -> None:
    """REQ-INFER-SOTA-014: one-class outcomes keep telemetry but block AUROC."""
    _write_json(tmp_path / "results" / mod.EXP2862_FILENAME, _exp2862_payload())
    _write_jsonl(tmp_path / "data" / "eval_manifests" / "fever_20260522.jsonl", _manifest_rows())

    def all_correct_runner(
        *,
        model_spec: dict[str, Any],
        examples: list[mod.MicroPanelExample],
        selected_python: str,
        env: dict[str, str],
        random_seed: int,
    ) -> list[dict[str, Any]]:
        del selected_python, env, random_seed
        return [
            {
                "example_id": example.example_id,
                "model_hf_id": model_spec["hf_id"],
                "model_path": model_spec["model_path"],
                "response_text": example.expected_answer,
                "tokens_generated": 1,
                "tokens": [example.expected_answer],
                "token_logprobs": [math.log(0.8)],
            }
            for example in examples
        ]

    artifact = mod.run_experiment(
        ExperimentConfig(
            repo_root=tmp_path,
            output_path=tmp_path / "results" / "one-class.json",
            n_examples=10,
            started_at=7.0,
            clock=lambda: 8.0,
        ),
        cached_pair_fn=lambda *, gpu_indices, preferred_quant: None,
        panel_runner_fn=all_correct_runner,
    )

    assert artifact["first_token_confidence_available"] is True
    assert artifact["spilled_energy_available"] is True
    assert artifact["first_token_confidence_auroc"] is None
    assert artifact["spilled_energy_auroc"] is None
    assert artifact["blocked_metrics"] == ["blocked_auroc_undefined_label_variance"]
