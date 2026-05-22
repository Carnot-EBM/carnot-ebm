"""Tests for Exp 2875 SOTA energy/logprob micro-panel corrigendum.

Spec: REQ-INFER-SOTA-016,
      SCENARIO-INFER-SOTA-016-001,
      SCENARIO-INFER-SOTA-016-002
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pytest

from carnot.reporting import sota_energy_micro_panel_logprob_corrigendum_v2 as mod
from carnot.reporting.sota_energy_micro_panel_logprob_corrigendum_v2 import (
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


def _exp2874_payload(model_path: Path, *, clean: bool = True) -> dict[str, Any]:
    return {
        "sota_runtime_clean": clean,
        "sota_runtime_ready_v4": clean,
        "selected_model_hf_id": GEMMA26,
        "selected_model_path": str(model_path),
        "llama_cpp_gpu_offload_verified": True,
        "model_specs": [
            {"hf_id": QWEN, "legacy_smoke_only": False},
            {"hf_id": GEMMA31, "legacy_smoke_only": False},
            {"hf_id": GEMMA26, "legacy_smoke_only": False, "model_path": str(model_path)},
        ],
        "preconditions_checked": [{"resource": "llama_cpp_gpu_offload", "available": True}],
    }


def _manifest_rows(count: int = 6) -> list[dict[str, Any]]:
    labels = ["SUPPORTS", "REFUTES", "NOT ENOUGH INFO"]
    rows: list[dict[str, Any]] = []
    for index in range(count):
        rows.append(
            {
                "dataset": "FEVER",
                "stable_id": f"exp2875-fixture-{index:02d}",
                "prompt": f"Evidence sentence {index}.",
                "claim": f"Fixture claim {index}.",
                "label_text": labels[index % len(labels)],
            }
        )
    return rows


def _prepare_inputs(tmp_path: Path, *, clean: bool = True) -> Path:
    model_path = tmp_path / "models" / "gemma-4-26B-A4B-it-UD-Q4_K_M.gguf"
    model_path.parent.mkdir(parents=True)
    model_path.write_text("tiny gguf fixture", encoding="utf-8")
    _write_json(tmp_path / "results" / mod.EXP2874_FILENAME, _exp2874_payload(model_path, clean=clean))
    _write_jsonl(tmp_path / "data" / "eval_manifests" / "fever_20260522.jsonl", _manifest_rows())
    return model_path


def _telemetry_ok(**extra: Any) -> dict[str, Any]:
    row = {
        "response_text": "SUPPORTS",
        "logprobs_available": True,
        "substitute_telemetry_used": False,
        "token_logprobs": [math.log(0.9)],
        "tokens": ["SUPPORTS"],
        "top_logprobs": [{"SUPPORTS": math.log(0.9), "REFUTES": math.log(0.1)}],
        "telemetry_source": "injected_logprobs",
    }
    row.update(extra)
    return row


def _other_answer(expected: str) -> str:
    for answer in ("SUPPORTS", "REFUTES", "UNKNOWN"):
        if answer != expected:
            return answer
    raise AssertionError("unreachable")


def _panel_runner(*, with_logprobs: bool, with_substitute: bool = False) -> mod.PanelRunnerFn:
    def run(
        *,
        model_spec: dict[str, Any],
        examples: list[mod.MicroPanelExample],
        random_seed: int,
    ) -> list[dict[str, Any]]:
        del random_seed
        rows: list[dict[str, Any]] = []
        for index, example in enumerate(examples):
            correct = index % 2 == 0
            answer = example.expected_answer if correct else _other_answer(example.expected_answer)
            row: dict[str, Any] = {
                "example_id": example.example_id,
                "model_hf_id": model_spec["hf_id"],
                "model_path": model_spec["model_path"],
                "response_text": answer,
                "tokens_generated": 1,
                "duration_s": 0.05,
            }
            if with_logprobs:
                probability = 0.9 if correct else 0.1
                row.update(
                    {
                        "tokens": [answer],
                        "token_logprobs": [math.log(probability)],
                        "top_logprobs": [
                            {answer: math.log(probability), "other": math.log(1.0 - probability)}
                        ],
                    }
                )
            elif with_substitute:
                row["substitute_score"] = 0.1 if correct else 0.9
                row["substitute_telemetry_used"] = True
                row["substitute_telemetry_source"] = "llama_cpp_logits_entropy"
            rows.append(row)
        return rows

    return run


def _config(tmp_path: Path, output_name: str = mod.OUTPUT_FILENAME) -> ExperimentConfig:
    return ExperimentConfig(
        repo_root=tmp_path,
        output_path=tmp_path / "results" / output_name,
        run_date="20260522",
        n_prompts=6,
        started_at=10.0,
        clock=lambda: 12.5,
        tests_run=[
            "pytest tests/python/test_experiment_2875_sota_energy_micro_panel_logprob_corrigendum_v2.py"
        ],
    )


def test_scenario_infer_sota_016_clean_logprob_panel_writes_required_artifact(
    tmp_path: Path,
) -> None:
    """SCENARIO-INFER-SOTA-016-001: non-empty rows plus logprob telemetry are clean."""
    _prepare_inputs(tmp_path)

    artifact = mod.run_experiment(
        _config(tmp_path),
        telemetry_probe_fn=lambda **_: _telemetry_ok(),
        panel_runner_fn=_panel_runner(with_logprobs=True),
    )

    assert json.loads((_config(tmp_path).output_path).read_text(encoding="utf-8")) == artifact
    assert set(REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert artifact["honest_verdict"] == "micro_panel_clean_no_benchmark_claim"
    assert artifact["micro_panel_clean"] is True
    assert artifact["blocked_reason"] == ""
    assert artifact["selected_model_hf_id"] == GEMMA26
    assert artifact["n_prompts"] == 6
    assert artifact["n_nonempty_responses"] == 6
    assert artifact["logprobs_available"] is True
    assert artifact["substitute_telemetry_used"] is False
    assert artifact["auroc_if_computable"] == pytest.approx(1.0)
    assert artifact["benchmark_claim_made"] is False
    assert artifact["tests_run"] == [
        "pytest tests/python/test_experiment_2875_sota_energy_micro_panel_logprob_corrigendum_v2.py"
    ]
    assert artifact["duration_s"] == pytest.approx(2.5)
    assert len(artifact["prompt_rows"]) == 6
    assert all(row["response_nonempty"] for row in artifact["prompt_rows"])
    assert all(row["logprobs_available"] for row in artifact["prompt_rows"])


def test_scenario_infer_sota_016_missing_preflight_telemetry_blocks_before_panel(
    tmp_path: Path,
) -> None:
    """SCENARIO-INFER-SOTA-016-002: missing llama.cpp telemetry blocks honestly."""
    _prepare_inputs(tmp_path)

    artifact = mod.run_experiment(
        _config(tmp_path, "missing-preflight.json"),
        telemetry_probe_fn=lambda **_: {
            "response_text": "SUPPORTS",
            "logprobs_available": False,
            "substitute_telemetry_used": False,
            "blocked_reason": "blocked_logprobs_unavailable",
            "telemetry_source": "injected_text_only",
        },
        panel_runner_fn=lambda **_: pytest.fail("panel must not run without telemetry"),
    )

    assert artifact["honest_verdict"] == "blocked_logprobs_unavailable"
    assert artifact["micro_panel_clean"] is False
    assert artifact["blocked_reason"] == "blocked_logprobs_unavailable"
    assert artifact["n_prompts"] == 0
    assert artifact["n_nonempty_responses"] == 0
    assert artifact["prompt_rows"] == []
    assert artifact["benchmark_claim_made"] is False
    assert artifact["preconditions_checked"][-1]["resource"] == "llama_cpp_logprob_or_substitute_telemetry"
    assert artifact["preconditions_checked"][-1]["available"] is False


def test_req_infer_sota_016_text_only_panel_rows_are_not_clean(
    tmp_path: Path,
) -> None:
    """REQ-INFER-SOTA-016: non-empty text without per-row telemetry remains blocked."""
    _prepare_inputs(tmp_path)

    artifact = mod.run_experiment(
        _config(tmp_path, "text-only-panel.json"),
        telemetry_probe_fn=lambda **_: _telemetry_ok(
            logprobs_available=False,
            substitute_telemetry_used=True,
            substitute_score=0.42,
            telemetry_source="llama_cpp_logits_entropy",
            token_logprobs=[],
        ),
        panel_runner_fn=_panel_runner(with_logprobs=False),
    )

    assert artifact["honest_verdict"] == "blocked_logprobs_unavailable"
    assert artifact["micro_panel_clean"] is False
    assert artifact["blocked_reason"] == "blocked_logprobs_unavailable"
    assert artifact["n_prompts"] == 6
    assert artifact["n_nonempty_responses"] == 6
    assert artifact["logprobs_available"] is False
    assert artifact["substitute_telemetry_used"] is False
    assert artifact["auroc_if_computable"] is None
    assert all(row["telemetry_score"] is None for row in artifact["prompt_rows"])


def test_req_infer_sota_016_empty_rows_have_exact_blocked_verdict(tmp_path: Path) -> None:
    """REQ-INFER-SOTA-016: logprobs do not rescue empty response rows."""
    _prepare_inputs(tmp_path)

    def empty_logprob_panel(
        *,
        model_spec: dict[str, Any],
        examples: list[mod.MicroPanelExample],
        random_seed: int,
    ) -> list[dict[str, Any]]:
        del random_seed
        return [
            {
                "example_id": example.example_id,
                "model_hf_id": model_spec["hf_id"],
                "model_path": model_spec["model_path"],
                "response_text": "" if index else example.expected_answer,
                "tokens_generated": 1,
                "tokens": [example.expected_answer],
                "token_logprobs": [math.log(0.8)],
            }
            for index, example in enumerate(examples)
        ]

    artifact = mod.run_experiment(
        _config(tmp_path, "empty-logprob-panel.json"),
        telemetry_probe_fn=lambda **_: _telemetry_ok(),
        panel_runner_fn=empty_logprob_panel,
    )

    assert artifact["honest_verdict"] == "blocked_empty_responses"
    assert artifact["blocked_reason"] == "blocked_empty_responses"
    assert artifact["micro_panel_clean"] is False
    assert artifact["n_nonempty_responses"] == 1
    assert artifact["logprobs_available"] is True
    assert artifact["benchmark_claim_made"] is False


@pytest.mark.parametrize(
    ("mutate", "blocked_reason"),
    [
        (lambda payload, _path: payload.update({"sota_runtime_clean": False}), "blocked_exp2874_sota_runtime_not_clean"),
        (
            lambda payload, _path: payload.update({"selected_model_hf_id": "Qwen/Qwen3.5-0.8B"}),
            "blocked_selected_model_not_mandated",
        ),
        (
            lambda payload, _path: payload.update({"selected_model_path": str(_path.with_suffix('.missing'))}),
            "blocked_selected_model_path_missing",
        ),
        (
            lambda payload, _path: payload.update({"llama_cpp_gpu_offload_verified": False}),
            "blocked_llama_cpp_gpu_offload",
        ),
    ],
)
def test_req_infer_sota_016_preconditions_block_before_live_calls(
    tmp_path: Path,
    mutate: Any,
    blocked_reason: str,
) -> None:
    """REQ-INFER-SOTA-016: Exp 2874, model, path, and GPU gates are preconditions."""
    model_path = _prepare_inputs(tmp_path)
    exp_path = tmp_path / "results" / mod.EXP2874_FILENAME
    payload = json.loads(exp_path.read_text(encoding="utf-8"))
    mutate(payload, model_path)
    _write_json(exp_path, payload)

    artifact = mod.run_experiment(
        _config(tmp_path, f"{blocked_reason}.json"),
        telemetry_probe_fn=lambda **_: pytest.fail("telemetry probe should not run"),
        panel_runner_fn=lambda **_: pytest.fail("panel should not run"),
    )

    assert artifact["honest_verdict"] == blocked_reason
    assert artifact["blocked_reason"] == blocked_reason
    assert artifact["micro_panel_clean"] is False
    assert artifact["selected_model_hf_id"] == str(payload.get("selected_model_hf_id") or "")
    assert artifact["benchmark_claim_made"] is False


def test_req_infer_sota_016_substitute_scores_can_support_clean_diagnostics(
    tmp_path: Path,
) -> None:
    """REQ-INFER-SOTA-016: documented substitute telemetry can score rows."""
    _prepare_inputs(tmp_path)

    artifact = mod.run_experiment(
        _config(tmp_path, "substitute-panel.json"),
        telemetry_probe_fn=lambda **_: _telemetry_ok(
            logprobs_available=False,
            substitute_telemetry_used=True,
            substitute_score=0.1,
            telemetry_source="llama_cpp_logits_entropy",
            token_logprobs=[],
        ),
        panel_runner_fn=_panel_runner(with_logprobs=False, with_substitute=True),
    )

    assert artifact["micro_panel_clean"] is True
    assert artifact["logprobs_available"] is False
    assert artifact["substitute_telemetry_used"] is True
    assert artifact["auroc_if_computable"] == pytest.approx(1.0)
    assert all(row["substitute_telemetry_used"] for row in artifact["prompt_rows"])


def test_req_infer_sota_016_telemetry_helpers_parse_logprobs_and_logits() -> None:
    """REQ-INFER-SOTA-016: llama.cpp telemetry is normalized without fake values."""
    raw = {
        "choices": [
            {
                "text": " SUPPORTS",
                "logprobs": {
                    "tokens": [" ", "SUPPORTS"],
                    "token_logprobs": [None, True, math.log(0.8)],
                    "top_logprobs": [
                        {"SUPPORTS": math.log(0.8), "REFUTES": math.log(0.2), "skip": False}
                    ],
                },
            }
        ],
        "usage": {"completion_tokens": 2},
    }

    parsed = mod.extract_completion_telemetry(raw)
    assert parsed["response_text"] == " SUPPORTS"
    assert parsed["tokens"] == [" ", "SUPPORTS"]
    assert parsed["token_logprobs"] == [pytest.approx(math.log(0.8))]
    assert parsed["top_logprobs"] == [
        {"SUPPORTS": pytest.approx(math.log(0.8)), "REFUTES": pytest.approx(math.log(0.2))}
    ]
    assert mod.first_token_confidence([" ", "SUPPORTS"], [math.log(0.25), math.log(0.8)]) == pytest.approx(0.8)
    assert mod.logits_entropy_score([2.0, 0.0]) == pytest.approx(0.365334, abs=1e-6)
    assert mod.compute_auroc([1, 0], [0.9, 0.1]) == pytest.approx(1.0)
    assert mod.compute_auroc([1, 1], [0.2, 0.3]) is None


def test_req_infer_sota_016_helper_edge_cases_and_short_panel_block(tmp_path: Path) -> None:
    """REQ-INFER-SOTA-016: helper edge cases stay deterministic and honest."""
    _prepare_inputs(tmp_path)

    short_panel = mod.run_experiment(
        ExperimentConfig(
            repo_root=tmp_path,
            output_path=tmp_path / "results" / "short-panel.json",
            run_date="20260522",
            n_prompts=3,
            manifest_paths=(Path("data/eval_manifests/missing.jsonl"),),
            started_at=1.0,
            clock=lambda: 2.0,
        ),
        telemetry_probe_fn=lambda **_: _telemetry_ok(),
        panel_runner_fn=lambda **_: pytest.fail("panel should not run without local labels"),
    )
    assert short_panel["blocked_reason"] == "blocked_insufficient_micro_panel_rows"
    assert short_panel["n_prompts"] == 0

    default_config = ExperimentConfig(repo_root=tmp_path)
    assert default_config.resolved_output_path() == tmp_path / "results" / mod.OUTPUT_FILENAME
    assert default_config.resolved_exp2874_path() == tmp_path / "results" / mod.EXP2874_FILENAME
    assert mod.normalize_expected_answer("maybe") == "MAYBE"
    assert mod.classify_response("") is None
    assert mod.classify_response("Not enough info.") == "UNKNOWN"
    assert mod.classify_response("I cannot tell") is None
    assert mod.compute_auroc([], []) is None
    with pytest.raises(ValueError, match="same length"):
        mod.compute_auroc([1], [0.1, 0.2])
    with pytest.raises(ValueError, match="finite"):
        mod.compute_auroc([1, 0], [math.nan, 0.2])
    assert mod.extract_completion_telemetry({"choices": [{"message": {"content": "ok"}}]}) == {
        "response_text": "ok",
        "completion_tokens": 0,
        "tokens": [],
        "token_logprobs": [],
        "top_logprobs": [],
    }
    assert mod.extract_completion_telemetry(None)["response_text"] == ""
    assert mod.first_token_confidence([" ", "\n"], [math.log(0.4), math.log(0.6)]) == pytest.approx(0.4)
    assert mod.first_token_confidence([], []) is None
    assert mod.logits_entropy_score([]) is None
    selected = mod.select_micro_panel(
        tmp_path,
        n_prompts=10,
        manifest_paths=(Path("data/eval_manifests/missing.jsonl"), Path("data/eval_manifests/fever_20260522.jsonl")),
    )
    assert len(selected) == 6
    assert mod._read_json(tmp_path / "missing.json") == {}
