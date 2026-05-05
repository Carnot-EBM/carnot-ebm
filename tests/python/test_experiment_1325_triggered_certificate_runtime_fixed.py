"""Tests for Exp 1325 runtime-fixed triggered certificate rerun.

Spec: REQ-VERIFY-1325,
      SCENARIO-VERIFY-1325
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from carnot.reporting import triggered_certificate_extraction_v5_runtime_fixed_dccd_gbnf as mod


QWEN_SPEC = {
    "name": "Qwen3.6-35B-A3B",
    "hf_id": "unsloth/Qwen3.6-35B-A3B-GGUF",
    "gpu": 0,
    "model_path": "/cache/Qwen3.6-35B-A3B-Q4_K_M.gguf",
}
GEMMA_SPEC = {
    "name": "Gemma4-31B-it",
    "hf_id": "unsloth/gemma-4-31B-it-GGUF",
    "gpu": 1,
    "model_path": "/cache/gemma-4-31B-it-Q4_K_M.gguf",
}


def _exp1323(*, recovered: bool = True) -> dict[str, Any]:
    return {
        "status": "complete",
        "min_tokens_recovered": recovered,
        "models_used": [QWEN_SPEC["hf_id"], GEMMA_SPEC["hf_id"]],
        "recommended_certificate_runtime_settings": {
            "avoid_stop_strings": ["\n"],
            "chat_template": False,
            "grammar": "none_for_initial_health_gate_then_reenable_bounded_certificate_schema",
            "max_tokens": 96,
            "prompt_variant": "certificate_shaped_prompt",
            "stop": ["</s>", "<eos>"],
            "temperature": 0.0,
            "top_p": 1.0,
        },
    }


def _exp1324() -> dict[str, Any]:
    return {
        "status": "complete",
        "minimum_parseable_attempts_to_recover": 6,
        "parse_recovery_recommendation": (
            "Apply parser repair plus runtime settings and prompt schema cleanup before exp1325."
        ),
        "exp1325_fix_priorities": [
            "runtime settings",
            "prompt schema",
            "parser repair",
            "grammar coverage",
            "DCCD compact encoding with hardcoded-solution leakage guard",
        ],
    }


def _exp1312() -> dict[str, Any]:
    return {
        "status": "complete",
        "certificate_parse_rate": 0.71223,
        "certificate_truthfulness_rate": 0.69697,
    }


def _row(
    item_id: str,
    raw_output: str,
    parsed_label: str,
    verifier_label: str,
    *,
    compact_encoding: bool = False,
    token_count: int = 3,
    hf_id: str = QWEN_SPEC["hf_id"],
) -> dict[str, Any]:
    return {
        "item_id": item_id,
        "raw_output": raw_output,
        "parsed_label": parsed_label,
        "verifier_label": verifier_label,
        "expected_label": verifier_label,
        "verified": parsed_label in {verifier_label, "ABSTAIN"} if verifier_label == "UNKNOWN" else parsed_label == verifier_label,
        "compact_encoding": compact_encoding,
        "token_count": token_count,
        "generation_source": "live_sota_llamacpp",
        "hf_id": hf_id,
        "perturbation_index": 0,
    }


def _exp1311() -> dict[str, Any]:
    return {
        "status": "complete",
        "run_date": "20260505",
        "answer_stability_score": 0.9,
        "headline_result_allowed": True,
        "models_used": [QWEN_SPEC["hf_id"], GEMMA_SPEC["hf_id"]],
        "responses": [
            _row("cb_sat_schedule", "Final label: SAT", "SAT", "SAT"),
            _row("cb_unsat_capacity", "UNSAT", "UNSAT", "UNSAT", hf_id=GEMMA_SPEC["hf_id"]),
            _row(
                "cb_unknown_missing_bound",
                "Final label: UNKNOWN.",
                "UNSAT",
                "UNKNOWN",
                token_count=4,
            ),
            _row(
                "cb_compact_sat",
                "",
                "ABSTAIN",
                "SAT",
                compact_encoding=True,
                token_count=1,
                hf_id=GEMMA_SPEC["hf_id"],
            ),
        ],
    }


def test_exp1325_label_tail_parser_repairs_raw_non_json_certificates() -> None:
    """REQ-VERIFY-1325-3/5: parser repair recovers bounded label tails only."""
    parsed = mod.parse_certificate_text_v5("The answer is UNKNOWN.")
    assert parsed.parseable is True
    assert parsed.certificate["final_answer"] == "UNKNOWN"
    assert parsed.repair_kind == "label_tail"

    assert mod.parse_certificate_text_v5("").parseable is False
    assert mod.parse_certificate_text_v5("own own own").errors == ["no_json_object"]


def test_exp1325_builds_complete_runtime_fixed_artifact() -> None:
    """REQ-VERIFY-1325-2/3/4/5/6/7: rerun computes four-path verifier metrics."""
    artifact = mod.build_runtime_fixed_artifact(
        exp1311_artifact=_exp1311(),
        exp1312_artifact=_exp1312(),
        exp1323_artifact=_exp1323(),
        exp1324_artifact=_exp1324(),
        model_specs=[QWEN_SPEC, GEMMA_SPEC],
        run_date="20260505",
        project_root="/repo",
    )

    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert artifact["status"] == "complete"
    assert artifact["models_used"] == [QWEN_SPEC["hf_id"], GEMMA_SPEC["hf_id"]]
    assert artifact["runtime_settings_used"]["max_tokens"] == 96
    assert "\n" not in artifact["runtime_settings_used"]["stop"]
    assert artifact["certificate_parse_rate"] == pytest.approx(13 / 14)
    assert artifact["certificate_truthfulness_rate"] == pytest.approx(10 / 13)
    assert artifact["parse_rate_delta_over_exp1312"] == pytest.approx(round(13 / 14 - 0.71223, 6))
    assert artifact["empty_or_one_token_rate"] == pytest.approx(0.25)
    assert artifact["dccd_delta_over_grammar_only"] == pytest.approx(0.25)
    assert artifact["repair_success_rate"] == pytest.approx(1.0)
    assert artifact["grammar_projection_tax_proxy"]["rows_measured"] == 4
    assert artifact["path_metrics"]["raw_trigger"]["parseable"] == 3
    assert artifact["path_metrics"]["gbnf_constrained"]["truthful"] == 2
    assert artifact["path_metrics"]["dccd_compact"]["truthful"] == 3
    assert artifact["path_metrics"]["repaired_certificate"]["attempts"] == 2
    assert artifact["headline_result_allowed"] is True
    assert artifact["honest_verdict"] == "certificate_parse_gate_open_runtime_fixed_v5"
    assert "parser repair" in artifact["minimal_changes_applied"]


def test_exp1325_blocks_when_exp1323_token_gate_not_recovered() -> None:
    """REQ-VERIFY-1325-2: min-token recovery is a terminal prerequisite."""
    artifact = mod.build_runtime_fixed_artifact(
        exp1311_artifact=_exp1311(),
        exp1312_artifact=_exp1312(),
        exp1323_artifact=_exp1323(recovered=False),
        exp1324_artifact=_exp1324(),
        model_specs=[QWEN_SPEC, GEMMA_SPEC],
        run_date="20260505",
        project_root="/repo",
    )

    assert artifact["status"] == "blocked"
    assert artifact["blocked_reason"] == "exp1323_min_tokens_not_recovered"
    assert artifact["headline_result_allowed"] is False
    assert artifact["honest_verdict"] == "blocked_exp1323_min_tokens_not_recovered"


def test_exp1325_headline_gate_rejects_legacy_or_tiny_slices() -> None:
    """REQ-VERIFY-1325-4/7: legacy models and underpowered slices cannot be headline."""
    legacy = mod.build_runtime_fixed_artifact(
        exp1311_artifact=_exp1311(),
        exp1312_artifact=_exp1312(),
        exp1323_artifact=_exp1323(),
        exp1324_artifact=_exp1324(),
        model_specs=[{"name": "tiny", "hf_id": "legacy/small", "gpu": "cpu"}],
        run_date="20260505",
        project_root="/repo",
    )
    assert legacy["status"] == "blocked"
    assert legacy["blocked_reason"] == "cached_sota_pair_not_loadable"

    tiny = mod.build_runtime_fixed_artifact(
        exp1311_artifact=_exp1311() | {"responses": _exp1311()["responses"][:1]},
        exp1312_artifact=_exp1312(),
        exp1323_artifact=_exp1323(),
        exp1324_artifact=_exp1324(),
        model_specs=[QWEN_SPEC, GEMMA_SPEC],
        run_date="20260505",
        project_root="/repo",
        min_headline_cases=4,
    )
    assert tiny["status"] == "complete"
    assert tiny["headline_result_allowed"] is False
    assert tiny["headline_blocker"] == "insufficient_verifier_backed_cases"


def test_exp1325_run_experiment_writes_in_progress_then_final(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-VERIFY-1325-1 / SCENARIO-VERIFY-1325: output starts in-progress."""
    results = tmp_path / "results"
    results.mkdir()
    paths = {
        "exp1311_path": results / "experiment_1311.json",
        "exp1312_path": results / "experiment_1312.json",
        "exp1323_path": results / "experiment_1323.json",
        "exp1324_path": results / "experiment_1324.json",
    }
    paths["exp1311_path"].write_text(json.dumps(_exp1311()), encoding="utf-8")
    paths["exp1312_path"].write_text(json.dumps(_exp1312()), encoding="utf-8")
    paths["exp1323_path"].write_text(json.dumps(_exp1323()), encoding="utf-8")
    paths["exp1324_path"].write_text(json.dumps(_exp1324()), encoding="utf-8")

    writes: list[dict[str, Any]] = []
    real_write = mod._write_json

    def recording_write(path: Path, payload: dict[str, Any]) -> None:
        writes.append(payload)
        real_write(path, payload)

    def cached_pair(*, gpu_indices: tuple[int, int], preferred_quant: str) -> list[dict[str, Any]]:
        assert gpu_indices == (0, 1)
        assert preferred_quant == "Q4_K_M"
        return [QWEN_SPEC, GEMMA_SPEC]

    monkeypatch.setattr(mod, "_write_json", recording_write)
    output_path = results / "experiment_1325.json"

    artifact = mod.run_experiment(
        project_root=tmp_path,
        run_date="20260505",
        output_path=output_path,
        cached_pair_fn=cached_pair,
        **paths,
    )

    assert writes[0]["status"] == "in_progress"
    assert writes[-1]["status"] == "complete"
    assert json.loads(output_path.read_text(encoding="utf-8")) == artifact
