"""Tests for Exp 1323 SOTA GGUF token-health diagnostic.

Spec: REQ-VERIFY-1323,
      SCENARIO-VERIFY-1323
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from carnot.reporting import sota_gguf_token_health_prompt_runtime_diagnostic as mod
from carnot.reporting.sota_gguf_token_health_prompt_runtime_diagnostic import (
    REQUIRED_ARTIFACT_FIELDS,
    PromptVariant,
    RawProbeGeneration,
    build_prompt_variants,
    build_token_health_artifact,
    run_experiment,
)


QWEN_SPEC = {
    "name": "Qwen3.6-35B-A3B",
    "hf_id": "unsloth/Qwen3.6-35B-A3B-GGUF",
    "gpu": 0,
    "model_path": "/cache/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf",
}
GEMMA_SPEC = {
    "name": "Gemma4-31B-it",
    "hf_id": "unsloth/gemma-4-31B-it-GGUF",
    "gpu": 1,
    "model_path": "/cache/gemma-4-31B-it-Q4_K_M.gguf",
}


def _cached_pair(*, gpu_indices: tuple[int, int], preferred_quant: str) -> list[dict[str, Any]]:
    assert gpu_indices == (0, 1)
    assert preferred_quant == "Q4_K_M"
    return [dict(QWEN_SPEC), dict(GEMMA_SPEC)]


def _write_prior_artifacts(root: Path) -> None:
    results = root / "results"
    results.mkdir()
    (results / "experiment_1310_sota_gguf_llamacpp_smoke_load.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "models_used": [QWEN_SPEC["hf_id"], GEMMA_SPEC["hf_id"]],
                "per_model_results": [
                    {"hf_id": QWEN_SPEC["hf_id"], "token_count": 4, "error": None},
                    {"hf_id": GEMMA_SPEC["hf_id"], "token_count": 1, "error": None},
                ],
            }
        ),
        encoding="utf-8",
    )
    (results / "experiment_1311_sota_constraintbench_satquest_answer_stability.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "models_used": [QWEN_SPEC["hf_id"], GEMMA_SPEC["hf_id"]],
                "responses": [
                    {
                        "hf_id": QWEN_SPEC["hf_id"],
                        "raw_output": "",
                        "token_count": 1,
                        "error": None,
                    },
                    {
                        "hf_id": GEMMA_SPEC["hf_id"],
                        "raw_output": " SAT",
                        "token_count": 2,
                        "error": None,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    (results / "experiment_1312_triggered_certificate_extraction_dccd_gbnf.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "models_used": [QWEN_SPEC["hf_id"], GEMMA_SPEC["hf_id"]],
                "certificate_parse_rate": 0.71223,
                "path_metrics": {"raw_trigger": {"parseable_rate": 0.0}},
                "attempts": [
                    {
                        "hf_id": QWEN_SPEC["hf_id"],
                        "path": "raw_trigger",
                        "prompt_chars": 173,
                        "parseable": False,
                    },
                    {
                        "hf_id": GEMMA_SPEC["hf_id"],
                        "path": "dccd_compact",
                        "prompt_chars": 72,
                        "parseable": True,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )


def test_exp1323_prompt_grid_contains_required_runtime_variants() -> None:
    """REQ-VERIFY-1323-4/5: the diagnostic grid covers the mandated causes."""
    variants = build_prompt_variants()

    assert [variant.name for variant in variants] == [
        "baseline_prompt",
        "chat_template_prompt",
        "no_stop_string_prompt",
        "larger_max_token_budget",
        "certificate_shaped_prompt",
    ]
    assert variants[0].stop == ["\n", "</s>", "<eos>"]
    assert variants[1].use_chat_template is True
    assert variants[2].stop == []
    assert variants[3].max_tokens > variants[0].max_tokens
    assert variants[4].certificate_shaped is True
    assert variants[4].grammar == "none"


def test_exp1323_builds_complete_artifact_with_recovered_certificate_tokens(
    tmp_path: Path,
) -> None:
    """REQ-VERIFY-1323-2/3/5/6/7/8: recovered certificate output gates headline use."""
    _write_prior_artifacts(tmp_path)

    def generation_fn(spec: dict[str, Any], variant: PromptVariant) -> RawProbeGeneration:
        if variant.certificate_shaped and spec["hf_id"] == QWEN_SPEC["hf_id"]:
            return RawProbeGeneration(text="", token_count=1, elapsed_seconds=0.1)
        if variant.certificate_shaped:
            return RawProbeGeneration(
                text='{"label":"SAT","constraints":["x1"],"verifier":"cnf"}',
                token_count=9,
                elapsed_seconds=0.2,
                raw_response={
                    "choices": [
                        {
                            "logprobs": {
                                "token_logprobs": [-0.2, -0.4],
                                "top_logprobs": [
                                    {" A": -0.2, " B": -1.2},
                                    {" C": -0.4, " D": -2.0},
                                ],
                            }
                        }
                    ]
                },
            )
        if variant.name == "baseline_prompt":
            return RawProbeGeneration(text="", token_count=1, elapsed_seconds=0.05)
        return RawProbeGeneration(
            text="SAT because x1 satisfies the fixture.",
            token_count=6,
            elapsed_seconds=0.05,
            used_chat_template=variant.use_chat_template,
        )

    artifact = build_token_health_artifact(
        project_root=tmp_path,
        run_date="20260505",
        cached_pair_fn=_cached_pair,
        generation_fn=generation_fn,
        generation_source="live_sota_llamacpp",
    )

    assert set(REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert artifact["status"] == "complete"
    assert artifact["models_used"] == [QWEN_SPEC["hf_id"], GEMMA_SPEC["hf_id"]]
    assert artifact["prompt_variants_tested"] == [variant.name for variant in build_prompt_variants()]
    assert artifact["prior_artifact_context"]["exp1311"]["empty_or_one_token_rate"] == 0.5
    assert artifact["empty_or_one_token_rate"] == pytest.approx(0.3)
    assert artifact["min_tokens_recovered"] is True
    assert artifact["topk_logprob_available"] is True
    assert artifact["entropy_production_rate_available"] is True
    assert artifact["token_health_summary"]["token_logprob_count"] == 2
    assert artifact["certificate_parse_delta_with_probe_gate"]["baseline_exp1312_certificate_parse_rate"] == 0.71223
    assert artifact["certificate_parse_delta_with_probe_gate"]["probe_gate_certificate_skeleton_rate"] == 0.5
    assert artifact["recommended_certificate_runtime_settings"]["avoid_stop_strings"] == ["\n"]
    assert artifact["recommended_certificate_runtime_settings"]["max_tokens"] >= 96
    assert artifact["headline_result_allowed"] is True
    assert artifact["honest_verdict"] == "token_health_recovered_certificate_prompt_multitoken"
    cert_rows = [
        row for row in artifact["per_variant_results"] if row["variant"] == "certificate_shaped_prompt"
    ]
    assert [row["certificate_skeleton_available"] for row in cert_rows] == [False, True]


def test_exp1323_marks_logprob_entropy_unavailable_without_runtime_payload(
    tmp_path: Path,
) -> None:
    """REQ-VERIFY-1323-6: missing top-k data is explicit rather than inferred."""
    _write_prior_artifacts(tmp_path)

    artifact = build_token_health_artifact(
        project_root=tmp_path,
        run_date="20260505",
        cached_pair_fn=_cached_pair,
        generation_fn=lambda _spec, _variant: RawProbeGeneration(
            text="SAT because x1 satisfies the fixture.",
            token_count=6,
            elapsed_seconds=0.1,
        ),
        generation_source="live_sota_llamacpp",
    )

    assert artifact["topk_logprob_available"] is False
    assert artifact["entropy_production_rate_available"] is False
    assert artifact["token_health_summary"]["missing_api_reason"] == "llama_cpp_response_missing_top_logprobs"

    unrecovered = build_token_health_artifact(
        project_root=tmp_path,
        run_date="20260505",
        cached_pair_fn=_cached_pair,
        generation_fn=lambda _spec, _variant: RawProbeGeneration(
            text="",
            token_count=1,
            elapsed_seconds=0.1,
        ),
        generation_source="live_sota_llamacpp",
    )

    assert unrecovered["min_tokens_recovered"] is False
    assert (
        unrecovered["recommended_certificate_runtime_settings"]["status"]
        == "blocked_until_certificate_shaped_prompt_emits_multi_token_output"
    )
    assert unrecovered["headline_result_allowed"] is False


def test_exp1323_cached_pair_and_import_blockers_are_terminal(tmp_path: Path) -> None:
    """REQ-VERIFY-1323-3/7: missing models or llama.cpp runtime cannot be headline data."""
    _write_prior_artifacts(tmp_path)
    missing = build_token_health_artifact(
        project_root=tmp_path,
        run_date="20260505",
        cached_pair_fn=lambda *, gpu_indices, preferred_quant: None,
        generation_fn=lambda _spec, _variant: pytest.fail("generation must not run"),
    )

    assert missing["status"] == "blocked"
    assert missing["blocked_reason"] == "cached_sota_pair_not_loadable"
    assert missing["headline_result_allowed"] is False
    assert missing["honest_verdict"] == "blocked_cached_sota_pair_not_loadable"

    import_blocked = build_token_health_artifact(
        project_root=tmp_path,
        run_date="20260505",
        cached_pair_fn=_cached_pair,
        llama_importer=lambda: (False, None, "ImportError: no module named llama_cpp"),
        generation_fn=None,
    )

    assert import_blocked["status"] == "blocked"
    assert import_blocked["models_used"] == [QWEN_SPEC["hf_id"], GEMMA_SPEC["hf_id"]]
    assert import_blocked["llama_cpp_import_ok"] is False
    assert import_blocked["llama_cpp_import_error"] == "ImportError: no module named llama_cpp"
    assert import_blocked["honest_verdict"] == "blocked_llama_cpp_import_failed"


def test_exp1323_completion_helpers_parse_text_chat_and_token_fallbacks() -> None:
    """REQ-VERIFY-1323-5/6: nonstandard llama.cpp payloads still yield bounded rows."""

    class TokenizingLlama:
        def tokenize(self, data: bytes, *, add_bos: bool) -> list[int]:
            assert data == b"SAT"
            assert add_bos is False
            return [1, 2]

    class FailingTokenizer:
        def tokenize(self, _data: bytes, *, add_bos: bool) -> list[int]:
            assert add_bos is False
            raise RuntimeError("tokenizer failed")

    assert mod._completion_text({"choices": [{"text": "SAT"}]}) == "SAT"
    assert mod._completion_text({"choices": [{"message": {"content": "UNSAT"}}]}) == "UNSAT"
    assert mod._completion_text({"choices": [{"message": "not-a-dict"}]}) == ""
    assert mod._completion_text("UNKNOWN") == "UNKNOWN"
    assert mod._completion_text({"choices": []}) == ""
    assert mod._completion_token_count({"usage": {"completion_tokens": 0}}, "SAT", object()) == 0
    assert mod._completion_token_count({}, "SAT", TokenizingLlama()) == 2
    assert mod._completion_token_count({}, "SAT label", FailingTokenizer()) == 2
    assert (
        mod._certificate_skeleton_available(
            '{"label":"SAT","constraints":[],"verifier":"cnf"}'
        )
        is True
    )
    assert mod._certificate_skeleton_available("SAT only") is False
    assert mod._read_json(Path("/tmp/carnot-exp1323-definitely-missing.json")) == {}
    assert mod._resolved_specs([{**QWEN_SPEC, "hf_id": "legacy/small"}, GEMMA_SPEC]) == []
    assert mod._extract_logprobs({"choices": [None]}) == {"token_logprobs": [], "top_logprobs": []}
    assert mod._extract_logprobs({"choices": [{"logprobs": "bad"}]}) == {
        "token_logprobs": [],
        "top_logprobs": [],
    }


def test_exp1323_live_llama_path_uses_imported_runner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-VERIFY-1323-3/5: live path delegates to the llama.cpp collection runner."""
    _write_prior_artifacts(tmp_path)

    class FakeLlama:
        pass

    def fake_collect(
        specs: list[dict[str, Any]],
        variants: list[PromptVariant],
        *,
        llama_class: type[Any],
    ) -> list[dict[str, Any]]:
        assert llama_class is FakeLlama
        return [
            mod._row_from_generation(
                specs[0],
                variants[-1],
                RawProbeGeneration(text="multi token certificate", token_count=3),
                "live_sota_llamacpp",
            )
        ]

    monkeypatch.setattr(mod, "_collect_with_llama", fake_collect)

    artifact = build_token_health_artifact(
        project_root=tmp_path,
        run_date="20260505",
        cached_pair_fn=_cached_pair,
        llama_importer=lambda: (True, FakeLlama, None),
        generation_fn=None,
    )

    assert artifact["llama_cpp_import_ok"] is True
    assert artifact["min_tokens_recovered"] is True


def test_exp1323_run_experiment_writes_in_progress_then_final(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-VERIFY-1323-1 / SCENARIO-VERIFY-1323: artifact starts in-progress."""
    _write_prior_artifacts(tmp_path)
    writes: list[dict[str, Any]] = []
    real_write = mod._write_json

    def recording_write(path: Path, payload: dict[str, Any]) -> None:
        writes.append(payload)
        real_write(path, payload)

    monkeypatch.setattr(mod, "_write_json", recording_write)
    output_path = tmp_path / "results" / "experiment_1323.json"

    artifact = run_experiment(
        project_root=tmp_path,
        run_date="20260505",
        output_path=output_path,
        cached_pair_fn=_cached_pair,
        generation_fn=lambda _spec, _variant: RawProbeGeneration(
            text="SAT because x1 satisfies the fixture.",
            token_count=6,
            elapsed_seconds=0.1,
        ),
        generation_source="live_sota_llamacpp",
    )
    written = json.loads(output_path.read_text(encoding="utf-8"))

    assert writes[0]["status"] == "in_progress"
    assert writes[-1]["status"] == "complete"
    assert written == artifact
