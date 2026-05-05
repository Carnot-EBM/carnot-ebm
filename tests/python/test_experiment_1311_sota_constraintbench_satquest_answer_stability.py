"""Tests for Exp 1311 SOTA ConstraintBench/SATQuest answer stability.

Spec: REQ-VERIFY-1311,
      SCENARIO-VERIFY-1311
"""

from __future__ import annotations

import builtins
import json
import sys
import types
from pathlib import Path
from typing import Any

import pytest

from carnot.reporting import sota_constraintbench_satquest_answer_stability as mod
from carnot.reporting.sota_constraintbench_satquest_answer_stability import (
    REQUIRED_ARTIFACT_FIELDS,
    MicroItem,
    RawGeneration,
    build_answer_stability_artifact,
    build_micro_slice,
    parse_final_label,
    run_experiment,
    verify_item_label,
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


def test_exp1311_default_micro_slice_has_required_fixture_mix() -> None:
    """REQ-VERIFY-1311-3: fixtures cover families, labels, and compact encoding."""
    items = build_micro_slice()

    assert len(items) == 10
    assert sum(item.family == "constraintbench" for item in items) == 5
    assert sum(item.family == "satquest" for item in items) == 5
    assert {item.expected_label for item in items} == {"SAT", "UNSAT", "UNKNOWN"}
    assert any(item.compact_encoding for item in items)
    assert all(item.item_id and item.prompt for item in items)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("SAT", "SAT"),
        ("final: satisfiable", "SAT"),
        ("UNSAT", "UNSAT"),
        ("unsatisfiable by unit conflict", "UNSAT"),
        ("unknown", "UNKNOWN"),
        ("I must abstain", "ABSTAIN"),
        ("SAT or UNSAT", "ABSTAIN"),
        ("no bounded label here", "ABSTAIN"),
    ],
)
def test_exp1311_parse_final_label_is_bounded(text: str, expected: str) -> None:
    """REQ-VERIFY-1311-4: parser accepts only bounded final labels."""
    assert parse_final_label(text) == expected


def test_exp1311_verifier_labels_sat_unsat_unknown_and_fallback() -> None:
    """REQ-VERIFY-1311-5: Z3-style verification labels deterministic fixtures."""
    sat_item = MicroItem(
        item_id="sat",
        family="satquest",
        prompt="(x1)",
        expected_label="SAT",
        num_variables=1,
        clauses=((1,),),
    )
    unsat_item = MicroItem(
        item_id="unsat",
        family="constraintbench",
        prompt="x1 and not x1",
        expected_label="UNSAT",
        num_variables=1,
        clauses=((1,), (-1,)),
    )
    unknown_item = MicroItem(
        item_id="unknown",
        family="constraintbench",
        prompt="missing capacity bound",
        expected_label="UNKNOWN",
        num_variables=0,
        clauses=None,
    )

    assert verify_item_label(sat_item, "SAT", backend="z3").verified is True
    assert verify_item_label(sat_item, "UNSAT", backend="pure_python").verified is False
    assert verify_item_label(unsat_item, "UNSAT", backend="z3").verified is True
    assert verify_item_label(unsat_item, "SAT", backend="pure_python").verified is False
    unknown_result = verify_item_label(unknown_item, "UNKNOWN", backend="pure_python")
    assert unknown_result.verified is True
    assert unknown_result.verifier_label == "UNKNOWN"


def test_exp1311_verifier_falls_back_when_z3_backend_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-VERIFY-1311-5: pure-Python CNF fallback handles unavailable Z3."""
    item = MicroItem("sat", "satquest", "x1", "SAT", 1, ((1,),))

    def fail_z3(_item: MicroItem) -> bool:
        raise RuntimeError("z3 unavailable")

    monkeypatch.setattr(mod, "_z3_cnf_sat", fail_z3)

    result = verify_item_label(item, "SAT")

    assert result.verified is True
    assert result.verifier_backend == "pure_python_cnf"


def test_exp1311_completion_helpers_cover_nonstandard_outputs() -> None:
    """REQ-VERIFY-1311-4: nonstandard llama outputs still produce bounded rows."""

    class TokenizingLlama:
        def tokenize(self, data: bytes, *, add_bos: bool) -> list[int]:
            assert data == b"SAT"
            assert add_bos is False
            return [1, 2]

    class BadTokenizingLlama:
        def tokenize(self, _data: bytes, *, add_bos: bool) -> list[int]:
            assert add_bos is False
            raise RuntimeError("tokenizer failed")

    assert mod._completion_text(" SAT") == " SAT"
    assert mod._completion_text({"choices": []}) == ""
    assert mod._completion_token_count({}, "SAT", TokenizingLlama()) == 2
    assert mod._completion_token_count({}, "SAT label", BadTokenizingLlama()) == 2


def test_exp1311_import_llama_class_success_and_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """REQ-VERIFY-1311-7: llama.cpp import probe records success and failure."""

    class FakeLlama:
        pass

    fake_module = types.SimpleNamespace(Llama=FakeLlama)
    monkeypatch.setitem(sys.modules, "llama_cpp", fake_module)
    assert mod._import_llama_class() == (True, FakeLlama, None)

    monkeypatch.delitem(sys.modules, "llama_cpp", raising=False)
    real_import = builtins.__import__

    def guarded_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "llama_cpp":
            raise ModuleNotFoundError("no module named llama_cpp")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    ok, llama_class, error = mod._import_llama_class()

    assert ok is False
    assert llama_class is None
    assert error == "ModuleNotFoundError: no module named llama_cpp"


def test_exp1311_metric_artifact_uses_stability_disagreement_and_verification(
    tmp_path: Path,
) -> None:
    """REQ-VERIFY-1311-6/7: artifact exposes required metrics and headline gate."""
    items = [
        MicroItem("sat", "satquest", "x1", "SAT", 1, ((1,),)),
        MicroItem("unsat", "constraintbench", "x1 and not x1", "UNSAT", 1, ((1,), (-1,))),
    ]
    scripted = {
        ("Qwen3.6-35B-A3B", "sat", 0): "SAT",
        ("Qwen3.6-35B-A3B", "sat", 1): "SAT",
        ("Qwen3.6-35B-A3B", "unsat", 0): "UNSAT",
        ("Qwen3.6-35B-A3B", "unsat", 1): "UNSAT",
        ("Gemma4-31B-it", "sat", 0): "SAT",
        ("Gemma4-31B-it", "sat", 1): "UNSAT",
        ("Gemma4-31B-it", "unsat", 0): "UNKNOWN",
        ("Gemma4-31B-it", "unsat", 1): "UNKNOWN",
    }

    def generation_fn(
        spec: dict[str, Any],
        item: MicroItem,
        perturbation_index: int,
        prompt: str,
        max_tokens: int,
    ) -> RawGeneration:
        assert prompt
        assert max_tokens == 6
        return RawGeneration(
            text=scripted[(spec["name"], item.item_id, perturbation_index)], token_count=1
        )

    artifact = build_answer_stability_artifact(
        project_root=tmp_path,
        run_date="20260505",
        cached_pair_fn=_cached_pair,
        generation_fn=generation_fn,
        generation_source="live_sota_llamacpp",
        items=items,
    )

    assert set(REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert artifact["status"] == "complete"
    assert artifact["models_used"] == [QWEN_SPEC["hf_id"], GEMMA_SPEC["hf_id"]]
    assert artifact["constraintbench_items"] == 1
    assert artifact["satquest_items"] == 1
    assert artifact["answer_stability_score"] == pytest.approx(0.75)
    assert artifact["cross_model_disagreement_rate"] == pytest.approx(1.0)
    assert artifact["pysat_verified_rate"] == pytest.approx(0.625)
    assert artifact["feasibility_rate"] == pytest.approx(0.75)
    assert artifact["unknown_or_abstain_rate"] == pytest.approx(0.25)
    assert artifact["headline_result_allowed"] is True
    assert artifact["honest_verdict"] == "sota_constraint_satquest_stability_audit_complete"
    assert len(artifact["responses"]) == 8


def test_exp1311_meaningful_disagreement_is_counted(tmp_path: Path) -> None:
    """REQ-VERIFY-1311-6: SAT-vs-UNSAT cross-model splits are meaningful."""
    item = MicroItem("sat", "satquest", "x1", "SAT", 1, ((1,),))

    def generation_fn(
        spec: dict[str, Any],
        _item: MicroItem,
        _perturbation_index: int,
        _prompt: str,
        _max_tokens: int,
    ) -> RawGeneration:
        label = "SAT" if spec["name"] == "Qwen3.6-35B-A3B" else "UNSAT"
        return RawGeneration(text=label, token_count=1)

    artifact = build_answer_stability_artifact(
        project_root=tmp_path,
        run_date="20260505",
        cached_pair_fn=_cached_pair,
        generation_fn=generation_fn,
        generation_source="live_sota_llamacpp",
        items=[item],
    )

    assert artifact["cross_model_disagreement_items"] == 1
    assert artifact["meaningful_disagreement_items"] == 1
    assert artifact["meaningful_disagreement_rate"] == 1.0


def test_exp1311_cached_pair_blocker_is_terminal_non_headline(tmp_path: Path) -> None:
    """REQ-VERIFY-1311-2: missing local SOTA pair aborts without fake data."""
    artifact = build_answer_stability_artifact(
        project_root=tmp_path,
        run_date="20260505",
        cached_pair_fn=lambda *, gpu_indices, preferred_quant: None,
        generation_fn=lambda *_args, **_kwargs: pytest.fail("generation must not run"),
    )

    assert artifact["status"] == "blocked"
    assert artifact["blocked_reason"] == "cached_sota_pair_not_loadable"
    assert artifact["headline_result_allowed"] is False
    assert artifact["honest_verdict"] == "blocked_cached_sota_pair_not_loadable"
    assert artifact["models_used"] == []


def test_exp1311_cached_pair_exception_is_terminal_non_headline(tmp_path: Path) -> None:
    """REQ-VERIFY-1311-2: resolver exceptions become explicit blockers."""

    def bad_cached_pair(**_kwargs: Any) -> list[dict[str, Any]]:
        raise RuntimeError("resolver exploded")

    artifact = build_answer_stability_artifact(
        project_root=tmp_path,
        run_date="20260505",
        cached_pair_fn=bad_cached_pair,
    )

    assert artifact["status"] == "blocked"
    assert artifact["blocked_reason"] == "cached_sota_pair_exception"
    assert artifact["cached_sota_pair_error"] == "RuntimeError: resolver exploded"
    assert artifact["honest_verdict"] == "blocked_cached_sota_pair_exception"


def test_exp1311_invalid_cached_pair_shape_blocks(tmp_path: Path) -> None:
    """REQ-VERIFY-1311-2: non-mandated or pathless specs cannot be headline data."""
    artifact = build_answer_stability_artifact(
        project_root=tmp_path,
        run_date="20260505",
        cached_pair_fn=lambda *, gpu_indices, preferred_quant: [
            {**QWEN_SPEC, "hf_id": "legacy/small-model"},
            {**GEMMA_SPEC, "model_path": None},
        ],
    )

    assert artifact["status"] == "blocked"
    assert artifact["blocked_reason"] == "cached_sota_pair_not_loadable"


def test_exp1311_llama_import_failure_blocks_after_model_resolution(tmp_path: Path) -> None:
    """REQ-VERIFY-1311-7: llama.cpp import failure cannot produce headline data."""
    artifact = build_answer_stability_artifact(
        project_root=tmp_path,
        run_date="20260505",
        cached_pair_fn=_cached_pair,
        llama_importer=lambda: (False, None, "ImportError: no module named llama_cpp"),
        generation_fn=None,
    )

    assert artifact["status"] == "blocked"
    assert artifact["blocked_reason"] == "llama_cpp_import_failed"
    assert artifact["llama_cpp_import_error"] == "ImportError: no module named llama_cpp"
    assert artifact["headline_result_allowed"] is False
    assert artifact["models_used"] == [QWEN_SPEC["hf_id"], GEMMA_SPEC["hf_id"]]


def test_exp1311_all_generation_failures_block_terminally(tmp_path: Path) -> None:
    """REQ-VERIFY-1311-2: resolved SOTA specs still block if no model can run."""
    item = MicroItem("sat", "satquest", "x1", "SAT", 1, ((1,),))

    def generation_fn(*_args: Any, **_kwargs: Any) -> RawGeneration:
        raise RuntimeError("model load failed")

    artifact = build_answer_stability_artifact(
        project_root=tmp_path,
        run_date="20260505",
        cached_pair_fn=_cached_pair,
        generation_fn=generation_fn,
        generation_source="live_sota_llamacpp",
        items=[item],
    )

    assert artifact["status"] == "blocked"
    assert artifact["blocked_reason"] == "sota_generation_failed"
    assert artifact["honest_verdict"] == "blocked_sota_generation_failed"
    assert artifact["headline_result_allowed"] is False
    assert artifact["generation_errors"] == artifact["observed_response_count"]


def test_exp1311_run_experiment_writes_json(tmp_path: Path) -> None:
    """REQ-VERIFY-1311-1: run_experiment writes the completed artifact."""
    out_path = (
        tmp_path / "results" / "experiment_1311_sota_constraintbench_satquest_answer_stability.json"
    )
    item = MicroItem("sat", "satquest", "x1", "SAT", 1, ((1,),))

    artifact = run_experiment(
        project_root=tmp_path,
        run_date="20260505",
        output_path=out_path,
        cached_pair_fn=_cached_pair,
        generation_fn=lambda *_args, **_kwargs: RawGeneration(text="SAT", token_count=1),
        generation_source="live_sota_llamacpp",
        items=[item],
    )

    assert json.loads(out_path.read_text(encoding="utf-8")) == artifact
    assert artifact["headline_result_allowed"] is True


def test_exp1311_llama_collection_records_model_exception(tmp_path: Path) -> None:
    """REQ-VERIFY-1311-2: live model load failures become response errors."""

    class FailingLlama:
        def __init__(self, **_kwargs: Any) -> None:
            raise RuntimeError("load failed")

    item = MicroItem("sat", "satquest", "x1", "SAT", 1, ((1,),))

    artifact = build_answer_stability_artifact(
        project_root=tmp_path,
        run_date="20260505",
        cached_pair_fn=_cached_pair,
        llama_importer=lambda: (True, FailingLlama, None),
        items=[item],
    )

    assert artifact["status"] == "blocked"
    assert artifact["blocked_reason"] == "sota_generation_failed"
    assert all(row["error"] == "RuntimeError: load failed" for row in artifact["responses"])


def test_exp1311_fake_llama_collection_path_is_deterministic(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-1311: live collection path uses deterministic llama settings."""

    class FakeLlama:
        calls: list[dict[str, Any]] = []
        closed = 0

        def __init__(self, **kwargs: Any) -> None:
            self.calls.append(kwargs)

        def __call__(self, prompt: str, **kwargs: Any) -> dict[str, Any]:
            assert "SAT, UNSAT, or UNKNOWN" in prompt
            assert kwargs["max_tokens"] == 6
            assert kwargs["temperature"] == 0.0
            return {"choices": [{"text": " SAT"}], "usage": {"completion_tokens": 1}}

        def close(self) -> None:
            type(self).closed += 1

    item = MicroItem("sat", "satquest", "x1", "SAT", 1, ((1,),))
    FakeLlama.calls = []
    FakeLlama.closed = 0

    artifact = build_answer_stability_artifact(
        project_root=tmp_path,
        run_date="20260505",
        cached_pair_fn=_cached_pair,
        llama_importer=lambda: (True, FakeLlama, None),
        items=[item],
    )

    assert artifact["status"] == "complete"
    assert artifact["headline_result_allowed"] is True
    assert len(artifact["responses"]) == 4
    assert [call["main_gpu"] for call in FakeLlama.calls] == [0, 1]
    assert FakeLlama.closed == 2


def test_exp1311_empty_majority_label_abstains() -> None:
    """REQ-VERIFY-1311-6: absent model labels are counted as abstention."""
    assert mod._majority_label([]) == "ABSTAIN"
