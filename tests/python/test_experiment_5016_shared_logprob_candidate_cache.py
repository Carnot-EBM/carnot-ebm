"""Tests for Exp 5016 shared logprob candidate cache.

Spec refs: REQ-VERIFY-5016, SCENARIO-VERIFY-5016.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from carnot import experiment_5016_shared_logprob_candidate_cache as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "verification" / "spec.md"


def _corpus_rows() -> list[dict[str, Any]]:
    return [
        {
            "row_id": "musr:0",
            "corpus": "MuSR/murder_mysteries",
            "question": "Who had the stronger motive?",
            "context": "A short mystery fixture.",
            "choices": ["Ada", "Bea"],
            "gold": "Ada",
        },
        {
            "row_id": "musr:1",
            "corpus": "MuSR/murder_mysteries",
            "question": "Who had the alibi?",
            "context": "A second mystery fixture.",
            "choices": ["Cal", "Dia"],
            "gold": "Dia",
        },
    ]


def _candidate(index: int, answer: str = "Ada") -> dict[str, Any]:
    return {
        "candidate_id": f"candidate-{index}",
        "answer": answer,
        "reasoning": f"Step 1: reason {index}\nANSWER: {answer}",
        "steps": [f"Step 1: reason {index}", f"ANSWER: {answer}"],
        "cache_index": index,
        "temperature": 0.7,
        "generation_model": mod.MODEL_NAME,
        "gpu": 0,
        "source": "shared_logprob_candidate_cache",
        "token_logprobs": [-0.1 - index, -0.2 - index],
        "top_logprobs": [{" Ada": -0.1, " Bea": -2.0}],
        "mean_logprob": -0.15 - index,
        "uprm_marker_logprobs": [{"+": -0.2, "-": -1.7}, {" +": -0.3, " -": -1.5}],
    }


def _cache_row(
    row: dict[str, Any],
    *,
    row_index: int,
    k_candidates: int,
    random_seed: int,
    server_port: int,
) -> dict[str, Any]:
    del server_port
    return mod.build_cache_row(
        row=row,
        row_index=row_index,
        candidates=[_candidate(index, row["choices"][index % 2]) for index in range(k_candidates)],
        k_candidates=k_candidates,
        random_seed=random_seed,
    )


def test_req_verify_5016_spec_declares_shared_cache_contract() -> None:
    """REQ-VERIFY-5016: OpenSpec anchors the shared logprob cache fields."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    for marker in (
        "REQ-VERIFY-5016",
        "SCENARIO-VERIFY-5016",
        "experiment_5016_shared_logprob_candidate_cache.py",
        "results/experiment_5016_shared_logprob_candidate_cache.json",
        "blocked_<resource>",
        "success_logprob_candidate_cache_built_musr_n<N>_k<K>",
        "uprm_marker_logprobs",
        "completion_probabilities",
    ):
        assert marker in spec
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert field in spec
        assert principle["principle"] in spec


def test_scenario_verify_5016_jsonl_schema_round_trips(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-5016: a cache row preserves logprob telemetry through JSONL."""

    path = tmp_path / "cache.jsonl"
    row = _cache_row(
        _corpus_rows()[0],
        row_index=0,
        k_candidates=5,
        random_seed=mod.RANDOM_SEED,
        server_port=mod.DEFAULT_SERVER_PORT,
    )

    mod.append_jsonl_atomic(path, row)
    loaded = mod.read_complete_cache_rows(path, k_candidates=5)
    summary = mod.cache_summary(path, k_candidates=5)

    assert len(loaded) == 1
    assert loaded[0] == row
    assert mod.validate_cache_row(loaded[0], k_candidates=5) == []
    assert summary == {
        "n_cached_rows": 1,
        "n_questions": 1,
        "min_candidates_per_question": 5,
        "has_per_token_logprobs": True,
        "corpora_cached": ["MuSR/murder_mysteries"],
    }
    assert json.loads(json.dumps(row)) == row


def test_scenario_verify_5016_resume_skips_complete_questions(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-5016: capped runs resume by skipping complete question rows."""

    gguf = tmp_path / "gemma.gguf"
    gguf.write_text("fixture", encoding="utf-8")
    cache_path = tmp_path / mod.CACHE_RELATIVE_PATH
    artifact_path = tmp_path / mod.RESULT_RELATIVE_PATH
    mod.append_jsonl_atomic(
        cache_path,
        _cache_row(
            _corpus_rows()[0],
            row_index=0,
            k_candidates=5,
            random_seed=mod.RANDOM_SEED,
            server_port=mod.DEFAULT_SERVER_PORT,
        ),
    )
    calls: list[int] = []

    def builder(**kwargs: Any) -> dict[str, Any]:
        calls.append(int(kwargs["row_index"]))
        return _cache_row(**kwargs)

    artifact = mod.run(
        root=tmp_path,
        artifact_path=artifact_path,
        gguf_resolver=lambda: str(gguf),
        server_probe=lambda port: mod.PreconditionCheck(
            "llama_server_logprobs",
            True,
            "server returns completion_probabilities with top_logprobs",
            f"http://127.0.0.1:{port}/completion",
        ),
        corpus_loader=lambda limit: _corpus_rows()[:limit],
        candidate_row_builder=builder,
        min_questions=2,
        k_candidates=5,
        now=lambda: 10.0,
    )

    assert calls == [1]
    assert artifact["honest_verdict"] == "success_logprob_candidate_cache_built_musr_n2_k5"
    assert artifact["candidate_cache_built"] is True
    assert artifact["n_questions"] == 2
    assert artifact["n_cached_rows"] == 2
    assert artifact["candidates_per_question"] == 5
    assert artifact["has_per_token_logprobs"] is True
    assert artifact["corpora_cached"] == ["MuSR/murder_mysteries"]
    assert artifact["inference_substrate"] == "live_llm_inference"
    assert artifact["model_specs"]["gguf_path"] == gguf.as_posix()
    assert artifact["cache_jsonl_path"] == cache_path.as_posix()
    assert artifact["reproducibility_checksum"].startswith("sha256:")
    assert mod.artifact_schema_errors(artifact) == []
    assert len(cache_path.read_text(encoding="utf-8").splitlines()) == 2
    assert json.loads(artifact_path.read_text(encoding="utf-8")) == artifact


def test_scenario_verify_5016_blocked_artifact_names_missing_resource(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-5016: missing preconditions block without fake cache claims."""

    artifact = mod.run(
        root=tmp_path,
        gguf_resolver=lambda: None,
        server_probe=lambda port: mod.PreconditionCheck(
            "llama_server_logprobs", False, "server unavailable", str(port)
        ),
        corpus_loader=lambda _limit: [],
        candidate_row_builder=_cache_row,
        min_questions=2,
        now=lambda: 3.0,
    )

    assert artifact["honest_verdict"] == "blocked_gemma_gguf_cache"
    assert artifact["candidate_cache_built"] is False
    assert artifact["n_questions"] == 0
    assert artifact["n_cached_rows"] == 0
    assert artifact["has_per_token_logprobs"] is False
    assert artifact["inference_substrate"] == "precondition_check_only"
    assert artifact["preconditions_checked"][0]["available"] is False
    assert mod.artifact_schema_errors(artifact) == []


def test_req_verify_5016_validation_reports_malformed_rows(tmp_path: Path) -> None:
    """REQ-VERIFY-5016: malformed cache rows are not counted as complete."""

    path = tmp_path / "cache.jsonl"
    bad = {
        "schema": mod.CACHE_ROW_SCHEMA,
        "row_id": "bad",
        "corpus": "MuSR/murder_mysteries",
        "candidates": [{"token_logprobs": [], "uprm_marker_logprobs": [{"+": -1.0}]}],
    }
    mod.append_jsonl_atomic(path, bad)
    mod.append_jsonl_atomic(path, {"not": "complete"})
    path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    mixed_bad = {
        "schema": mod.CACHE_ROW_SCHEMA,
        "row_id": "",
        "corpus": "wrong",
        "candidates": [
            "not-a-candidate",
            {
                "token_logprobs": [-0.1],
                "uprm_marker_logprobs": ["not-a-marker-row"],
            },
            {
                "token_logprobs": [-0.1],
                "uprm_marker_logprobs": [{"+": "not-a-number", "-": -1.0}],
            },
        ],
    }
    errors = mod.validate_cache_row(bad, k_candidates=5)
    mixed_errors = mod.validate_cache_row(mixed_bad, k_candidates=3)
    artifact = mod.build_artifact(
        honest_verdict="blocked_generation_or_cache_error",
        root=tmp_path,
        cache_path=path,
        preconditions_checked=[],
        gguf_path=None,
        min_questions=2,
        k_candidates=5,
        started_at=0.0,
        finished_at=1.0,
    )

    assert "candidates" in errors
    assert "candidate_0_token_logprobs" in errors
    assert "candidate_0_uprm_marker_logprobs" in errors
    assert "candidate_0" in mixed_errors
    assert "candidate_1_uprm_marker_logprobs" in mixed_errors
    assert "candidate_2_uprm_marker_logprobs" in mixed_errors
    assert mod._finite_number(True) is False
    assert "corpus" in mixed_errors
    assert "row_id" in mixed_errors
    assert mod.read_complete_cache_rows(path, k_candidates=5) == []
    assert artifact["candidate_cache_built"] is False
    assert artifact["honest_verdict"] == "blocked_generation_or_cache_error"


def test_req_verify_5016_error_paths_remain_schema_valid(tmp_path: Path) -> None:
    """REQ-VERIFY-5016: corpus and generation errors produce blocked artifacts."""

    gguf = tmp_path / "gemma.gguf"
    gguf.write_text("fixture", encoding="utf-8")

    corpus_blocked = mod.run(
        root=tmp_path / "corpus",
        gguf_resolver=lambda: str(gguf),
        server_probe=lambda _port: mod.PreconditionCheck("llama_server_logprobs", True, "ok"),
        corpus_loader=lambda _limit: (_ for _ in ()).throw(RuntimeError("no corpus")),
        candidate_row_builder=_cache_row,
        min_questions=2,
        now=lambda: 4.0,
    )
    generated_bad = mod.run(
        root=tmp_path / "generated",
        gguf_resolver=lambda: str(gguf),
        server_probe=lambda _port: mod.PreconditionCheck("llama_server_logprobs", True, "ok"),
        corpus_loader=lambda limit: _corpus_rows()[:limit],
        candidate_row_builder=lambda **_kwargs: {"schema": "bad"},
        min_questions=2,
        now=lambda: 5.0,
    )
    bad_schema = dict(generated_bad)
    bad_schema.update(
        {
            "spec_refs": [],
            "field_principles": {},
            "candidate_cache_built": "no",
            "has_per_token_logprobs": "no",
            "n_questions": "0",
            "candidates_per_question": "5",
            "n_cached_rows": "0",
            "preconditions_checked": {},
            "honest_verdict": "maybe",
        }
    )

    assert corpus_blocked["honest_verdict"] == "blocked_musr_corpus"
    assert corpus_blocked["preconditions_checked"][-1]["detail"] == "RuntimeError: no corpus"
    assert mod.artifact_schema_errors(corpus_blocked) == []
    assert generated_bad["honest_verdict"] == "blocked_generation_or_cache_error"
    assert generated_bad["preconditions_checked"][-1]["resource"] == "generation_or_cache_error"
    assert "generated cache row is malformed" in generated_bad["preconditions_checked"][-1]["detail"]
    for field in (
        "candidate_cache_built",
        "candidates_per_question",
        "field_principles",
        "has_per_token_logprobs",
        "honest_verdict",
        "n_cached_rows",
        "n_questions",
        "preconditions_checked",
        "spec_refs",
    ):
        assert field in mod.artifact_schema_errors(bad_schema)
