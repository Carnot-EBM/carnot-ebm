"""Tests for Exp 5086 uPRM logprob cache retry.

Spec refs: REQ-VERIFY-5086, SCENARIO-VERIFY-5086.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_5086_uprm_logprob_cache_retry as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "verification" / "spec.md"


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _exp5085_gate(*, ready: bool = True) -> dict[str, Any]:
    return {
        "honest_verdict": "success_llamacpp_logprob_endpoint_ready" if ready else "blocked_endpoint",
        "logprob_endpoint_ready": ready,
        "endpoint_url": "http://127.0.0.1:46097",
        "sample_completion": {
            "model_hf_id": "unsloth/gemma-4-26B-A4B-it-GGUF",
            "model_path": "/models/gemma-4-26B-A4B-it-Q4_K_M.gguf",
            "route": "http://127.0.0.1:46097/completion",
        },
        "model_specs": {
            "resolved_models": {
                "flagship_moe": {
                    "hf_id": "unsloth/Qwen3.6-35B-A3B-GGUF",
                    "resolved_path": "/models/qwen.gguf",
                },
                "flagship_dense": {
                    "hf_id": "unsloth/gemma-4-31B-it-GGUF",
                    "resolved_path": "/models/gemma-31b.gguf",
                },
                "middle_moe": {
                    "hf_id": "unsloth/gemma-4-26B-A4B-it-GGUF",
                    "resolved_path": "/models/gemma-26b.gguf",
                },
            }
        },
    }


def _candidate_rows_5058() -> list[dict[str, Any]]:
    return [
        {
            "schema": "carnot.experiment_5058_sota_candidate_refresh_inwriting.row.v1",
            "row_id": "MuSR/murder_mysteries:0/sota5058-0000",
            "corpus": mod.MUSR_CORPUS,
            "question_id": "MuSR/murder_mysteries:0",
            "question_index": 0,
            "candidate_index": 0,
            "question": "Who did it?",
            "choices": ["Ada", "Bea"],
            "model_id": "unsloth/Qwen3.6-35B-A3B-GGUF",
            "model_path": "/models/qwen.gguf",
            "answer_text": "Ada checked the lock.\nTherefore Ada.",
            "parsed_answer": "Ada",
            "parse_status": "parsed",
            "prompt_hash": "a" * 64,
        },
        {
            "schema": "carnot.experiment_5058_sota_candidate_refresh_inwriting.row.v1",
            "row_id": "MuSR/murder_mysteries:0/sota5058-0001",
            "corpus": mod.MUSR_CORPUS,
            "question_id": "MuSR/murder_mysteries:0",
            "question_index": 0,
            "candidate_index": 1,
            "question": "Who did it?",
            "choices": ["Ada", "Bea"],
            "model_id": "unsloth/Qwen3.6-35B-A3B-GGUF",
            "model_path": "/models/qwen.gguf",
            "answer_text": "Bea had the key.",
            "parsed_answer": "Bea",
            "parse_status": "parsed",
            "prompt_hash": "b" * 64,
        },
        {
            "schema": "carnot.experiment_5058_sota_candidate_refresh_inwriting.row.v1",
            "row_id": "MuSR/murder_mysteries:1/sota5058-0000",
            "corpus": mod.MUSR_CORPUS,
            "question_id": "MuSR/murder_mysteries:1",
            "question_index": 1,
            "candidate_index": 0,
            "question": "Who had motive?",
            "choices": ["Cal", "Dia"],
            "model_id": "unsloth/Qwen3.6-35B-A3B-GGUF",
            "model_path": "/models/qwen.gguf",
            "answer_text": "Cal argued first.",
            "parsed_answer": "Cal",
            "parse_status": "parsed",
            "prompt_hash": "c" * 64,
        },
    ]


def _fallback_rows_5029() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in _candidate_rows_5058():
        rows.append(
            {
                "schema": "carnot.shared_logprob_candidate_cache_v2.candidate_row.v1",
                "corpus": mod.MUSR_CORPUS,
                "question_id": row["question_id"],
                "question_index": row["question_index"],
                "candidate_id": f"{row['question_id']}/cached-{row['candidate_index']}",
                "candidate_index": row["candidate_index"],
                "question": row["question"],
                "context": f"context for {row['question_id']}",
                "choices": row["choices"],
                "gold": row["choices"][0],
                "answer": row["answer_text"],
            }
        )
    return rows


def _telemetry(text: str = "Ada checked the lock.\nTherefore Ada.") -> dict[str, Any]:
    tokens = ["Ada", " checked", " the", " lock", ".\n", "Therefore", " Ada", "."]
    return {
        "completion_text": text,
        "tokens": tokens,
        "token_logprobs": [-0.1, -0.2, -0.3, -0.4, -0.5, -0.6, -0.7, -0.8],
        "top_logprobs": [
            {"Ada": -0.1, "Bea": -2.0},
            {" checked": -0.2, " had": -1.5},
            {" the": -0.3},
            {" lock": -0.4},
            {".\n": -0.5},
            {"Therefore": -0.6},
            {" Ada": -0.7, " Bea": -1.7},
            {".": -0.8},
        ],
    }


def test_req_verify_5086_spec_declares_retry_cache_contract() -> None:
    """REQ-VERIFY-5086: OpenSpec anchors the retry cache and artifact fields."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    for marker in (
        "REQ-VERIFY-5086",
        "SCENARIO-VERIFY-5086",
        "experiment_5086_uprm_logprob_cache_retry.py",
        "results/experiment_5086_uprm_logprob_cache_retry_v467.json",
        "results/experiment_5086_uprm_logprob_cache_retry_v467.jsonl",
        "blocked_uprm_logprob_cache_retry_endpoint_failed",
        "success_uprm_logprob_cache_retry_ready_n",
        "unsloth/Qwen3.6-35B-A3B-GGUF",
        "unsloth/gemma-4-31B-it-GGUF",
        "unsloth/gemma-4-26B-A4B-it-GGUF",
    ):
        assert marker in spec
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert field in spec
        assert principle["principle"] in spec


def test_req_verify_5086_candidate_loader_prefers_5058_and_enriches_from_5029(
    tmp_path: Path,
) -> None:
    """REQ-VERIFY-5086: existing candidate text is reused, not regenerated."""

    _write_jsonl(tmp_path / mod.EXP5058_CACHE_RELATIVE_PATH, _candidate_rows_5058())
    _write_jsonl(tmp_path / mod.EXP5029_CACHE_RELATIVE_PATH, _fallback_rows_5029())

    rows, summary = mod.load_candidate_inputs(root=tmp_path, min_questions=2)

    assert summary["cache_input_path"].endswith("experiment_5058_sota_candidate_refresh_inwriting.jsonl")
    assert summary["candidate_source"] == "exp5058_enriched_by_exp5029"
    assert summary["n_questions"] == 2
    assert len(rows) == 3
    assert rows[0]["candidate_text"].startswith("Ada checked")
    assert rows[0]["context"] == "context for MuSR/murder_mysteries:0"
    assert rows[0]["source_candidate_id"] == "MuSR/murder_mysteries:0/sota5058-0000"


def test_scenario_verify_5086_row_provenance_and_step_boundaries() -> None:
    """SCENARIO-VERIFY-5086: row telemetry preserves token and step provenance."""

    candidate = mod.load_candidate_inputs_from_rows(
        primary_rows=[_candidate_rows_5058()[0]],
        fallback_rows=_fallback_rows_5029(),
        min_questions=1,
    )[0][0]
    prompt = mod.build_scoring_prompt(candidate)
    row = mod.build_cache_row(
        candidate=candidate,
        telemetry=_telemetry(),
        prompt=prompt,
        endpoint_used="http://127.0.0.1:46097/completion",
        model_hf_id="unsloth/gemma-4-26B-A4B-it-GGUF",
        gguf_path="/models/gemma-4-26B-A4B-it-Q4_K_M.gguf",
        random_seed=mod.RANDOM_SEED,
    )

    assert mod.validate_cache_row(row) == []
    assert row["question_id"] == "MuSR/murder_mysteries:0"
    assert row["candidate_id"] == "MuSR/murder_mysteries:0/sota5058-0000"
    assert row["model_hf_id"] == "unsloth/gemma-4-26B-A4B-it-GGUF"
    assert len(row["prompt_hash"]) == 64
    assert len(row["response_hash"]) == 64
    assert row["token_count"] == 8
    assert row["top_logprobs_available"] is True
    assert [step["step_index"] for step in row["step_boundaries"]] == [0, 1]
    assert row["step_boundaries"][0]["token_start"] == 0
    assert row["step_boundaries"][1]["token_end"] == 8


def test_scenario_verify_5086_run_success_resumes_and_writes_artifact(
    tmp_path: Path,
) -> None:
    """SCENARIO-VERIFY-5086: endpoint-backed scoring appends only missing rows."""

    _write_json(tmp_path / mod.EXP5085_RELATIVE_PATH, _exp5085_gate())
    _write_jsonl(tmp_path / mod.EXP5058_CACHE_RELATIVE_PATH, _candidate_rows_5058())
    _write_jsonl(tmp_path / mod.EXP5029_CACHE_RELATIVE_PATH, _fallback_rows_5029())
    cache_path = tmp_path / mod.CACHE_RELATIVE_PATH
    artifact_path = tmp_path / mod.RESULT_RELATIVE_PATH
    first_candidate = mod.load_candidate_inputs(root=tmp_path, min_questions=2)[0][0]
    preexisting = mod.build_cache_row(
        candidate=first_candidate,
        telemetry=_telemetry(),
        prompt=mod.build_scoring_prompt(first_candidate),
        endpoint_used="http://127.0.0.1:46097/completion",
        model_hf_id="unsloth/gemma-4-26B-A4B-it-GGUF",
        gguf_path="/models/gemma-4-26B-A4B-it-Q4_K_M.gguf",
        random_seed=mod.RANDOM_SEED,
    )
    mod.append_jsonl_row(cache_path, preexisting)

    scorer_calls: list[str] = []

    def endpoint_probe(endpoint: str, timeout_s: float) -> dict[str, Any]:
        return {
            "available": True,
            "endpoint_used": endpoint.rstrip("/") + "/completion",
            "detail": f"ready within {timeout_s}",
            "token_logprob_count": 2,
            "top_logprob_row_count": 2,
        }

    def scorer(*, candidate: dict[str, Any], prompt: str, endpoint: str, seed: int) -> dict[str, Any]:
        scorer_calls.append(candidate["candidate_id"])
        assert "Candidate trajectory:" in prompt
        assert endpoint == "http://127.0.0.1:46097/completion"
        assert seed >= mod.RANDOM_SEED
        return _telemetry(str(candidate["candidate_text"]))

    artifact = mod.run(
        root=tmp_path,
        artifact_path=artifact_path,
        cache_path=cache_path,
        min_questions=2,
        endpoint_probe=endpoint_probe,
        candidate_scorer=scorer,
        now=iter([10.0, 70.0]).__next__,
        write=True,
    )
    rows = mod.read_complete_rows(cache_path)

    assert artifact["honest_verdict"] == "success_uprm_logprob_cache_retry_ready_n2"
    assert artifact["logprob_cache_ready"] is True
    assert artifact["step_cache_ready"] is True
    assert artifact["n_questions"] == 2
    assert artifact["n_candidates"] == 3
    assert artifact["n_rows_complete"] == 3
    assert artifact["parse_rate"] == pytest.approx(1.0)
    assert artifact["top_logprob_coverage"] == pytest.approx(1.0)
    assert artifact["resume_summary"]["existing_complete_rows"] == 1
    assert artifact["resume_summary"]["appended_rows"] == 2
    assert len(scorer_calls) == 2
    assert len(rows) == 3
    assert json.loads(artifact_path.read_text(encoding="utf-8")) == artifact
    assert mod.artifact_schema_errors(artifact) == []


def test_scenario_verify_5086_endpoint_failure_blocks_honestly(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-5086: unavailable Exp5085 endpoint emits blocked artifact."""

    _write_json(tmp_path / mod.EXP5085_RELATIVE_PATH, _exp5085_gate())
    _write_jsonl(tmp_path / mod.EXP5058_CACHE_RELATIVE_PATH, _candidate_rows_5058())
    _write_jsonl(tmp_path / mod.EXP5029_CACHE_RELATIVE_PATH, _fallback_rows_5029())

    artifact = mod.run(
        root=tmp_path,
        artifact_path=tmp_path / mod.RESULT_RELATIVE_PATH,
        cache_path=tmp_path / mod.CACHE_RELATIVE_PATH,
        min_questions=2,
        endpoint_probe=lambda endpoint, timeout_s: {
            "available": False,
            "endpoint_used": endpoint.rstrip("/") + "/completion",
            "detail": "connection refused",
            "token_logprob_count": 0,
            "top_logprob_row_count": 0,
        },
        candidate_scorer=lambda **_: pytest.fail("scorer must not run when endpoint blocks"),
        now=iter([1.0, 2.0]).__next__,
        write=True,
    )

    assert artifact["honest_verdict"] == "blocked_uprm_logprob_cache_retry_endpoint_failed"
    assert artifact["logprob_cache_ready"] is False
    assert artifact["step_cache_ready"] is False
    assert artifact["n_candidates"] == 3
    assert artifact["n_rows_complete"] == 0
    assert artifact["endpoint_used"] == "http://127.0.0.1:46097/completion"
    assert artifact["preconditions_checked"]["exp5085_artifact"]["available"] is True
    assert artifact["preconditions_checked"]["endpoint_live_probe"]["available"] is False
    assert artifact["preconditions_checked"]["disk_space"]["available"] is True
    assert not (tmp_path / mod.CACHE_RELATIVE_PATH).exists()
    assert mod.artifact_schema_errors(artifact) == []
