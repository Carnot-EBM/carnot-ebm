"""Tests for Exp 5029 rescored MuSR logprob candidate cache v2.

Spec refs: REQ-VERIFY-5029, SCENARIO-VERIFY-5029.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_5029_shared_logprob_candidate_cache_v2 as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "verification" / "spec.md"


def _questions() -> list[dict[str, Any]]:
    return [
        {
            "question_id": "q0000",
            "question_index": 0,
            "corpus": mod.CORPUS,
            "question": "Who had motive?",
            "context": "Ada and Bea argued in the library.",
            "choices": ["Ada", "Bea"],
            "gold": "Ada",
            "checkpoint_path": "/tmp/q0000.json",
            "candidates": [
                {"candidate_index": 0, "answer": "Ada"},
                {"candidate_index": 1, "answer": "Bea"},
            ],
        },
        {
            "question_id": "q0001",
            "question_index": 1,
            "corpus": mod.CORPUS,
            "question": "Who had the alibi?",
            "context": "Cal signed the register before Dia arrived.",
            "choices": ["Cal", "Dia"],
            "gold": "Cal",
            "checkpoint_path": "/tmp/q0001.json",
            "candidates": [
                {"candidate_index": 0, "answer": "Cal"},
                {"candidate_index": 1, "answer": "Dia"},
            ],
        },
    ]


def _telemetry(answer: str = "Ada") -> dict[str, Any]:
    return {
        "completion_text": "+",
        "tokens": [" Candidate", f" {answer}", " +"],
        "token_logprobs": [-0.1, -0.2, -0.3],
        "top_logprobs": [{"Ada": -0.2}, {"+": -0.05, "-": -3.2}],
        "marker_top_logprobs": {"+": -0.05, "-": -3.2},
    }


def _scored_row(question: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    return mod.build_scored_candidate_row(
        question=question,
        candidate=candidate,
        telemetry=_telemetry(str(candidate["answer"])),
        random_seed=mod.RANDOM_SEED,
        server_port=mod.DEFAULT_SERVER_PORT,
    )


def test_req_verify_5029_spec_declares_rescore_contract() -> None:
    """REQ-VERIFY-5029: OpenSpec names the v2 rescore cache and zero-row gate."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    for marker in (
        "REQ-VERIFY-5029",
        "SCENARIO-VERIFY-5029",
        "experiment_5029_shared_logprob_candidate_cache_v2.py",
        "results/experiment_5029_shared_logprob_candidate_cache_v2.json",
        "blocked_cache_zero_rows",
        "rescored_not_regenerated",
        "uprm_marker_logprobs",
    ):
        assert marker in spec
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert field in spec
        assert principle["principle"] in spec


def test_scenario_verify_5029_candidate_jsonl_round_trips(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-5029: candidate rows preserve token and marker logprobs."""

    path = tmp_path / "cache.jsonl"
    row = _scored_row(_questions()[0], _questions()[0]["candidates"][0])

    mod.append_jsonl_row(path, row)
    loaded = mod.read_complete_candidate_rows(path)
    summary = mod.cache_summary(path, min_questions=1)

    assert loaded == [row]
    assert mod.validate_candidate_row(row) == []
    assert summary["n_cached_rows"] == 1
    assert summary["n_questions"] == 1
    assert summary["has_per_token_logprobs"] is True
    assert summary["corpora_cached"] == [mod.CORPUS]
    assert json.loads(json.dumps(row)) == row


def test_req_verify_5029_parser_and_checkpoint_edges(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-VERIFY-5029: parser and checkpoint loaders fail closed locally."""

    completion = mod.parse_logprob_payload(
        {
            "content": "+",
            "completion_probabilities": [
                "skip",
                {
                    "token": " +",
                    "logprob": "-0.1",
                    "top_logprobs": [
                        {"token": "+", "logprob": -0.1},
                        {"token": "-", "logprob": -2.0},
                        {"token": "bad", "logprob": True},
                    ],
                },
            ],
        }
    )
    choices = mod.parse_logprob_payload(
        {
            "choices": [
                {
                    "text": "+",
                    "logprobs": {
                        "tokens": [" +"],
                        "token_logprobs": [True, "bad", "-0.4"],
                        "top_logprobs": [{"+": "-0.3", "-": -1.3, "bad": False}],
                    },
                }
            ]
        }
    )
    empty = mod.parse_logprob_payload({"content": "x"})

    assert completion["tokens"] == [" +"]
    assert completion["token_logprobs"] == [-0.1]
    assert completion["top_logprobs"] == [{"+": -0.1, "-": -2.0}]
    assert mod._first_marker_top_logprobs(completion["top_logprobs"]) == {
        "+": -0.1,
        "-": -2.0,
    }
    assert choices["token_logprobs"] == [-0.4]
    assert choices["top_logprobs"] == [{"+": -0.3, "-": -1.3}]
    assert empty["token_logprobs"] == []
    assert mod._first_marker_top_logprobs([{"x": -1.0}]) is None
    assert "MARKER:" in mod.build_scoring_prompt(_questions()[0], {"answer": "Ada"})

    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    (checkpoint_dir / "q0000.json").write_text(
        json.dumps({"gold": "Ada", "answers": ["Ada", None, "Bea"]}),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        mod.harness,
        "load_musr_murder_mysteries",
        lambda limit: [
            {
                "row_id": "musr:0",
                "question": "Q?",
                "context": "C",
                "choices": ["Ada", "Bea"],
                "gold": "Ada",
            }
        ],
    )
    loaded = mod.load_cached_musr_candidate_questions(limit=1, checkpoint_dir=checkpoint_dir)
    assert loaded[0]["question_id"] == "musr:0"
    assert len(loaded[0]["candidates"]) == 2

    monkeypatch.setattr(
        mod.harness,
        "load_musr_murder_mysteries",
        lambda limit: (_ for _ in ()).throw(RuntimeError("no dataset")),
    )
    fallback = mod.load_cached_musr_candidate_questions(limit=1, checkpoint_dir=checkpoint_dir)
    assert fallback[0]["question_id"] == "q0000"

    bad_object = tmp_path / "bad_object.json"
    bad_object.write_text("[]", encoding="utf-8")
    bad_answers = tmp_path / "bad_answers.json"
    bad_answers.write_text(json.dumps({"answers": "Ada"}), encoding="utf-8")
    empty_answers = tmp_path / "empty_answers.json"
    empty_answers.write_text(json.dumps({"answers": [None, ""]}), encoding="utf-8")

    with pytest.raises(ValueError, match="not an object"):
        mod._question_from_checkpoint(checkpoint_path=bad_object, question_index=0)
    with pytest.raises(ValueError, match="lacks answers"):
        mod._question_from_checkpoint(checkpoint_path=bad_answers, question_index=0)
    with pytest.raises(ValueError, match="no non-empty answers"):
        mod._question_from_checkpoint(checkpoint_path=empty_answers, question_index=0)


def test_scenario_verify_5029_resume_skips_done_candidate_rows(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-5029: resume skips complete candidates and flushes new rows."""

    gguf = tmp_path / "gemma.gguf"
    gguf.write_text("fixture", encoding="utf-8")
    cache_path = tmp_path / mod.CACHE_RELATIVE_PATH
    artifact_path = tmp_path / mod.RESULT_RELATIVE_PATH
    first_question = _questions()[0]
    mod.append_jsonl_row(cache_path, _scored_row(first_question, first_question["candidates"][0]))
    calls: list[tuple[str, int]] = []

    def scorer(**kwargs: Any) -> dict[str, Any]:
        question = kwargs["question"]
        candidate = kwargs["candidate"]
        calls.append((question["question_id"], int(candidate["candidate_index"])))
        return _telemetry(str(candidate["answer"]))

    artifact = mod.run(
        root=tmp_path,
        artifact_path=artifact_path,
        cache_path=cache_path,
        gguf_resolver=lambda: gguf.as_posix(),
        server_probe=lambda port: mod.PreconditionCheck(
            "llama_server_logprobs",
            True,
            "server returns completion_probabilities with marker top_logprobs",
            f"http://127.0.0.1:{port}/completion",
        ),
        candidate_loader=lambda limit: _questions()[:limit],
        candidate_scorer=scorer,
        min_questions=2,
        now=lambda: 100.0,
    )

    assert calls == [("q0000", 1), ("q0001", 0), ("q0001", 1)]
    assert artifact["honest_verdict"] == "success_logprob_cache_rescored_musr_n2"
    assert artifact["candidate_cache_built"] is True
    assert artifact["rescored_not_regenerated"] is True
    assert artifact["n_questions"] == 2
    assert artifact["n_cached_rows"] == 4
    assert artifact["candidates_per_question"] == 2
    assert artifact["has_per_token_logprobs"] is True
    assert artifact["corpora_cached"] == [mod.CORPUS]
    assert artifact["inference_substrate"] == "live_llm_inference"
    assert artifact["model_specs"]["gguf_path"] == gguf.as_posix()
    assert artifact["cache_jsonl_path"] == cache_path.as_posix()
    assert artifact["reproducibility_checksum"].startswith("sha256:")
    assert mod.artifact_schema_errors(artifact) == []
    assert len(cache_path.read_text(encoding="utf-8").splitlines()) == 4
    assert json.loads(artifact_path.read_text(encoding="utf-8")) == artifact


def test_scenario_verify_5029_partial_question_errors_preserve_rows(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-5029: malformed later candidates do not erase flushed rows."""

    gguf = tmp_path / "gemma.gguf"
    gguf.write_text("fixture", encoding="utf-8")
    calls: list[int] = []

    def scorer(**kwargs: Any) -> dict[str, Any]:
        candidate = kwargs["candidate"]
        calls.append(int(candidate["candidate_index"]))
        if int(candidate["candidate_index"]) == 1:
            return {"token_logprobs": [-0.2], "top_logprobs": [{"x": -1.0}]}
        return _telemetry(str(candidate["answer"]))

    weird_question = dict(_questions()[0])
    weird_question["candidates"] = ["not-a-candidate", *_questions()[0]["candidates"]]
    artifact = mod.run(
        root=tmp_path,
        gguf_resolver=lambda: gguf.as_posix(),
        server_probe=lambda _port: mod.PreconditionCheck("llama_server_logprobs", True, "ok"),
        candidate_loader=lambda _limit: [weird_question],
        candidate_scorer=scorer,
        min_questions=1,
        now=lambda: 11.0,
    )

    assert calls == [0, 1]
    assert artifact["honest_verdict"] == "success_logprob_cache_rescored_musr_n1"
    assert artifact["candidate_cache_built"] is True
    assert artifact["n_cached_rows"] == 1
    assert artifact["question_errors"]
    assert artifact["preconditions_checked"][-1]["resource"] == "question_scoring_errors"


def test_scenario_verify_5029_incomplete_nonzero_cache_blocks(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-5029: nonzero rows below the question floor are incomplete."""

    gguf = tmp_path / "gemma.gguf"
    gguf.write_text("fixture", encoding="utf-8")
    artifact = mod.run(
        root=tmp_path,
        gguf_resolver=lambda: gguf.as_posix(),
        server_probe=lambda _port: mod.PreconditionCheck("llama_server_logprobs", True, "ok"),
        candidate_loader=lambda _limit: _questions(),
        candidate_scorer=lambda **kwargs: _telemetry(str(kwargs["candidate"]["answer"]))
        if kwargs["question"]["question_id"] == "q0000"
        else (_ for _ in ()).throw(RuntimeError("second question failed")),
        min_questions=2,
        now=lambda: 13.0,
    )

    assert artifact["honest_verdict"] == "blocked_incomplete_musr_n1"
    assert artifact["candidate_cache_built"] is False
    assert artifact["n_cached_rows"] == 2
    assert artifact["question_errors"]


def test_scenario_verify_5029_zero_row_run_is_blocked(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-5029: zero scored rows is blocked_cache_zero_rows."""

    gguf = tmp_path / "gemma.gguf"
    gguf.write_text("fixture", encoding="utf-8")

    def scorer(**_kwargs: Any) -> dict[str, Any]:
        raise RuntimeError("synthetic scorer failure")

    artifact = mod.run(
        root=tmp_path,
        gguf_resolver=lambda: gguf.as_posix(),
        server_probe=lambda _port: mod.PreconditionCheck("llama_server_logprobs", True, "ok"),
        candidate_loader=lambda limit: _questions()[:limit],
        candidate_scorer=scorer,
        min_questions=2,
        now=lambda: 7.0,
    )

    assert artifact["honest_verdict"] == "blocked_cache_zero_rows"
    assert artifact["candidate_cache_built"] is False
    assert artifact["n_cached_rows"] == 0
    assert artifact["question_errors"]
    assert artifact["preconditions_checked"][-1]["resource"] == "question_scoring_errors"
    assert mod.artifact_schema_errors(artifact) == []


def test_req_verify_5029_preconditions_and_validation_fail_closed(tmp_path: Path) -> None:
    """REQ-VERIFY-5029: missing resources and malformed rows do not count."""

    path = tmp_path / "cache.jsonl"
    malformed = {
        "schema": "bad",
        "corpus": "bad",
        "question_id": "",
        "candidate_index": 0,
        "answer": "",
        "candidate_id": "",
        "rescored_not_regenerated": False,
        "token_logprobs": [True, "bad"],
        "uprm_marker_logprobs": ["bad", {"+": -0.1}],
    }
    mod.append_jsonl_row(path, malformed)
    path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    blocked = mod.run(
        root=tmp_path,
        gguf_resolver=lambda: None,
        server_probe=lambda _port: mod.PreconditionCheck(
            "llama_server_logprobs", False, "server unavailable"
        ),
        candidate_loader=lambda _limit: [],
        min_questions=2,
        now=lambda: 9.0,
    )
    bad_artifact = dict(blocked)
    bad_artifact.update(
        {
            "spec_refs": [],
            "field_principles": {},
            "candidate_cache_built": "no",
            "rescored_not_regenerated": "yes",
            "has_per_token_logprobs": "no",
            "n_questions": "0",
            "candidates_per_question": "0",
            "n_cached_rows": "0",
            "preconditions_checked": {},
            "honest_verdict": "maybe",
        }
    )
    impossible_built = dict(blocked)
    impossible_built.update(
        {
            "candidate_cache_built": True,
            "n_cached_rows": 0,
            "rescored_not_regenerated": False,
        }
    )
    gguf = tmp_path / "gemma.gguf"
    gguf.write_text("fixture", encoding="utf-8")
    checks, _gguf_path, questions = mod.check_preconditions(
        root=tmp_path,
        gguf_resolver=lambda: gguf.as_posix(),
        server_probe=lambda _port: mod.PreconditionCheck("llama_server_logprobs", True, "ok"),
        candidate_loader=lambda _limit: (_ for _ in ()).throw(RuntimeError("no cache")),
        min_questions=1,
        server_port=mod.DEFAULT_SERVER_PORT,
    )
    second_dir = tmp_path / "results" / "gpqa_candidate_checkpoints"
    second_dir.mkdir(parents=True)
    (second_dir / "q0000.json").write_text("{}", encoding="utf-8")
    mod.append_jsonl_row(path, _scored_row(_questions()[0], _questions()[0]["candidates"][0]))
    second_corpus_artifact = mod.build_artifact(
        honest_verdict="success_pending_cache_summary",
        root=tmp_path,
        artifact_path=tmp_path / "artifact.json",
        cache_path=path,
        preconditions_checked=[],
        gguf_path=gguf,
        candidate_set_sha256="sha256:test",
        min_questions=1,
        started_at=0.0,
        finished_at=1.0,
    )

    validation_errors = mod.validate_candidate_row(malformed)
    assert {"schema", "corpus", "question_id", "candidate_id", "answer"}.issubset(
        validation_errors
    )
    assert "rescored_not_regenerated" in validation_errors
    assert "token_logprobs" in mod.validate_candidate_row(malformed)
    assert "uprm_marker_logprobs" in mod.validate_candidate_row(malformed)
    assert len(mod.read_complete_candidate_rows(path)) == 1
    assert blocked["honest_verdict"] == "blocked_gemma_gguf_cache"
    assert blocked["candidate_cache_built"] is False
    assert checks[-1].resource == "musr_candidate_checkpoints"
    assert checks[-1].available is False
    assert questions == []
    assert second_corpus_artifact["corpora_cached"] == [mod.CORPUS, "GPQA"]
    assert "n_cached_rows" in mod.artifact_schema_errors(impossible_built)
    assert "rescored_not_regenerated" in mod.artifact_schema_errors(impossible_built)
    for field in (
        "candidate_cache_built",
        "candidates_per_question",
        "field_principles",
        "has_per_token_logprobs",
        "honest_verdict",
        "n_cached_rows",
        "n_questions",
        "preconditions_checked",
        "rescored_not_regenerated",
        "spec_refs",
    ):
        assert field in mod.artifact_schema_errors(bad_artifact)
