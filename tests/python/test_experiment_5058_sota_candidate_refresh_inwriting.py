"""Tests for Exp 5058 SOTA MuSR candidate refresh with delayed constraints.

Spec refs: REQ-VERIFY-5058, SCENARIO-VERIFY-5058.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_5058_sota_candidate_refresh_inwriting as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "verification" / "spec.md"


def _gate_state(*, ready: bool = True, top_logprobs: bool = False) -> dict[str, Any]:
    usable = (
        [
            {
                "role": "flagship_moe",
                "hf_id": "unsloth/Qwen3.6-35B-A3B-GGUF",
                "model_path": "/models/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf",
            }
        ]
        if ready
        else []
    )
    return {
        "schema": "carnot.experiment_5057_gate_state_preflight_v465.v1",
        "honest_verdict": "complete_gate_state_preflight_partial_ready",
        "sota_models_ready": ready,
        "top_logprob_or_confidence_ready": top_logprobs,
        "sota_judge_ready": bool(ready and top_logprobs),
        "usable_sota_models": usable,
        "legacy_models_smoke_only": True,
        "model_specs": {
            "flagship_moe": {
                "hf_id": "unsloth/Qwen3.6-35B-A3B-GGUF",
                "preferred_quant": "Q4_K_M",
                "resolved_path": "/models/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf" if ready else None,
                "missing_diagnostic": None if ready else "missing",
            },
            "flagship_dense": {
                "hf_id": "unsloth/gemma-4-31B-it-GGUF",
                "preferred_quant": "Q4_K_M",
                "resolved_path": None,
                "missing_diagnostic": "missing",
            },
            "middle_moe": {
                "hf_id": "unsloth/gemma-4-26B-A4B-it-GGUF",
                "preferred_quant": "Q4_K_M",
                "resolved_path": None,
                "missing_diagnostic": "missing",
            },
        },
        "endpoint_summary": {
            "completion_ready": False,
            "top_logprob_ready": False,
            "confidence_ready": False,
        },
    }


def _frozen_rows() -> list[dict[str, Any]]:
    return [
        {
            "schema": "carnot.shared_logprob_candidate_cache_v2.candidate_row.v1",
            "corpus": "MuSR/murder_mysteries",
            "question_id": "MuSR/murder_mysteries:0",
            "question_index": 0,
            "candidate_id": "MuSR/murder_mysteries:0/cached-0",
            "candidate_index": 0,
            "question": "Who is the most likely murderer?",
            "context": "A short case involving Mackenzie and Ana.",
            "choices": ["Mackenzie", "Ana"],
            "answer": "Mackenzie",
            "gold": "Mackenzie",
            "source": "distributional_energy_verifier_musr_checkpoints",
        },
        {
            "schema": "carnot.shared_logprob_candidate_cache_v2.candidate_row.v1",
            "corpus": "MuSR/murder_mysteries",
            "question_id": "MuSR/murder_mysteries:0",
            "question_index": 0,
            "candidate_id": "MuSR/murder_mysteries:0/cached-1",
            "candidate_index": 1,
            "question": "Who is the most likely murderer?",
            "context": "A short case involving Mackenzie and Ana.",
            "choices": ["Mackenzie", "Ana"],
            "answer": "Ana",
            "gold": "Mackenzie",
            "source": "distributional_energy_verifier_musr_checkpoints",
        },
        {
            "schema": "carnot.shared_logprob_candidate_cache_v2.candidate_row.v1",
            "corpus": "MuSR/murder_mysteries",
            "question_id": "MuSR/murder_mysteries:1",
            "question_index": 1,
            "candidate_id": "MuSR/murder_mysteries:1/cached-0",
            "candidate_index": 0,
            "question": "Who is the most likely murderer?",
            "context": "A short case involving Blair and Casey.",
            "choices": ["Blair", "Casey"],
            "answer": "Casey",
            "gold": "Casey",
            "source": "distributional_energy_verifier_musr_checkpoints",
        },
    ]


def _write_gate(root: Path, payload: dict[str, Any]) -> Path:
    path = root / mod.GATE_STATE_RELATIVE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    return path


def test_req_verify_5058_spec_declares_delayed_constraint_cache_contract() -> None:
    """REQ-VERIFY-5058: OpenSpec anchors the Exp5058 cache and artifact."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    for marker in (
        "REQ-VERIFY-5058",
        "SCENARIO-VERIFY-5058",
        "experiment_5058_sota_candidate_refresh_inwriting.py",
        "results/experiment_5058_sota_candidate_refresh_inwriting.json",
        "results/experiment_5058_sota_candidate_refresh_inwriting.jsonl",
        "blocked_sota_models_unavailable",
        "delayed_constraints_used",
        "used_top_logprobs",
        "duplicate_rate",
        "answer_diversity",
        "unsloth/Qwen3.6-35B-A3B-GGUF",
        "legacy_models_smoke_only",
    ):
        assert marker in spec
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert field in spec
        assert principle["principle"] in spec


def test_scenario_verify_5058_blocks_fast_when_sota_models_unavailable(
    tmp_path: Path,
) -> None:
    """SCENARIO-VERIFY-5058: no mandated SOTA model writes the exact blocker."""

    _write_gate(tmp_path, _gate_state(ready=False))
    artifact = mod.run(
        root=tmp_path,
        artifact_path=tmp_path / mod.RESULT_RELATIVE_PATH,
        cache_path=tmp_path / mod.CACHE_RELATIVE_PATH,
        frozen_candidate_loader=lambda _root: _frozen_rows(),
        now=lambda: 10.0,
        write=True,
    )

    assert artifact["honest_verdict"] == "blocked_sota_models_unavailable"
    assert artifact["candidate_refresh_ready"] is False
    assert artifact["n_questions"] == 0
    assert artifact["n_candidates"] == 0
    assert artifact["candidate_cache_path"].endswith(".jsonl")
    assert artifact["used_top_logprobs"] is False
    assert artifact["delayed_constraints_used"] is False
    assert artifact["legacy_models_smoke_only"] is True
    assert mod.artifact_schema_errors(artifact) == []
    assert json.loads((tmp_path / mod.RESULT_RELATIVE_PATH).read_text(encoding="utf-8")) == artifact
    assert not (tmp_path / mod.CACHE_RELATIVE_PATH).exists()


def test_scenario_verify_5058_top_logprob_absence_uses_delayed_constraints(
    tmp_path: Path,
) -> None:
    """SCENARIO-VERIFY-5058: absent top-logprobs do not block cache refresh."""

    _write_gate(tmp_path, _gate_state(ready=True, top_logprobs=False))
    artifact = mod.run(
        root=tmp_path,
        artifact_path=tmp_path / mod.RESULT_RELATIVE_PATH,
        cache_path=tmp_path / mod.CACHE_RELATIVE_PATH,
        frozen_candidate_loader=lambda _root: _frozen_rows(),
        now=lambda: 20.0,
        write=True,
    )
    rows = mod.read_complete_candidate_rows(tmp_path / mod.CACHE_RELATIVE_PATH)

    assert artifact["honest_verdict"] == "complete_sota_candidate_refresh_ready_d1_d6"
    assert artifact["candidate_refresh_ready"] is True
    assert artifact["n_questions"] == 2
    assert artifact["n_candidates"] == 3
    assert artifact["parse_rate"] == pytest.approx(1.0)
    assert artifact["duplicate_rate"] == pytest.approx(1.0)
    assert artifact["answer_diversity"]["unique_answers"] == 3
    assert artifact["used_top_logprobs"] is False
    assert artifact["delayed_constraints_used"] is True
    assert artifact["legacy_models_smoke_only"] is True
    assert artifact["d1_d6_readiness"]["ready"] is True
    assert mod.artifact_schema_errors(artifact) == []
    assert len(rows) == 3
    assert all(row["model_id"] == "unsloth/Qwen3.6-35B-A3B-GGUF" for row in rows)
    assert all(row["parse_status"] == "parsed" for row in rows)
    assert all(len(row["prompt_hash"]) == 64 for row in rows)
    assert all(row["structured_constraints"]["schema_name"] == "musr_delayed_answer_v1" for row in rows)
    assert rows[0]["decoding_parameters"]["constraint_timing"] == "delayed_after_draft"
    assert json.loads((tmp_path / mod.RESULT_RELATIVE_PATH).read_text(encoding="utf-8")) == artifact


def test_scenario_verify_5058_resume_skips_complete_rows(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-5058: resume appends missing rows without duplication."""

    cache_path = tmp_path / mod.CACHE_RELATIVE_PATH
    model = mod.select_headline_model(_gate_state(ready=True))
    existing = mod.build_candidate_row(_frozen_rows()[0], model_spec=model, used_top_logprobs=False)
    mod.append_jsonl_row(cache_path, existing)
    rows, resume = mod.ensure_candidate_cache(
        cache_path=cache_path,
        frozen_rows=_frozen_rows(),
        model_spec=model,
        used_top_logprobs=False,
    )

    assert len(rows) == 3
    assert resume == {
        "existing_complete_rows": 1,
        "appended_rows": 2,
        "target_rows": 3,
        "skipped_existing_rows": 1,
    }
    persisted_ids = [
        json.loads(line)["row_id"]
        for line in cache_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert persisted_ids.count(existing["row_id"]) == 1
    assert len(set(persisted_ids)) == 3


def test_req_verify_5058_parser_metrics_and_schema_edges(tmp_path: Path) -> None:
    """REQ-VERIFY-5058: delayed schema parsing and validators fail closed."""

    parsed = mod.parse_delayed_constraints(
        '{"answer":"Ana","evidence_spans":["car cleaning"],"confidence":0.61}',
        ["Mackenzie", "Ana"],
    )
    coerced = mod.parse_delayed_constraints("I think it was Mackenzie.", ["Mackenzie", "Ana"])
    failed = mod.parse_delayed_constraints("Unknown suspect", ["Mackenzie", "Ana"])

    assert parsed["parse_status"] == "parsed"
    assert parsed["parsed_answer"] == "Ana"
    assert parsed["structured_constraints"]["raw_format"] == "json_object"
    assert coerced["parse_status"] == "parsed"
    assert coerced["parsed_answer"] == "Mackenzie"
    assert failed["parse_status"] == "parse_failed"
    assert failed["parsed_answer"] == ""

    model = mod.select_headline_model(_gate_state(ready=True))
    bad_row = mod.build_candidate_row(
        {**_frozen_rows()[0], "answer": "Unknown suspect"},
        model_spec=model,
        used_top_logprobs=True,
    )
    assert bad_row["parse_status"] == "parse_failed"
    assert "parsed_answer" in mod.validate_candidate_row({**bad_row, "parsed_answer": ""})
    assert "schema" in mod.validate_candidate_row({**bad_row, "schema": "bad"})
    assert "structured_constraints" in mod.validate_candidate_row(
        {**bad_row, "structured_constraints": []}
    )
    assert "model_id" in mod.validate_candidate_row({**bad_row, "model_id": "legacy/tiny"})
    assert "prompt_hash" in mod.validate_candidate_row({**bad_row, "prompt_hash": "short"})
    assert "decoding_parameters" in mod.validate_candidate_row(
        {**bad_row, "decoding_parameters": []}
    )
    assert "legacy_model_used" in mod.validate_candidate_row(
        {**bad_row, "legacy_model_used": True}
    )

    rows = [
        mod.build_candidate_row(row, model_spec=model, used_top_logprobs=False)
        for row in _frozen_rows()
    ]
    metrics = mod.compute_refresh_metrics(rows, _frozen_rows())
    assert metrics["parse_rate"] == pytest.approx(1.0)
    assert metrics["duplicate_rate"] == pytest.approx(1.0)
    assert metrics["answer_diversity"]["unique_answer_rate"] == pytest.approx(1.0)

    artifact = mod.build_blocked_artifact(
        root=tmp_path,
        artifact_path=tmp_path / mod.RESULT_RELATIVE_PATH,
        cache_path=tmp_path / mod.CACHE_RELATIVE_PATH,
        gate_state={},
        duration_s=0.0,
        reason="sota_models_unavailable",
    )
    mutations = (
        ({key: value for key, value in artifact.items() if key != "parse_rate"}, "parse_rate"),
        ({**artifact, "schema": "wrong"}, "schema"),
        ({**artifact, "spec_refs": []}, "spec_refs"),
        ({**artifact, "model_specs": []}, "model_specs"),
        ({**artifact, "candidate_refresh_ready": "yes"}, "candidate_refresh_ready"),
        ({**artifact, "legacy_models_smoke_only": False}, "legacy_models_smoke_only"),
        ({**artifact, "n_questions": -1}, "n_questions"),
        ({**artifact, "n_candidates": "3"}, "n_candidates"),
        ({**artifact, "parse_rate": 2.0}, "parse_rate"),
        ({**artifact, "duplicate_rate": True}, "duplicate_rate"),
        ({**artifact, "answer_diversity": []}, "answer_diversity"),
        ({**artifact, "candidate_cache_path": ""}, "candidate_cache_path"),
        ({**artifact, "honest_verdict": "maybe"}, "honest_verdict"),
    )
    for mutated, field in mutations:
        assert field in mod.artifact_schema_errors(mutated)


def test_req_verify_5058_loaders_handle_missing_and_malformed_inputs(tmp_path: Path) -> None:
    """REQ-VERIFY-5058: local loaders and model selection expose exact failures."""

    with pytest.raises(FileNotFoundError):
        mod.load_gate_state(tmp_path)

    gate_path = _write_gate(tmp_path, _gate_state(ready=True))
    assert mod.load_gate_state(tmp_path)["sota_models_ready"] is True
    gate_path.write_text("{bad json", encoding="utf-8")
    with pytest.raises(ValueError):
        mod.load_gate_state(tmp_path)
    gate_path.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError):
        mod.load_gate_state(tmp_path)

    assert mod.select_headline_model(_gate_state(ready=False)) is None
    no_usable = {**_gate_state(ready=True), "usable_sota_models": []}
    assert mod.select_headline_model(no_usable) is None
    fallback_role = {
        **_gate_state(ready=True),
        "usable_sota_models": [
            {
                "role": "unexpected_role",
                "hf_id": "unsloth/gemma-4-31B-it-GGUF",
                "model_path": "/models/gemma.gguf",
            }
        ],
    }
    assert mod.select_headline_model(fallback_role)["role"] == "unexpected_role"
    assert mod.default_frozen_candidate_loader(tmp_path / "missing") == []

    frozen_path = tmp_path / mod.FROZEN_CANDIDATE_CACHE_RELATIVE_PATH
    frozen_path.parent.mkdir(parents=True, exist_ok=True)
    other_corpus = {**_frozen_rows()[0], "corpus": "Other"}
    frozen_path.write_text(
        "\n".join(
            [
                "",
                json.dumps(_frozen_rows()[0], sort_keys=True),
                json.dumps(other_corpus, sort_keys=True),
            ]
        ),
        encoding="utf-8",
    )
    assert len(mod.default_frozen_candidate_loader(tmp_path)) == 1
    frozen_path.write_text(
        "\n".join(["", json.dumps(_frozen_rows()[0], sort_keys=True), "{bad json"]),
        encoding="utf-8",
    )
    with pytest.raises(ValueError):
        mod.default_frozen_candidate_loader(tmp_path)

    assert mod.parse_delayed_constraints("", ["A"])["parse_status"] == "parse_failed"
    odd_row = {
        **_frozen_rows()[0],
        "question_id": "",
        "question_index": "bad",
        "candidate_index": "bad",
        "choices": "not-a-list",
        "answer": "Mackenzie",
    }
    odd_candidate = mod.build_candidate_row(
        odd_row,
        model_spec=mod.select_headline_model(_gate_state(ready=True)),
        used_top_logprobs=False,
    )
    assert odd_candidate["question_id"] == "MuSR/murder_mysteries:0"
    assert odd_candidate["candidate_index"] == 0
    assert odd_candidate["choices"] == []
