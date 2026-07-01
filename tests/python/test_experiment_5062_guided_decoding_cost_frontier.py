"""Tests for Exp 5062 guided decoding cost frontier.

Spec refs: REQ-VERIFY-5062, SCENARIO-VERIFY-5062.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_5062_guided_decoding_cost_frontier as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "verification" / "spec.md"


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _prompt_hash(question_id: str) -> str:
    return hashlib.sha256(f"fixture prompt {question_id}".encode()).hexdigest()


def _question_fixture() -> list[tuple[str, str, list[str], list[str]]]:
    return [
        ("fixture:0", "A", ["A", "B"], ["B", "A", "B"]),
        ("fixture:1", "C", ["C", "D"], ["D", "C", "D"]),
        ("fixture:2", "E", ["E", "F"], ["F", "E", "F"]),
        ("fixture:3", "G", ["G", "H"], ["H", "G", "H"]),
    ]


def _frozen_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for question_index, (question_id, gold, choices, answers) in enumerate(_question_fixture()):
        for candidate_index, answer in enumerate(answers):
            rows.append(
                {
                    "schema": "carnot.shared_logprob_candidate_cache_v2.candidate_row.v1",
                    "candidate_id": f"{question_id}/cached-{candidate_index}",
                    "question_id": question_id,
                    "question_index": question_index,
                    "candidate_index": candidate_index,
                    "corpus": "MuSR/murder_mysteries",
                    "question": "Who is the most likely murderer?",
                    "choices": choices,
                    "answer": answer,
                    "gold": gold,
                    "source": "fixture_frozen_464_cache",
                }
            )
    return rows


def _refreshed_rows(*, duplicate_answers_only: bool = False) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for frozen in _frozen_rows():
        answer = str(frozen["answer"])
        if duplicate_answers_only:
            answer = str(frozen["choices"][0])
        question_id = str(frozen["question_id"])
        candidate_index = int(frozen["candidate_index"])
        rows.append(
            {
                "schema": "carnot.experiment_5058_sota_candidate_refresh_inwriting.row.v1",
                "row_id": f"{question_id}/sota5058-{candidate_index:04d}",
                "question_id": question_id,
                "question_index": int(frozen["question_index"]),
                "candidate_index": candidate_index,
                "corpus": "MuSR/murder_mysteries",
                "question": frozen["question"],
                "choices": list(frozen["choices"]),
                "answer_text": answer,
                "parsed_answer": answer,
                "parse_status": "parsed",
                "prompt_hash": _prompt_hash(question_id),
                "model_id": "unsloth/Qwen3.6-35B-A3B-GGUF",
                "model_role": "flagship_moe",
                "model_path": "/models/qwen.gguf",
                "legacy_model_used": False,
                "decoding_parameters": {"temperature": 0.7, "max_tokens": 64},
                "structured_constraints": {
                    "allowed_answers": list(frozen["choices"]),
                    "answer_in_allowed_choices": answer in frozen["choices"],
                    "constraint_checks": {"allowed_choice": answer in frozen["choices"]},
                },
                "source_provenance": {
                    "source": "frozen_464_musr_candidate_cache",
                    "source_candidate_id": frozen["candidate_id"],
                    "source_answer_text": answer,
                },
            }
        )
    return rows


def _exp5058(root: Path, *, ready: bool = True) -> dict[str, Any]:
    return {
        "schema": "carnot.experiment_5058_sota_candidate_refresh_inwriting.v1",
        "honest_verdict": "complete_sota_candidate_refresh_ready_d1_d6"
        if ready
        else "blocked_sota_models_unavailable",
        "candidate_refresh_ready": ready,
        "candidate_cache_path": (root / mod.EXP5058_CACHE_RELATIVE_PATH).as_posix(),
        "n_questions": 4 if ready else 0,
        "n_candidates": 12 if ready else 0,
        "legacy_models_smoke_only": True,
        "model_specs": {
            role: {"hf_id": hf_id, "resolved_path": f"/models/{role}.gguf"}
            for role, hf_id in mod.MANDATED_MODEL_SPECS.items()
        },
    }


def _exp5059(*, best_arm_available: bool = True) -> dict[str, Any]:
    return {
        "schema": "carnot.experiment_5059_d1_sota_refresh_audit.v1",
        "honest_verdict": "complete_d1_sota_refresh_audit_no_proper_win_plus_0p500",
        "best_arm_available": best_arm_available,
        "verifier_is_oracle": False,
        "legacy_models_smoke_only": True,
        "model_specs": {"mandated_sota": dict(mod.MANDATED_MODEL_SPECS)},
        "refreshed_candidate_metrics": {
            "predictions": ["A", "C", "E", "G"],
            "paired_correct": {
                "verifier": [1, 1, 1, 1],
                "tuned_self_consistency": [0, 0, 0, 0],
            },
        },
    }


def _setup_root(tmp_path: Path, *, duplicate_answers_only: bool = False) -> Path:
    root = tmp_path / "root"
    _write_json(root / mod.EXP5058_RESULT_RELATIVE_PATH, _exp5058(root))
    _write_json(root / mod.EXP5059_RESULT_RELATIVE_PATH, _exp5059())
    _write_jsonl(
        root / mod.EXP5058_CACHE_RELATIVE_PATH,
        _refreshed_rows(duplicate_answers_only=duplicate_answers_only),
    )
    _write_jsonl(root / mod.FROZEN_CANDIDATE_CACHE_RELATIVE_PATH, _frozen_rows())
    return root


def test_req_verify_5062_spec_declares_guided_frontier_contract() -> None:
    """REQ-VERIFY-5062: OpenSpec anchors the guided frontier artifact."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    module_text = (REPO / mod.MODULE_RELATIVE_PATH).read_text(encoding="utf-8")

    for marker in (
        "REQ-VERIFY-5062",
        "SCENARIO-VERIFY-5062",
        "experiment_5062_guided_decoding_cost_frontier.py",
        "results/experiment_5062_guided_decoding_cost_frontier.json",
        "rerank-only",
        "candidate_difference_rate",
        "NFE",
        "legacy_models_smoke_only",
    ):
        assert marker in spec
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert field in spec
    for hf_id in mod.MANDATED_MODEL_SPECS.values():
        assert hf_id in spec
        assert hf_id in module_text


def test_scenario_verify_5062_blocks_when_upstream_gates_missing(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-5062: missing refresh or best arm fails fast."""

    root = tmp_path / "root"
    _write_json(root / mod.EXP5058_RESULT_RELATIVE_PATH, _exp5058(root, ready=False))
    _write_json(root / mod.EXP5059_RESULT_RELATIVE_PATH, _exp5059())

    refresh_blocked = mod.run(root=root, artifact_path=tmp_path / "refresh.json", write=True)

    assert refresh_blocked["honest_verdict"] == "blocked_candidate_refresh_unavailable"
    assert refresh_blocked["guided_decoding_executed"] is False
    assert refresh_blocked["arms_differentiated"] is False
    assert refresh_blocked["candidate_difference_rate"] == 0.0
    assert refresh_blocked["guided_accuracy"] is None
    assert mod.artifact_schema_errors(refresh_blocked) == []
    assert json.loads((tmp_path / "refresh.json").read_text(encoding="utf-8")) == refresh_blocked

    _write_json(root / mod.EXP5058_RESULT_RELATIVE_PATH, _exp5058(root, ready=True))
    _write_json(root / mod.EXP5059_RESULT_RELATIVE_PATH, _exp5059(best_arm_available=False))

    arm_blocked = mod.run(root=root, artifact_path=tmp_path / "arm.json", write=False)

    assert arm_blocked["honest_verdict"] == "blocked_exp5059_best_arm_unavailable"
    assert arm_blocked["guided_decoding_executed"] is False
    assert arm_blocked["arms_differentiated"] is False
    assert mod.artifact_schema_errors(arm_blocked) == []


def test_scenario_verify_5062_executes_differentiated_frontier(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-5062: guided, unguided, and rerank-only arms are separated."""

    root = _setup_root(tmp_path)
    artifact_path = tmp_path / "out.json"

    artifact = mod.run(
        root=root,
        artifact_path=artifact_path,
        max_prompts=4,
        candidates_per_prompt=3,
        seed=20260701,
        write=True,
    )

    assert artifact["honest_verdict"].startswith("complete_guided_decoding_cost_frontier")
    assert artifact["guided_decoding_executed"] is True
    assert artifact["arms_differentiated"] is True
    assert artifact["candidate_difference_rate"] > 0.0
    assert artifact["matched_prompt_count"] == 4
    assert artifact["legacy_models_smoke_only"] is True
    assert artifact["model_specs"]["headline_generation_model"]["hf_id"] in (
        mod.MANDATED_MODEL_SPECS.values()
    )
    assert artifact["guided_accuracy"] is not None
    assert artifact["unguided_accuracy"] is not None
    assert artifact["rerank_only_accuracy"] == pytest.approx(1.0)
    assert artifact["delta_guided_vs_unguided"] == pytest.approx(
        artifact["guided_accuracy"] - artifact["unguided_accuracy"]
    )
    assert set(artifact["generated_tokens_by_arm"]) == set(mod.ARM_NAMES)
    assert set(artifact["nfe_by_arm"]) == set(mod.ARM_NAMES)
    assert artifact["verifier_calls_by_arm"]["rerank_only"] > 0
    assert "judge_calls_by_arm" in artifact
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE

    for row in artifact["matched_rows"]:
        assert row["prompt_hash"]
        assert row["seed"] >= 20260701
        assert row["guided"]["prompt_hash"] == row["unguided"]["prompt_hash"]
        assert row["guided"]["seed"] == row["unguided"]["seed"]
        assert row["guided"]["model_family"] == row["unguided"]["model_family"]
        assert row["guided"]["candidate_hash"]
        assert row["unguided"]["candidate_hash"]
        assert row["rerank_only"]["selection_stage"] == "post_generation_rerank"

    assert mod.artifact_schema_errors(artifact) == []
    assert json.loads(artifact_path.read_text(encoding="utf-8")) == artifact


def test_controls_not_differentiated_when_candidate_hashes_match(tmp_path: Path) -> None:
    """REQ-VERIFY-5062: identical guided/unguided candidates block frontier claims."""

    root = _setup_root(tmp_path, duplicate_answers_only=True)

    artifact = mod.run(
        root=root,
        artifact_path=tmp_path / "out.json",
        max_prompts=4,
        candidates_per_prompt=3,
        write=False,
    )

    assert artifact["guided_decoding_executed"] is True
    assert artifact["arms_differentiated"] is False
    assert artifact["candidate_difference_rate"] == 0.0
    assert artifact["honest_verdict"] == "complete_guided_decoding_controls_not_differentiated"
    assert mod.artifact_schema_errors(artifact) == []


def test_artifact_schema_errors_report_bad_guided_frontier_fields(tmp_path: Path) -> None:
    """REQ-VERIFY-5062: schema validation catches malformed required fields."""

    root = _setup_root(tmp_path)
    artifact = mod.run(root=root, artifact_path=tmp_path / "out.json", write=False)

    cases = [
        (lambda a: (a.pop("honest_verdict"), a)[1], "honest_verdict"),
        (lambda a: dict(a, schema="wrong"), "schema"),
        (lambda a: dict(a, spec_refs=["WRONG"]), "spec_refs"),
        (lambda a: dict(a, model_specs=[]), "model_specs"),
        (lambda a: dict(a, guided_decoding_executed="yes"), "guided_decoding_executed"),
        (lambda a: dict(a, arms_differentiated="yes"), "arms_differentiated"),
        (lambda a: dict(a, candidate_difference_rate=2.0), "candidate_difference_rate"),
        (lambda a: dict(a, guided_accuracy="bad"), "guided_accuracy"),
        (lambda a: dict(a, delta_guided_vs_unguided="bad"), "delta_guided_vs_unguided"),
        (lambda a: dict(a, generated_tokens_by_arm=[]), "generated_tokens_by_arm"),
        (lambda a: dict(a, nfe_by_arm={"guided": -1}), "nfe_by_arm"),
        (lambda a: dict(a, verifier_calls_by_arm={"guided": 1.5}), "verifier_calls_by_arm"),
        (lambda a: dict(a, latency_s_by_arm={"guided": -0.1}), "latency_s_by_arm"),
        (lambda a: dict(a, legacy_models_smoke_only=False), "legacy_models_smoke_only"),
    ]
    for mutate, expected in cases:
        assert expected in mod.artifact_schema_errors(mutate(dict(artifact)))


def test_req_verify_5062_helper_edges_and_blocked_matched_set(tmp_path: Path) -> None:
    """REQ-VERIFY-5062: helper edge cases fail closed with auditable fields."""

    missing_jsonl = tmp_path / "missing.jsonl"
    assert mod.read_jsonl(missing_jsonl) == []
    malformed_jsonl = tmp_path / "malformed.jsonl"
    malformed_jsonl.write_text("\nnot-json\n{\"ok\": true}\n", encoding="utf-8")
    assert mod.read_jsonl(malformed_jsonl) == [{"ok": True}]

    assert mod._number(True) is None
    assert mod._question_index({"question_index": "bad"}) == 0
    assert mod._candidate_index({"candidate_index": "bad"}) == 0
    assert mod._prompt_hash({"question_id": "qid", "question": "Q?", "choices": ["A"]})
    assert mod._refresh_cache_path(tmp_path, {}) == tmp_path / mod.EXP5058_CACHE_RELATIVE_PATH
    assert mod._gold_by_question([{"question_id": "q", "gold": "A"}], []) == {"q": "A"}
    assert mod._prediction_list({}) == []
    assert mod._round(None) is None
    assert mod._round(1 / 3) == pytest.approx(0.333333)

    rows = [
        {"question_id": "q0", "model_id": "bad", "answer": "A"},
        {
            "question_id": "q1",
            "model_id": "unsloth/Qwen3.6-35B-A3B-GGUF",
            "answer": "",
        },
        {
            "question_id": "q2",
            "model_id": "unsloth/Qwen3.6-35B-A3B-GGUF",
            "answer": "A",
            "legacy_model_used": True,
        },
        {
            "question_id": "q3",
            "model_id": "unsloth/Qwen3.6-35B-A3B-GGUF",
            "answer": "A",
        },
    ]
    assert mod._eligible_groups(rows, gold_by_question={}, candidates_per_prompt=2) == []

    pool = _refreshed_rows()[:2]
    fallback = mod._select_rerank_only(pool, prediction="missing", seed=1)
    assert fallback["row_id"] in {row["row_id"] for row in pool}

    assert mod._all_matched_rows_have_evidence([]) is False
    missing_prompt_rows = [
        {"seed": 1, "guided": {}, "unguided": {}}
    ] * mod.MIN_MATCHED_PROMPTS
    assert mod._all_matched_rows_have_evidence(missing_prompt_rows) is False
    nonmapping_arm_rows = [
        {"prompt_hash": "h", "seed": 1, "guided": [], "unguided": {}}
    ] * mod.MIN_MATCHED_PROMPTS
    assert mod._all_matched_rows_have_evidence(nonmapping_arm_rows) is False
    missing_hash_rows = [
        {
            "prompt_hash": "h",
            "seed": 1,
            "guided": {"prompt_hash": "h", "generated_tokens": 1},
            "unguided": {"candidate_hash": "c", "prompt_hash": "h", "generated_tokens": 1},
        }
    ] * mod.MIN_MATCHED_PROMPTS
    assert mod._all_matched_rows_have_evidence(missing_hash_rows) is False
    bad_token_rows = [
        {
            "prompt_hash": "h",
            "seed": 1,
            "guided": {"candidate_hash": "c", "prompt_hash": "h", "generated_tokens": "bad"},
            "unguided": {"candidate_hash": "c", "prompt_hash": "h", "generated_tokens": 1},
        }
    ] * mod.MIN_MATCHED_PROMPTS
    assert mod._all_matched_rows_have_evidence(bad_token_rows) is False
    bad_rows = [
        {
            "prompt_hash": "h",
            "seed": 1,
            "guided": {"candidate_hash": "c", "prompt_hash": "h", "generated_tokens": 1},
            "unguided": {
                "candidate_hash": "c",
                "prompt_hash": "h",
                "generated_tokens": 1,
                "model_id": "bad",
            },
        }
    ] * mod.MIN_MATCHED_PROMPTS
    assert mod._all_matched_rows_have_evidence(bad_rows) is False
    assert (
        mod._honest_verdict(arms_differentiated=True, delta=0.0)
        == "complete_guided_decoding_cost_frontier_no_improvement"
    )
    assert "guided_not_better" in mod._honest_verdict(
        arms_differentiated=True,
        delta=-0.25,
    )

    root = tmp_path / "blocked_root"
    _write_json(root / mod.EXP5058_RESULT_RELATIVE_PATH, _exp5058(root))
    _write_json(root / mod.EXP5059_RESULT_RELATIVE_PATH, _exp5059(best_arm_available=False))
    arm_path = tmp_path / "arm-blocked.json"
    arm_blocked = mod.run(root=root, artifact_path=arm_path, write=True)
    assert arm_blocked["honest_verdict"] == "blocked_exp5059_best_arm_unavailable"
    assert json.loads(arm_path.read_text(encoding="utf-8")) == arm_blocked

    _write_json(root / mod.EXP5059_RESULT_RELATIVE_PATH, _exp5059())
    _write_jsonl(root / mod.EXP5058_CACHE_RELATIVE_PATH, _refreshed_rows()[:3])
    _write_jsonl(root / mod.FROZEN_CANDIDATE_CACHE_RELATIVE_PATH, _frozen_rows()[:3])
    matched_blocked = mod.run(root=root, artifact_path=tmp_path / "matched.json", write=True)
    assert matched_blocked["honest_verdict"] == "blocked_matched_prompt_set_unavailable"
    assert mod.artifact_schema_errors(matched_blocked) == []
