"""Tests for Exp 5075 DCCD guided decoding scale frontier.

Spec refs: REQ-VERIFY-5075, SCENARIO-VERIFY-5075.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_5075_dccd_guided_decoding_scale as exp


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


def _model_specs() -> dict[str, dict[str, str | None]]:
    return {
        spec["role"]: {
            "hf_id": spec["hf_id"],
            "resolved_path": f"/models/{spec['role']}.gguf",
            "missing_diagnostic": None,
        }
        for spec in exp.MODEL_SPECS
    }


def _rows(n_questions: int = 6) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for question_index in range(n_questions):
        gold = "A" if question_index % 2 == 0 else "B"
        other = "B" if gold == "A" else "A"
        answers = ["outside_schema" if question_index % 2 == 0 else other, gold, other]
        for candidate_index, answer in enumerate(answers):
            rows.append(
                {
                    "schema": "carnot.experiment_5058_sota_candidate_refresh_inwriting.row.v1",
                    "row_id": f"fixture:{question_index}/sota5058-{candidate_index:04d}",
                    "question_id": f"fixture:{question_index}",
                    "question_index": question_index,
                    "candidate_index": candidate_index,
                    "corpus": "MuSR/murder_mysteries",
                    "question": "Who is the most likely murderer?",
                    "choices": ["A", "B"],
                    "gold": gold,
                    "answer_text": answer,
                    "parsed_answer": answer,
                    "parse_status": "parsed",
                    "prompt_hash": f"prompt-{question_index:03d}",
                    "model_id": exp.MODEL_SPECS[0]["hf_id"],
                    "model_role": exp.MODEL_SPECS[0]["role"],
                    "model_path": "/models/qwen.gguf",
                    "legacy_model_used": False,
                    "delayed_constraints_used": True,
                    "structured_constraints": {
                        "allowed_answers": ["A", "B"],
                        "answer_in_allowed_choices": answer in {"A", "B"},
                        "constraint_checks": {
                            "allowed_choice": answer in {"A", "B"},
                            "delayed_after_draft": True,
                            "nonempty_draft": True,
                            "schema_parseable": False,
                        },
                    },
                    "source_provenance": {
                        "source": "fixture",
                        "source_candidate_id": f"cached-{candidate_index}",
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
        "candidate_cache_path": (root / exp.EXP5058_CACHE_RELATIVE_PATH).as_posix(),
        "n_questions": 6 if ready else 0,
        "n_candidates": 18 if ready else 0,
        "legacy_models_smoke_only": True,
        "flagged_adversarial": True,
        "model_specs": _model_specs(),
    }


def _exp5059(*, best_arm_available: bool = True, n_questions: int = 6) -> dict[str, Any]:
    predictions = ["A" if idx % 2 == 0 else "B" for idx in range(n_questions)]
    return {
        "schema": "carnot.experiment_5059_d1_sota_refresh_audit.v1",
        "honest_verdict": "complete_d1_sota_refresh_audit_no_proper_win_plus_0p080",
        "best_arm_available": best_arm_available,
        "legacy_models_smoke_only": True,
        "model_specs": {"mandated_sota": {spec["role"]: spec["hf_id"] for spec in exp.MODEL_SPECS}},
        "refreshed_candidate_metrics": {
            "n_questions": n_questions,
            "predictions": predictions,
            "paired_correct": {"verifier": [1] * n_questions},
        },
    }


def _exp5071(*, live: bool = False) -> dict[str, Any]:
    return {
        "schema": "carnot.experiment_5071_gguf_logprob_preflight.v466",
        "honest_verdict": "complete_gguf_logprob_preflight_ready"
        if live
        else "complete_gguf_logprob_preflight_partial_ready",
        "inference_substrate": "live_local_sota_endpoint" if live else "deterministic_verifier",
        "sota_models_ready": True,
        "completion_endpoint_ready": live,
        "logprob_endpoint_ready": live,
        "top_logprob_or_confidence_ready": live,
        "live_completion_invoked": live,
        "flagged_adversarial": False,
        "model_specs": {
            "headline_models": [spec["hf_id"] for spec in exp.MODEL_SPECS],
            "resolved_models": _model_specs(),
        },
    }


def _setup_root(tmp_path: Path, *, n_questions: int = 6, live: bool = False) -> Path:
    root = tmp_path / "root"
    _write_json(root / exp.EXP5058_RESULT_RELATIVE_PATH, _exp5058(root))
    _write_json(root / exp.EXP5059_RESULT_RELATIVE_PATH, _exp5059(n_questions=n_questions))
    _write_json(root / exp.EXP5071_RESULT_RELATIVE_PATH, _exp5071(live=live))
    _write_jsonl(root / exp.EXP5058_CACHE_RELATIVE_PATH, _rows(n_questions))
    return root


def test_req_verify_5075_spec_declares_dccd_frontier_contract() -> None:
    """REQ-VERIFY-5075: OpenSpec anchors Exp 5075 before implementation."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    module_text = (REPO / exp.MODULE_RELATIVE_PATH).read_text(encoding="utf-8")

    for marker in (
        "REQ-VERIFY-5075",
        "SCENARIO-VERIFY-5075",
        "experiment_5075_dccd_guided_decoding_scale.py",
        "results/experiment_5075_dccd_guided_decoding_scale_v466.json",
        "unguided",
        "hard_constrained",
        "reward_guided",
        "dccd",
        "rerank_only",
        "token_budget_by_arm",
        "ci95_delta",
    ):
        assert marker in spec
    for field in exp.REQUIRED_ARTIFACT_FIELDS:
        assert field in spec
    for spec_row in exp.MODEL_SPECS:
        assert spec_row["hf_id"] in spec
        assert spec_row["hf_id"] in module_text


def test_scenario_verify_5075_blocks_when_required_upstream_missing(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-5075: missing refresh or best-arm evidence fails closed."""

    root = tmp_path / "root"
    _write_json(root / exp.EXP5058_RESULT_RELATIVE_PATH, _exp5058(root, ready=False))
    _write_json(root / exp.EXP5059_RESULT_RELATIVE_PATH, _exp5059())
    _write_json(root / exp.EXP5071_RESULT_RELATIVE_PATH, _exp5071())

    refresh_blocked = exp.run(root=root, artifact_path=tmp_path / "refresh.json", write=True)

    assert refresh_blocked["honest_verdict"] == "blocked_dccd_guided_frontier_candidate_refresh_unavailable"
    assert refresh_blocked["n_questions"] == 0
    assert refresh_blocked["beats_rerank_only"] is False
    assert all(value is False for value in refresh_blocked["live_local_sota_inference_by_arm"].values())
    assert exp.artifact_schema_errors(refresh_blocked) == []
    assert json.loads((tmp_path / "refresh.json").read_text(encoding="utf-8")) == refresh_blocked

    _write_json(root / exp.EXP5058_RESULT_RELATIVE_PATH, _exp5058(root, ready=True))
    _write_json(root / exp.EXP5059_RESULT_RELATIVE_PATH, _exp5059(best_arm_available=False))

    arm_blocked = exp.run(root=root, artifact_path=tmp_path / "arm.json", write=False)

    assert arm_blocked["honest_verdict"] == "blocked_dccd_guided_frontier_rerank_unavailable"
    assert arm_blocked["n_questions"] == 0
    assert exp.artifact_schema_errors(arm_blocked) == []


def test_dccd_selector_drafts_then_enforces_structure_without_gold() -> None:
    """REQ-VERIFY-5075: DCCD repair is draft-conditioned, not answer-key driven."""

    pool = _rows(1)[:3]
    draft = exp._select_unguided(pool, seed=20260701)

    selected, metadata = exp._select_dccd(pool, seed=20260701)

    assert exp._candidate_answer(draft) == "outside_schema"
    assert metadata["semantic_draft_answer"] == "outside_schema"
    assert exp._candidate_answer(selected) == "A"
    assert metadata["draft_structurally_valid"] is False
    assert metadata["structural_enforcement_applied"] is True
    assert metadata["uses_gold"] is False


def test_scenario_verify_5075_runs_underpowered_scale_frontier(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-5075: replay scale records all arms, costs, and CI."""

    root = _setup_root(tmp_path, n_questions=6, live=False)
    artifact_path = tmp_path / "out.json"

    artifact = exp.run(
        root=root,
        artifact_path=artifact_path,
        max_questions=6,
        candidates_per_prompt=3,
        seed=20260701,
        write=True,
    )

    assert artifact["honest_verdict"] == "complete_dccd_guided_frontier_no_headline_underpowered"
    assert artifact["inference_substrate"] == exp.REPLAY_SUBSTRATE
    assert artifact["n_questions"] == 6
    assert set(artifact["arms"]) == set(exp.ARM_NAMES)
    assert artifact["guided_accuracy"] == artifact["arms"]["reward_guided"]["answer_accuracy"]
    assert artifact["dccd_accuracy"] == artifact["arms"]["dccd"]["answer_accuracy"]
    assert artifact["rerank_only_accuracy"] == pytest.approx(1.0)
    assert artifact["delta_dccd_vs_rerank"] <= 0.0
    assert len(artifact["ci95_delta"]) == 2
    assert artifact["ci95_delta"][0] <= artifact["delta_dccd_vs_rerank"] <= artifact["ci95_delta"][1]
    assert artifact["beats_rerank_only"] is False
    assert artifact["flagged_adversarial"] is False
    assert artifact["upstream_flagged_adversarial_sources"] == [
        exp.EXP5058_RESULT_RELATIVE_PATH
    ]
    assert artifact["sample_power"]["verdict"] == "underpowered_no_live_local_sota"

    for arm in exp.ARM_NAMES:
        summary = artifact["arms"][arm]
        assert artifact["token_budget_by_arm"][arm] == summary["generated_tokens"]
        assert artifact["nfe_by_arm"][arm] == summary["nfe"]
        assert artifact["live_local_sota_inference_by_arm"][arm] is False
        assert summary["parse_rate"] == pytest.approx(1.0)
        assert 0.0 <= summary["validity_rate"] <= 1.0
        assert summary["wall_time_s"] >= 0.0

    assert artifact["arms"]["dccd"]["validity_rate"] == pytest.approx(1.0)
    assert artifact["candidate_difference_rate_by_arm_vs_unguided"]["dccd"] > 0.0
    assert artifact["candidate_diffs"]["dccd_vs_rerank_only"] >= 0.0
    assert artifact["field_principles"].keys() >= set(exp.REQUIRED_ARTIFACT_FIELDS)
    assert exp.artifact_schema_errors(artifact) == []
    assert json.loads(artifact_path.read_text(encoding="utf-8")) == artifact


def test_artifact_schema_errors_reject_malformed_5075_fields(tmp_path: Path) -> None:
    """REQ-VERIFY-5075: malformed required accounting fields fail validation."""

    root = _setup_root(tmp_path)
    artifact = exp.run(root=root, artifact_path=tmp_path / "out.json", write=False)

    cases = [
        (lambda a: (a.pop("honest_verdict"), a)[1], "honest_verdict"),
        (lambda a: dict(a, schema="wrong"), "schema"),
        (lambda a: dict(a, spec_refs=["WRONG"]), "spec_refs"),
        (lambda a: dict(a, model_specs=[]), "model_specs"),
        (lambda a: dict(a, n_questions=-1), "n_questions"),
        (lambda a: dict(a, arms=[]), "arms"),
        (lambda a: dict(a, unguided_accuracy="bad"), "unguided_accuracy"),
        (lambda a: dict(a, dccd_accuracy=2.0), "dccd_accuracy"),
        (lambda a: dict(a, delta_dccd_vs_rerank="bad"), "delta_dccd_vs_rerank"),
        (lambda a: dict(a, ci95_delta=[0.0]), "ci95_delta"),
        (lambda a: dict(a, nfe_by_arm={"dccd": -1.0}), "nfe_by_arm"),
        (lambda a: dict(a, token_budget_by_arm={"dccd": 1.5}), "token_budget_by_arm"),
        (lambda a: dict(a, beats_rerank_only="no"), "beats_rerank_only"),
        (lambda a: dict(a, flagged_adversarial="no"), "flagged_adversarial"),
    ]

    for mutate, expected in cases:
        assert expected in exp.artifact_schema_errors(mutate(dict(artifact)))


def test_req_verify_5075_helper_edges_and_live_success_verdict(tmp_path: Path) -> None:
    """REQ-VERIFY-5075: helper edges and live-success verdicts stay explicit."""

    assert exp.read_jsonl(tmp_path / "missing.jsonl") == []
    malformed = tmp_path / "bad.jsonl"
    malformed.write_text("\nnot-json\n{\"ok\": true}\n", encoding="utf-8")
    assert exp.read_jsonl(malformed) == [{"ok": True}]
    assert exp._number(True) is None
    assert exp._rate(0, 0) == 0.0
    assert exp._accuracy([]) is None
    assert exp._paired_ci95_delta([], []) == [0.0, 0.0]
    assert exp._honest_verdict(
        live_local_sota=True,
        beats_rerank_only=True,
        delta=0.125,
        n_questions=50,
    ) == "success_dccd_guided_frontier_beats_rerank_plus_0p125"

    root = _setup_root(tmp_path, n_questions=2, live=True)
    too_small = exp.run(
        root=root,
        artifact_path=tmp_path / "small.json",
        max_questions=2,
        candidates_per_prompt=3,
        write=False,
    )

    assert too_small["honest_verdict"] == "blocked_dccd_guided_frontier_matched_sample_too_small"
    assert too_small["n_questions"] == 2
    assert exp.artifact_schema_errors(too_small) == []
