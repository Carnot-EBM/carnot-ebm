"""Tests for Exp 5060 audited D4 second-corpus confirmation.

Spec refs: REQ-VERIFY-5060, SCENARIO-VERIFY-5060.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_5060_second_corpus_audit_v2 as mod


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


def _rows(
    *,
    n: int = 8,
    duplicate_sources: bool = False,
    leak_gold: bool = False,
    oracle_selection_feature: bool = False,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index in range(n):
        source_id = f"source-{index % 2}" if duplicate_sources else f"source-{index}"
        context = "Use the public constraints only."
        if leak_gold:
            context += " Gold label: A."
        rows.append(
            {
                "schema": "carnot.second_corpus_candidate_cache.row.v1",
                "row_id": f"cb::{index}",
                "source_row_id": source_id,
                "corpus": "ConstraintBench-exact-v1",
                "question": f"Solve fixture {index}",
                "context": context,
                "gold": "A",
                "label": "A",
                "candidates": [
                    {
                        "candidate_id": f"cb::{index}/wrong-0",
                        "answer": "B",
                        "cache_index": 0,
                        "temperature": "deterministic",
                        "label_correct": False,
                        "candidate_label": "incorrect",
                        "solver_score_used_for_selection": oracle_selection_feature,
                        "solver_verdict": {"objective_gap": 1.0},
                        "generation_model": None,
                    },
                    {
                        "candidate_id": f"cb::{index}/right-1",
                        "answer": "A",
                        "cache_index": 1,
                        "temperature": "deterministic",
                        "label_correct": True,
                        "candidate_label": "correct",
                        "solver_score_used_for_selection": oracle_selection_feature,
                        "solver_verdict": {"objective_gap": 0.0},
                        "generation_model": None,
                    },
                    {
                        "candidate_id": f"cb::{index}/wrong-2",
                        "answer": "B",
                        "cache_index": 2,
                        "temperature": "deterministic",
                        "label_correct": False,
                        "candidate_label": "incorrect",
                        "solver_score_used_for_selection": oracle_selection_feature,
                        "solver_verdict": {"objective_gap": 1.0},
                        "generation_model": None,
                    },
                    {
                        "candidate_id": f"cb::{index}/right-3",
                        "answer": "A",
                        "cache_index": 3,
                        "temperature": "deterministic",
                        "label_correct": True,
                        "candidate_label": "correct",
                        "solver_score_used_for_selection": oracle_selection_feature,
                        "solver_verdict": {"objective_gap": 0.0},
                        "generation_model": None,
                    },
                    {
                        "candidate_id": f"cb::{index}/wrong-4",
                        "answer": "B",
                        "cache_index": 4,
                        "temperature": "deterministic",
                        "label_correct": False,
                        "candidate_label": "incorrect",
                        "solver_score_used_for_selection": oracle_selection_feature,
                        "solver_verdict": {"objective_gap": 1.0},
                        "generation_model": None,
                    },
                ],
            }
        )
    return rows


def _write_5044(root: Path, rows: list[dict[str, Any]]) -> Path:
    cache_path = root / mod.EXP5044_CACHE_RELATIVE_PATH
    _write_jsonl(cache_path, rows)
    _write_json(
        root / mod.EXP5044_RESULT_RELATIVE_PATH,
        {
            "second_corpus_cache_built": True,
            "second_corpus_name": "ConstraintBench-exact-v1",
            "candidate_cache_path": cache_path.as_posix(),
            "n_questions": len(rows),
            "headroom_present": True,
            "verifier_is_oracle": False,
            "genuine_sc_accuracy": 0.0,
            "oracle_at_k": 1.0,
            "model_specs": {
                "candidate_generation": "deterministic_solver_backed_constraint_variants",
                "small_models_smoke_only": True,
            },
        },
    )
    return cache_path


def _write_5059(root: Path, *, proper_win: bool = True, available: bool = True) -> None:
    _write_json(
        root / mod.EXP5059_RESULT_RELATIVE_PATH,
        {
            "best_arm_available": available,
            "candidate_refresh_used": True,
            "proper_musr_win": proper_win,
            "verifier_is_oracle": False,
            "legacy_models_smoke_only": True,
            "checkpoint_path": "/checkpoints/d1/epoch_1",
            "model_specs": {
                "mandated_sota": dict(mod.MANDATED_MODEL_SPECS),
                "powered_d1_scorer": {
                    "checkpoint_path": "/checkpoints/d1/epoch_1",
                    "source": "results/experiment_5045_powered_lora_ebm_eorm_musr.json",
                },
            },
            "scorer_source": {
                "method": "cached_exp5045_powered_d1_selection_projection",
                "source_artifact": "results/experiment_5045_powered_lora_ebm_eorm_musr.json",
            },
        },
    )


def _score_correct(_checkpoint: str, texts: list[str]) -> list[float]:
    return [0.0 if "Candidate answer: A" in text else 1.0 for text in texts]


def test_req_verify_5060_spec_declares_audit_contract() -> None:
    """REQ-VERIFY-5060: OpenSpec anchors the audited D4 artifact."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    for marker in (
        "REQ-VERIFY-5060",
        "SCENARIO-VERIFY-5060",
        "experiment_5060_second_corpus_audit_v2.py",
        "results/experiment_5060_second_corpus_audit_v2.json",
        "row_hash_manifest",
        "second_corpus_audit_clean",
        "scoped clue",
        "retire",
    ):
        assert marker in spec
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert field in spec


def test_scenario_verify_5060_clean_confirmation_when_all_gates_pass(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-5060: clean D4 promotion requires audits and paired stats."""

    _write_5044(tmp_path, _rows())
    _write_5059(tmp_path, proper_win=True)
    artifact_path = tmp_path / mod.RESULT_RELATIVE_PATH

    artifact = mod.run(
        root=tmp_path,
        artifact_path=artifact_path,
        score_fn=_score_correct,
        bootstrap_samples=64,
        now=lambda: 100.0,
        write=True,
    )

    assert artifact["honest_verdict"].startswith("success_d4_clean_confirmation_")
    assert artifact["d4_verdict_class"] == "clean_confirmation"
    assert artifact["second_corpus_confirmed"] is True
    assert artifact["second_corpus_audit_clean"] is True
    assert artifact["leak_audit_passed"] is True
    assert artifact["oracle_provenance_passed"] is True
    assert artifact["duplicate_audit_passed"] is True
    assert artifact["legacy_models_smoke_only"] is True
    assert artifact["n_questions_second"] == 8
    assert artifact["delta_vs_tuned_sc_second"] == pytest.approx(1.0)
    assert artifact["paired_ci95_second"] == [1.0, 1.0]
    assert artifact["mcnemar_p_second"] == pytest.approx(0.007812)
    assert artifact["row_hash_manifest"]["n_rows"] == 8
    assert artifact["row_hash_manifest"]["n_duplicate_source_instances"] == 0
    assert mod.artifact_schema_errors(artifact) == []
    assert json.loads(artifact_path.read_text(encoding="utf-8")) == artifact


def test_scenario_verify_5060_positive_stats_become_scoped_clue_without_upstream_win(
    tmp_path: Path,
) -> None:
    """SCENARIO-VERIFY-5060: Exp5059 non-proper win prevents clean D4 confirmation."""

    _write_5044(tmp_path, _rows())
    _write_5059(tmp_path, proper_win=False)

    artifact = mod.run(
        root=tmp_path,
        artifact_path=tmp_path / "out.json",
        score_fn=_score_correct,
        bootstrap_samples=64,
        write=False,
    )

    assert artifact["honest_verdict"].startswith("complete_d4_scoped_clue_")
    assert artifact["d4_verdict_class"] == "scoped_clue"
    assert artifact["second_corpus_confirmed"] is False
    assert artifact["second_corpus_audit_clean"] is True
    assert artifact["upstream_exp5059_proper_win"] is False
    assert artifact["delta_vs_tuned_sc_second"] == pytest.approx(1.0)
    assert mod.artifact_schema_errors(artifact) == []


def test_scenario_verify_5060_retires_when_duplicate_or_oracle_audits_fail(
    tmp_path: Path,
) -> None:
    """SCENARIO-VERIFY-5060: audit failures retire rather than confirm D4."""

    duplicate_rows = _rows(duplicate_sources=True)
    _write_5044(tmp_path, duplicate_rows)
    _write_5059(tmp_path, proper_win=True)

    artifact = mod.run(
        root=tmp_path,
        artifact_path=tmp_path / "retired.json",
        score_fn=_score_correct,
        bootstrap_samples=64,
        write=False,
    )

    assert artifact["honest_verdict"].startswith("retired_d4_second_corpus_audit_failed_")
    assert artifact["d4_verdict_class"] == "retired"
    assert artifact["second_corpus_confirmed"] is False
    assert artifact["second_corpus_audit_clean"] is False
    assert artifact["duplicate_audit_passed"] is False
    assert artifact["row_hash_manifest"]["n_duplicate_source_instances"] == 6

    leak_receipt = mod.audit_scorer_texts(
        mod.sanitize_rows_for_scoring(_rows(leak_gold=True))
    )
    oracle_receipt = mod.audit_oracle_provenance(
        _rows(oracle_selection_feature=True),
        mod.sanitize_rows_for_scoring(_rows(oracle_selection_feature=True)),
    )
    assert leak_receipt["passed"] is False
    assert "gold_outside_candidate_answer" in leak_receipt["failures"][0]["reason"]
    assert oracle_receipt["passed"] is False
    assert oracle_receipt["raw_solver_score_used_for_selection_count"] > 0
    assert mod.artifact_schema_errors(artifact) == []


def test_req_verify_5060_blocks_and_validates_schema_edges(tmp_path: Path) -> None:
    """REQ-VERIFY-5060: closed preconditions and malformed artifacts fail closed."""

    rows = _rows()
    _write_5044(tmp_path, rows)
    _write_5059(tmp_path, available=False)

    blocked = mod.run(
        root=tmp_path,
        artifact_path=tmp_path / "blocked.json",
        score_fn=_score_correct,
        write=True,
    )

    assert blocked["honest_verdict"] == "blocked_exp5059_best_arm_unavailable"
    assert blocked["d4_verdict_class"] == "blocked"
    assert blocked["second_corpus_confirmed"] is False
    assert blocked["delta_vs_tuned_sc_second"] is None
    assert json.loads((tmp_path / "blocked.json").read_text(encoding="utf-8")) == blocked
    assert mod.artifact_schema_errors(blocked) == []

    missing_cache = mod.run(
        root=tmp_path / "missing",
        artifact_path=tmp_path / "missing.json",
        score_fn=_score_correct,
        write=True,
    )
    assert missing_cache["honest_verdict"] == "blocked_second_corpus_cache_unavailable"
    assert json.loads((tmp_path / "missing.json").read_text(encoding="utf-8")) == missing_cache
    missing_cache_no_write = mod.run(
        root=tmp_path / "missing_no_write",
        artifact_path=tmp_path / "missing_no_write.json",
        score_fn=_score_correct,
        write=False,
    )
    assert missing_cache_no_write["honest_verdict"] == "blocked_second_corpus_cache_unavailable"

    bad_json = tmp_path / "bad.json"
    bad_json.write_text("{bad", encoding="utf-8")
    assert mod.read_json_object(bad_json) is None
    assert mod.read_jsonl(tmp_path / "missing.jsonl") == []
    mixed_jsonl = tmp_path / "mixed.jsonl"
    mixed_jsonl.write_text("\n[]\n{\"ok\": true}\n", encoding="utf-8")
    assert mod.read_jsonl(mixed_jsonl) == [{"ok": True}]
    assert mod.number("bad") is None
    assert mod.number(True) is None
    assert mod._delta_label(None) == "unknown"
    assert mod._delta_label(-0.125) == "minus_0p125"
    assert mod.render_candidate_text({"question": "Q"}, {"answer": "A"}) == (
        "Candidate answer: A\nQuestion: Q"
    )
    assert mod.sanitize_rows_for_scoring([{"row_id": "x", "candidates": [None]}]) == []
    odd_manifest = mod.build_row_hash_manifest(
        [{"row_id": "odd", "source_row_id": "odd-source", "candidates": [None]}]
    )
    assert odd_manifest["n_rows"] == 1
    odd_oracle = mod.audit_oracle_provenance(
        [{"candidates": [None]}],
        [{"candidates": [None]}],
    )
    assert odd_oracle["passed"] is True
    assert mod.audit_train_test_overlap(rows, {"n_questions": 8})["passed"] is True
    assert mod.audit_train_test_overlap(
        [{**rows[0], "corpus": "MuSR/murder_mysteries"}],
        {"n_questions": 1},
    )["passed"] is False
    assert mod.load_second_corpus(tmp_path / "absent")[3] == "second_corpus_cache_unavailable"
    for dirname, payload, expected in (
        ("oracle", {"verifier_is_oracle": True}, "second_corpus_oracle_tainted"),
        (
            "not_built",
            {"verifier_is_oracle": False, "second_corpus_cache_built": False},
            "second_corpus_cache_not_built",
        ),
        (
            "no_headroom",
            {
                "verifier_is_oracle": False,
                "second_corpus_cache_built": True,
                "headroom_present": False,
            },
            "second_corpus_not_headroom_present",
        ),
        (
            "empty",
            {
                "verifier_is_oracle": False,
                "second_corpus_cache_built": True,
                "headroom_present": True,
                "candidate_cache_path": "empty.jsonl",
            },
            "second_corpus_cache_empty",
        ),
    ):
        subroot = tmp_path / dirname
        _write_json(subroot / mod.EXP5044_RESULT_RELATIVE_PATH, payload)
        assert mod.load_second_corpus(subroot)[3] == expected
    assert mod.load_exp5059_gate(tmp_path / "no5059")[1]["reason"] == "exp5059_artifact_unavailable"
    with pytest.raises(RuntimeError, match="score_fn returned"):
        mod.evaluate_second_corpus(
            mod.sanitize_rows_for_scoring(rows),
            checkpoint="/checkpoints/d1",
            score_fn=lambda _checkpoint, _texts: [],
        )
    blocked_no_write = mod.run(
        root=tmp_path,
        artifact_path=tmp_path / "blocked_nowrite.json",
        score_fn=_score_correct,
        write=False,
    )
    assert blocked_no_write["honest_verdict"] == "blocked_exp5059_best_arm_unavailable"

    for mutated, field in (
        ({key: value for key, value in blocked.items() if key != "honest_verdict"}, "honest_verdict"),
        ({**blocked, "schema": "wrong"}, "schema"),
        ({**blocked, "spec_refs": []}, "spec_refs"),
        ({**blocked, "model_specs": []}, "model_specs"),
        ({**blocked, "second_corpus_confirmed": "no"}, "second_corpus_confirmed"),
        ({**blocked, "second_corpus_audit_clean": "no"}, "second_corpus_audit_clean"),
        ({**blocked, "leak_audit_passed": "yes"}, "leak_audit_passed"),
        ({**blocked, "oracle_provenance_passed": "yes"}, "oracle_provenance_passed"),
        ({**blocked, "duplicate_audit_passed": "yes"}, "duplicate_audit_passed"),
        ({**blocked, "legacy_models_smoke_only": False}, "legacy_models_smoke_only"),
        ({**blocked, "n_questions_second": -1}, "n_questions_second"),
        ({**blocked, "delta_vs_tuned_sc_second": "1"}, "delta_vs_tuned_sc_second"),
        ({**blocked, "paired_ci95_second": [0.0]}, "paired_ci95_second"),
        ({**blocked, "mcnemar_p_second": 1.5}, "mcnemar_p_second"),
        ({**blocked, "row_hash_manifest": []}, "row_hash_manifest"),
        ({**blocked, "d4_verdict_class": "unknown"}, "d4_verdict_class"),
    ):
        assert field in mod.artifact_schema_errors(mutated)
