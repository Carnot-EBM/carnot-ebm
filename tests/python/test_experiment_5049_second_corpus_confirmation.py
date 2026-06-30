"""Tests for Exp 5049 second-corpus powered-verifier confirmation.

Spec refs: REQ-VERIFY-5049, SCENARIO-VERIFY-5049.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_5049_second_corpus_confirmation as mod


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


def _second_rows() -> list[dict[str, Any]]:
    rows = []
    for index in range(4):
        rows.append(
            {
                "row_id": f"cb::{index}",
                "corpus": "ConstraintBench-exact-v1",
                "question": f"Solve fixture {index}",
                "context": "Use the public constraints to choose the feasible optimum.",
                "gold": "A",
                "label": "ORACLE_SECRET",
                "candidates": [
                    {
                        "candidate_id": f"cb::{index}/wrong",
                        "answer": "B",
                        "cache_index": 0,
                        "temperature": "deterministic",
                        "label_correct": False,
                        "candidate_label": "incorrect ORACLE_SECRET",
                        "solver_verdict": {"reasons": ["ORACLE_SECRET"]},
                        "generation_model": "oracle-model-name",
                    },
                    {
                        "candidate_id": f"cb::{index}/right",
                        "answer": "A",
                        "cache_index": 1,
                        "temperature": "deterministic",
                        "label_correct": True,
                        "candidate_label": "correct ORACLE_SECRET",
                        "solver_verdict": {"reasons": ["ORACLE_SECRET"]},
                        "generation_model": "oracle-model-name",
                    },
                    {
                        "candidate_id": f"cb::{index}/wrong2",
                        "answer": "B",
                        "cache_index": 2,
                        "temperature": "deterministic",
                    },
                ],
            }
        )
    return rows


def _write_5044(root: Path, rows: list[dict[str, Any]] | None = None) -> Path:
    cache_path = root / mod.EXP5044_CACHE_RELATIVE_PATH
    _write_jsonl(cache_path, rows or _second_rows())
    _write_json(
        root / mod.EXP5044_RELATIVE_PATH,
        {
            "second_corpus_cache_built": True,
            "second_corpus_name": "ConstraintBench-exact-v1",
            "candidate_cache_path": cache_path.as_posix(),
            "n_questions": len(rows or _second_rows()),
            "headroom_present": True,
            "verifier_is_oracle": False,
            "genuine_sc_accuracy": 0.0,
            "oracle_at_k": 1.0,
            "model_specs": {"candidate_generation": "deterministic_solver_backed_constraint_variants"},
        },
    )
    return cache_path


def _write_5045(root: Path, *, delta: float = 0.08, oracle: bool = False) -> None:
    _write_json(
        root / mod.EXP5045_RELATIVE_PATH,
        {
            "verifier_is_oracle": oracle,
            "headroom_present": True,
            "powered_scorer_available": True,
            "scorer_trained": True,
            "powered_lora_ebm_accuracy": 0.665,
            "genuine_tuned_sc_accuracy": 0.665 - delta,
            "delta_vs_tuned_sc": delta,
            "checkpoint_path": "/tmp/checkpoint",
            "model_specs": {"lora_ebm": {"base_model": "Qwen/Qwen3.5-2B"}},
        },
    )


def _write_5046(root: Path, *, delta: float = -0.03, oracle: bool = False) -> None:
    _write_json(
        root / mod.EXP5046_RELATIVE_PATH,
        {
            "verifier_is_oracle": oracle,
            "headroom_present": True,
            "process_reward_available": True,
            "scalar_marker_only": False,
            "oracle_distinctness_enforced": True,
            "process_reward_accuracy": 0.555,
            "genuine_tuned_sc_accuracy": 0.555 - delta,
            "delta_vs_tuned_sc": delta,
            "model_specs": {"process_trace_model": {"hf_id": "unsloth/Qwen3.6-35B-A3B-GGUF"}},
        },
    )


def _write_5047(root: Path, *, delta: float = 0.02, degenerate: bool = False) -> None:
    _write_json(
        root / mod.EXP5047_RELATIVE_PATH,
        {
            "verifier_is_oracle": False,
            "headroom_present": True,
            "calibration_available": True,
            "degeneracy_guard_fired": degenerate,
            "calibrated_accuracy": 0.605,
            "genuine_tuned_sc_accuracy": 0.605 - delta,
            "delta_vs_tuned_sc": delta,
            "model_specs": {"calibration_readout": {"kind": "additive_fuzzy_kan_purm"}},
        },
    )


def test_req_verify_5049_spec_declares_confirmation_contract() -> None:
    """REQ-VERIFY-5049: OpenSpec anchors the Exp 5049 artifact contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    for marker in (
        "REQ-VERIFY-5049",
        "SCENARIO-VERIFY-5049",
        "experiment_5049_second_corpus_confirmation.py",
        "results/experiment_5049_second_corpus_confirmation.json",
        "second_corpus_confirmed",
        "best_arm_source",
        "MuSR margin did not transfer",
    ):
        assert marker in spec


def test_req_verify_5049_selects_best_non_degenerate_powered_arm(tmp_path: Path) -> None:
    """REQ-VERIFY-5049: oracle and degenerate upstream rows are rejected."""

    _write_5045(tmp_path, delta=0.08)
    _write_5046(tmp_path, delta=0.30, oracle=True)
    _write_5047(tmp_path, delta=0.50, degenerate=True)

    best, checks = mod.select_best_powered_arm(tmp_path)

    assert best is not None
    assert best.arm == "D1"
    assert best.source == mod.EXP5045_RELATIVE_PATH
    assert best.delta_vs_tuned_sc == pytest.approx(0.08)
    assert {check["arm"]: check["available"] for check in checks} == {
        "D1": True,
        "D2": False,
        "D3": False,
    }


def test_scenario_verify_5049_run_scores_without_oracle_fields(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-5049: candidate scoring never receives oracle metadata."""

    _write_5044(tmp_path)
    _write_5045(tmp_path, delta=0.08)
    _write_5046(tmp_path, delta=-0.03)
    _write_5047(tmp_path, delta=0.02)
    seen_texts: list[str] = []

    def score_fn(_checkpoint: str, texts: list[str]) -> list[float]:
        seen_texts.extend(texts)
        assert all("ORACLE_SECRET" not in text for text in texts)
        return [0.0 if "Candidate answer: A" in text else 1.0 for text in texts]

    artifact_path = tmp_path / mod.RESULT_RELATIVE_PATH
    artifact = mod.run(
        root=tmp_path,
        artifact_path=artifact_path,
        score_fn=score_fn,
        bootstrap_samples=64,
        now=lambda: 100.0,
        write=True,
    )

    assert seen_texts
    assert artifact["honest_verdict"].startswith(
        "success_second_corpus_confirms_musr_margin_constraintbench_exact_v1_"
    )
    assert artifact["second_corpus_confirmed"] is True
    assert artifact["best_arm"] == "D1"
    assert artifact["best_arm_source"] == mod.EXP5045_RELATIVE_PATH
    assert artifact["n_questions_second"] == 4
    assert artifact["genuine_sc_accuracy_second"] == pytest.approx(0.0)
    assert artifact["verifier_accuracy_second"] == pytest.approx(1.0)
    assert artifact["delta_vs_tuned_sc_second"] == pytest.approx(1.0)
    assert artifact["paired_ci95_second"] == [1.0, 1.0]
    assert artifact["mcnemar_p_second"] == pytest.approx(0.125)
    assert artifact["headroom_present"] is True
    assert artifact["verifier_is_oracle"] is False
    assert artifact["oracle_distinctness_enforced"] is True
    assert artifact["model_specs"]["candidate_rows"] == {
        "candidate_generation": "deterministic_solver_backed_constraint_variants"
    }
    assert mod.artifact_schema_errors(artifact) == []
    assert json.loads(artifact_path.read_text(encoding="utf-8")) == artifact


def test_scenario_verify_5049_complete_null_when_margin_does_not_transfer(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-5049: a second-corpus miss is reported as an honest null."""

    _write_5044(tmp_path)
    _write_5045(tmp_path, delta=0.08)

    def score_fn(_checkpoint: str, texts: list[str]) -> list[float]:
        return [0.0 if "Candidate answer: B" in text else 1.0 for text in texts]

    artifact = mod.run(
        root=tmp_path,
        artifact_path=tmp_path / mod.RESULT_RELATIVE_PATH,
        score_fn=score_fn,
        bootstrap_samples=64,
        write=False,
    )

    assert artifact["honest_verdict"].startswith(
        "complete_second_corpus_musr_margin_did_not_transfer_constraintbench_exact_v1_"
    )
    assert artifact["second_corpus_confirmed"] is False
    assert artifact["verifier_accuracy_second"] == pytest.approx(0.0)
    assert artifact["delta_vs_tuned_sc_second"] == pytest.approx(0.0)
    assert mod.artifact_schema_errors(artifact) == []


def test_req_verify_5049_blocked_and_schema_edges(tmp_path: Path) -> None:
    """REQ-VERIFY-5049: closed preconditions and malformed artifacts fail closed."""

    blank_jsonl = tmp_path / "blank.jsonl"
    blank_jsonl.write_text("\n{}\n", encoding="utf-8")
    assert mod._read_jsonl(blank_jsonl) == [{}]
    assert mod._number(True) is None
    assert mod._number("bad") is None
    assert mod._payload_common_ok({"verifier_is_oracle": False, "headroom_present": False}) == (
        False,
        "musr_headroom_not_present",
    )
    rejected_d1, rejected_check = mod._d1_from_payload(
        tmp_path / "d1.json",
        {
            "verifier_is_oracle": False,
            "headroom_present": True,
            "powered_scorer_available": True,
            "scorer_trained": True,
            "powered_lora_ebm_accuracy": 0.5,
            "delta_vs_tuned_sc": 0.1,
            "checkpoint_path": "",
        },
    )
    assert rejected_d1 is None
    assert rejected_check["detail"] == "powered_lora_ebm_not_deployable"

    assert mod.load_exp5044_rows(tmp_path / "missing_root")[3] == "second_corpus_cache_unavailable"
    _write_json(
        tmp_path / "oracle" / mod.EXP5044_RELATIVE_PATH,
        {"verifier_is_oracle": True},
    )
    assert mod.load_exp5044_rows(tmp_path / "oracle")[3] == "second_corpus_oracle_tainted"
    _write_json(
        tmp_path / "not_built" / mod.EXP5044_RELATIVE_PATH,
        {"verifier_is_oracle": False, "headroom_present": True, "second_corpus_cache_built": False},
    )
    assert mod.load_exp5044_rows(tmp_path / "not_built")[3] == "second_corpus_cache_not_built"
    _write_json(
        tmp_path / "empty" / mod.EXP5044_RELATIVE_PATH,
        {
            "verifier_is_oracle": False,
            "headroom_present": True,
            "second_corpus_cache_built": True,
            "candidate_cache_path": "empty.jsonl",
        },
    )
    assert mod.load_exp5044_rows(tmp_path / "empty")[3] == "second_corpus_cache_empty"

    _write_5044(tmp_path)
    missing_cache = mod.run(
        root=tmp_path / "missing_cache",
        artifact_path=tmp_path / "missing_cache.json",
        score_fn=lambda _checkpoint, texts: [0.0 for _text in texts],
        write=True,
    )
    assert missing_cache["honest_verdict"] == "blocked_second_corpus_cache_unavailable"
    assert json.loads((tmp_path / "missing_cache.json").read_text(encoding="utf-8")) == missing_cache

    no_arm = mod.run(
        root=tmp_path,
        artifact_path=tmp_path / "no_arm.json",
        score_fn=lambda _checkpoint, texts: [0.0 for _text in texts],
        write=True,
    )
    assert no_arm["honest_verdict"] == "blocked_no_non_degenerate_powered_verifier"
    assert no_arm["best_arm"] is None
    assert json.loads((tmp_path / "no_arm.json").read_text(encoding="utf-8")) == no_arm
    assert mod.artifact_schema_errors(no_arm) == []

    _write_5045(tmp_path, delta=0.08)
    bad_5044 = json.loads((tmp_path / mod.EXP5044_RELATIVE_PATH).read_text(encoding="utf-8"))
    bad_5044["headroom_present"] = False
    _write_json(tmp_path / mod.EXP5044_RELATIVE_PATH, bad_5044)
    no_cache = mod.run(
        root=tmp_path,
        artifact_path=tmp_path / "no_cache.json",
        score_fn=lambda _checkpoint, texts: [0.0 for _text in texts],
        write=False,
    )
    assert no_cache["honest_verdict"] == "blocked_second_corpus_not_headroom_present"

    _write_5044(tmp_path)
    rows = _second_rows()
    stripped = mod.sanitize_second_corpus_rows(rows)
    assert "label_correct" not in stripped[0]["candidates"][0]
    assert "solver_verdict" not in stripped[0]["candidates"][0]
    assert mod.oracle_distinctness_self_check(stripped) is True
    best, _checks = mod.select_best_powered_arm(tmp_path)
    assert best is not None
    with pytest.raises(RuntimeError, match="score_fn returned"):
        mod._d1_energy_by_id(best, stripped, lambda _checkpoint, _texts: [])
    with pytest.raises(RuntimeError, match="no deployable"):
        mod.evaluate_second_corpus(
            stripped,
            arm=mod.PoweredArm(
                arm="D9",
                source="fixture",
                scorer_kind="unsupported",
                delta_vs_tuned_sc=0.0,
                selection_accuracy=0.0,
                artifact_path=tmp_path / "d9.json",
                model_specs={},
            ),
            score_fn=lambda _checkpoint, texts: [0.0 for _text in texts],
        )
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(mod, "oracle_distinctness_self_check", lambda _rows: False)
        guard_blocked = mod.run(
            root=tmp_path,
            artifact_path=tmp_path / "guard_blocked.json",
            score_fn=lambda _checkpoint, texts: [0.0 for _text in texts],
            write=False,
        )
    assert guard_blocked["honest_verdict"] == "blocked_second_corpus_scoring_unavailable"
    assert "shared guard" in guard_blocked["blocked_error"]
    blocked_scoring = mod.run(
        root=tmp_path,
        artifact_path=tmp_path / "blocked_scoring.json",
        score_fn=lambda _checkpoint, _texts: (_ for _ in ()).throw(RuntimeError("score boom")),
        write=True,
    )
    assert blocked_scoring["honest_verdict"] == "blocked_second_corpus_scoring_unavailable"
    assert "score boom" in blocked_scoring["blocked_error"]
    assert mod._read_json_object(tmp_path / "missing.json") is None
    assert mod._read_jsonl(tmp_path / "missing.jsonl") == []
    bad_json = tmp_path / "bad.json"
    bad_json.write_text("{bad", encoding="utf-8")
    assert mod._read_json_object(bad_json) is None

    for mutated, field in (
        ({key: value for key, value in no_arm.items() if key != "duration_s"}, "duration_s"),
        ({**no_arm, "schema": "wrong"}, "schema"),
        ({**no_arm, "experiment": "wrong"}, "experiment"),
        ({**no_arm, "spec_refs": []}, "spec_refs"),
        ({**no_arm, "verifier_is_oracle": True}, "verifier_is_oracle"),
        ({**no_arm, "headroom_present": "yes"}, "headroom_present"),
        ({**no_arm, "second_corpus_confirmed": "no"}, "second_corpus_confirmed"),
        ({**no_arm, "source_artifacts": []}, "source_artifacts"),
        ({**no_arm, "model_specs": []}, "model_specs"),
        ({**no_arm, "n_questions_second": -1}, "n_questions_second"),
        ({**no_arm, "verifier_accuracy_second": 2.0}, "verifier_accuracy_second"),
        ({**no_arm, "delta_vs_tuned_sc_second": "0.1"}, "delta_vs_tuned_sc_second"),
        ({**no_arm, "paired_ci95_second": [0.0]}, "paired_ci95_second"),
        ({**no_arm, "mcnemar_p_second": 1.5}, "mcnemar_p_second"),
        ({**no_arm, "honest_verdict": "maybe"}, "honest_verdict"),
    ):
        assert field in mod.artifact_schema_errors(mutated)
