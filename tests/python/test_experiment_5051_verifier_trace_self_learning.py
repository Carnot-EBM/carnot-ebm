"""Tests for Exp 5051 verifier-trace self-learning.

Spec refs: REQ-VERIFY-5051, SCENARIO-VERIFY-5051.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from carnot import experiment_5051_verifier_trace_self_learning as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "verification" / "spec.md"


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def _write_checkpoint_rows(root: Path) -> None:
    rows = [
        ("Ana", "Bea", "Ana", "Bea", False, False),
        ("Cal", "Cal", "Cal", "Cal", False, False),
        ("Dia", "Eli", "Dia", "Eli", False, True),
        ("Fay", "Gus", "Gus", "Fay", True, False),
        ("Hal", "Ira", "Hal", "Ira", False, False),
        ("Jen", "Jen", "Ken", "Ken", False, False),
    ]
    ckdir = root / mod.MUSR_CHECKPOINT_RELATIVE_DIR
    for index, (gold, sc_answer, energy_answer, judge_answer, abstained, _unused) in enumerate(rows):
        answers = [sc_answer, energy_answer, gold, judge_answer]
        _write_json(
            ckdir / f"q{index:04d}.json",
            {
                "q": index,
                "gold": gold,
                "sc_answer": sc_answer,
                "energy_pure_answer": energy_answer,
                "energy_answer": energy_answer,
                "judge_answer": judge_answer,
                "energy_abstained": abstained,
                "answers": answers,
            },
        )


def _artifact(verifier_key: str) -> dict[str, Any]:
    verifier_correct = [0, 1, 0, 1, 0, 0]
    tuned_correct = [1, 1, 0, 0, 1, 0]
    oracle_correct = [1, 1, 1, 1, 1, 0]
    verifier_predictions = ["Ana", "Cal", "Dia", "Gus", "Hal", "Ken"]
    tuned_predictions = ["Bea", "Cal", "Eli", "Fay", "Ira", "Jen"]
    paired = {
        verifier_key: verifier_correct,
        "tuned_self_consistency": tuned_correct,
        "oracle_at_k": oracle_correct,
    }
    evaluation: dict[str, Any] = {
        "n_rows": len(verifier_correct),
        "paired_correct": paired,
        "tuned_self_consistency": {
            "accuracy": sum(tuned_correct) / len(tuned_correct),
            "predictions": tuned_predictions,
        },
        "oracle_at_k": sum(oracle_correct) / len(oracle_correct),
    }
    if verifier_key == "verifier":
        evaluation["verifier"] = {
            "accuracy": sum(verifier_correct) / len(verifier_correct),
            "predictions": verifier_predictions,
        }
    else:
        evaluation["predictions"] = {
            verifier_key: verifier_predictions,
            "tuned_self_consistency": tuned_predictions,
        }
    return {
        "honest_verdict": "complete_fixture",
        "evaluation": evaluation,
        "model_specs": {"fixture": True},
        "verifier_is_oracle": False,
    }


def _write_source_artifacts(root: Path) -> None:
    _write_json(root / mod.EXP5031_RELATIVE_PATH, _artifact("verifier"))
    _write_json(root / mod.EXP5033_RELATIVE_PATH, _artifact("ebrm"))
    _write_json(root / mod.EXP5045_RELATIVE_PATH, _artifact("verifier"))


def _fixture_root(tmp_path: Path) -> Path:
    _write_checkpoint_rows(tmp_path)
    _write_source_artifacts(tmp_path)
    return tmp_path


def _model_resolver(hf_id: str, _preferred_quant: str) -> str | None:
    if hf_id == mod.MODEL_SPECS["verifier_or_judge_model"]:
        return "/models/gemma-4-31B-it-Q4_K_M.gguf"
    return None


def test_req_verify_5051_spec_declares_trace_self_learning_contract() -> None:
    """REQ-VERIFY-5051: OpenSpec anchors the verifier-trace learning artifact."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    for marker in (
        "REQ-VERIFY-5051",
        "SCENARIO-VERIFY-5051",
        "experiment_5051_verifier_trace_self_learning.py",
        "results/experiment_5051_verifier_trace_self_learning.json",
        "self_learning_loop_executed",
        "near_miss_count",
        "verified_trace_count",
        "contamination_guard_passed",
        "fr11_evidence",
        "unsloth/Qwen3.6-35B-A3B-GGUF",
        "unsloth/gemma-4-31B-it-GGUF",
        "unsloth/gemma-4-26B-A4B-it-GGUF",
    ):
        assert marker in spec
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert field in spec
        assert principle["principle"] in spec


def test_scenario_verify_5051_split_integrity_keeps_trace_selection_train_only(
    tmp_path: Path,
) -> None:
    """SCENARIO-VERIFY-5051: near-miss trace selection excludes held-out IDs."""

    root = _fixture_root(tmp_path)
    checkpoint_rows = mod.load_checkpoint_rows(root / mod.MUSR_CHECKPOINT_RELATIVE_DIR)
    evidence = mod.load_source_evidence(root)
    split = mod.build_split_ids(sorted(checkpoint_rows), heldout_count=2)
    near_misses = mod.build_near_miss_dataset(evidence, checkpoint_rows, split)

    assert split["train_ids"] == ["q0000", "q0001", "q0002", "q0003"]
    assert split["heldout_ids"] == ["q0004", "q0005"]
    assert set(split["train_ids"]).isdisjoint(split["heldout_ids"])
    assert near_misses
    assert {row["row_id"] for row in near_misses} <= set(split["train_ids"])
    assert "q0004" not in {row["row_id"] for row in near_misses}


def test_scenario_verify_5051_contamination_guard_rejects_heldout_trace_ids() -> None:
    """SCENARIO-VERIFY-5051: held-out IDs cannot appear in traces or memory."""

    trace = {
        "row_id": "q0001",
        "trace_text": "OBSERVED_SIGNAL: q0001\nREVISION: keep candidates\nVERIFICATION: candidate_set_preserved\nMEMORY_UPDATE: insert rule",
    }
    memory = {"support_row_ids": ["q0001"], "rules": []}

    clean = mod.contamination_guard(
        train_ids=["q0000", "q0001"],
        heldout_ids=["q0002"],
        trace_inputs=[{"row_id": "q0001"}],
        verified_traces=[trace],
        memory=memory,
    )
    leaked = mod.contamination_guard(
        train_ids=["q0000", "q0001", "q0002"],
        heldout_ids=["q0002"],
        trace_inputs=[{"row_id": "q0002"}],
        verified_traces=[{**trace, "row_id": "q0002"}],
        memory={"support_row_ids": ["q0002"], "rules": []},
    )

    assert clean["passed"] is True
    assert clean["violations"] == []
    assert leaked["passed"] is False
    assert "split_overlap:q0002" in leaked["violations"]
    assert any("q0002" in item for item in leaked["violations"])


def test_scenario_verify_5051_structural_trace_filter_rejects_gold_leakage() -> None:
    """SCENARIO-VERIFY-5051: verifier integrity is structural, not answer leakage."""

    near_miss = {
        "row_id": "q0000",
        "source_experiment": 5045,
        "verifier_prediction": "Ana",
        "tuned_sc_prediction": "Bea",
        "energy_abstained": False,
        "verifier_sc_disagreement": True,
        "candidate_count": 4,
        "near_miss_reasons": ["verifier_wrong_oracle_recoverable"],
    }
    model = {
        "role": "verifier_or_judge_model",
        "hf_id": mod.MODEL_SPECS["verifier_or_judge_model"],
        "resolved_path": "/models/gemma.gguf",
    }

    good = mod.generate_revision_trace(near_miss, model)
    bad = {**good, "trace_text": good["trace_text"] + "\nGold answer is Bea."}
    malformed = {
        **good,
        "row_id": "q0004",
        "candidate_set_preserved": False,
        "features": {},
        "trace_text": "OBSERVED_SIGNAL: q0004",
    }

    assert mod.verify_trace_integrity(good, heldout_ids=["q0004"])["passed"] is True
    failed = mod.verify_trace_integrity(bad, heldout_ids=["q0004"])
    assert failed["passed"] is False
    assert "final_answer_leak" in failed["failed_checks"]
    malformed_failed = mod.verify_trace_integrity(malformed, heldout_ids=["q0004"])
    assert malformed_failed["passed"] is False
    assert "required_sections" in malformed_failed["failed_checks"]
    assert "candidate_set_preserved" in malformed_failed["failed_checks"]
    assert "structural_signal" in malformed_failed["failed_checks"]
    assert "heldout_id_leak:q0004" in malformed_failed["failed_checks"]
    verified, diagnostics = mod.filter_verified_traces([good, bad], heldout_ids=["q0004"])
    assert verified == [good | {"integrity_check": mod.verify_trace_integrity(good, heldout_ids=["q0004"])}]
    assert diagnostics["rejected_trace_count"] == 1


def test_scenario_verify_5051_helper_edges_and_schema_errors(tmp_path: Path) -> None:
    """REQ-VERIFY-5051: defensive edges stay explicit and schema checked."""

    assert mod._to_int_list("not-a-list") == []
    assert mod._to_prediction_list(None, 2) == [None, None]
    assert mod.build_split_ids([]) == {"train_ids": [], "heldout_ids": []}

    root = _fixture_root(tmp_path)
    checkpoint_rows = mod.load_checkpoint_rows(root / mod.MUSR_CHECKPOINT_RELATIVE_DIR)
    source = mod.load_source_evidence(root)[-1]
    no_match = mod.evaluate_heldout(
        source,
        checkpoint_rows,
        {"heldout_ids": ["q0001", "q9999"]},
        {"rules": [None]},
    )
    assert no_match["heldout_n"] == 1
    assert no_match["selector_decisions"][0]["selector"] == "pre_update_verifier"

    errors = mod.artifact_schema_errors(
        {
            "contamination_guard_passed": "yes",
            "self_learning_loop_executed": True,
            "near_miss_count": 0,
            "verified_trace_count": 0,
            "update_type": "wrong",
            "checkpoint_or_memory_path": "",
        }
    )
    assert "contamination_guard_passed_bool" in errors
    assert "near_miss_count_positive" in errors
    assert "verified_trace_count_positive" in errors
    assert "update_type" in errors
    assert "checkpoint_or_memory_path" in errors


def test_scenario_verify_5051_blocked_paths_write_honest_artifacts(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-5051: missing rows, missing SOTA, and no traces block."""

    missing_rows = mod.run(
        root=tmp_path,
        artifact_path=tmp_path / "missing.json",
        model_resolver=_model_resolver,
        write=True,
    )
    assert missing_rows["honest_verdict"] == "blocked_cached_musr_rows_missing"
    assert (tmp_path / "missing.json").is_file()

    root = _fixture_root(tmp_path / "no_model")
    no_model = mod.run(
        root=root,
        artifact_path=root / "no_model.json",
        model_resolver=lambda _hf_id, _quant: None,
        heldout_count=2,
        write=True,
    )
    assert no_model["honest_verdict"] == "blocked_no_mandated_local_sota_gguf"
    assert (root / "no_model.json").is_file()

    no_trace_root = tmp_path / "no_trace"
    ckdir = no_trace_root / mod.MUSR_CHECKPOINT_RELATIVE_DIR
    for index in range(3):
        _write_json(
            ckdir / f"q{index:04d}.json",
            {
                "q": index,
                "gold": "A",
                "sc_answer": "A",
                "energy_pure_answer": "A",
                "energy_abstained": False,
                "answers": ["A", "B"],
            },
        )
    clean = _artifact("verifier")
    clean["evaluation"]["paired_correct"]["verifier"] = [1, 1, 1]
    clean["evaluation"]["paired_correct"]["tuned_self_consistency"] = [1, 1, 1]
    clean["evaluation"]["paired_correct"]["oracle_at_k"] = [1, 1, 1]
    clean["evaluation"]["verifier"]["predictions"] = ["A", "A", "A"]
    clean["evaluation"]["tuned_self_consistency"]["predictions"] = ["A", "A", "A"]
    clean_ebrm = _artifact("ebrm")
    clean_ebrm["evaluation"]["paired_correct"]["ebrm"] = [1, 1, 1]
    clean_ebrm["evaluation"]["paired_correct"]["tuned_self_consistency"] = [1, 1, 1]
    clean_ebrm["evaluation"]["paired_correct"]["oracle_at_k"] = [1, 1, 1]
    clean_ebrm["evaluation"]["predictions"]["ebrm"] = ["A", "A", "A"]
    clean_ebrm["evaluation"]["predictions"]["tuned_self_consistency"] = ["A", "A", "A"]
    _write_json(no_trace_root / mod.EXP5031_RELATIVE_PATH, clean)
    _write_json(no_trace_root / mod.EXP5033_RELATIVE_PATH, clean_ebrm)
    _write_json(no_trace_root / mod.EXP5045_RELATIVE_PATH, clean)
    no_trace = mod.run(
        root=no_trace_root,
        artifact_path=no_trace_root / "no_trace.json",
        model_resolver=_model_resolver,
        heldout_count=1,
        write=True,
    )
    assert no_trace["honest_verdict"] == "blocked_verifier_trace_self_learning_no_verified_traces"
    assert no_trace["near_miss_count"] == 0


def test_scenario_verify_5051_run_writes_replay_memory_and_heldout_delta(
    tmp_path: Path,
) -> None:
    """SCENARIO-VERIFY-5051: verified traces create replay memory and held-out eval."""

    root = _fixture_root(tmp_path)
    artifact_path = tmp_path / mod.RESULT_RELATIVE_PATH

    artifact = mod.run(
        root=root,
        artifact_path=artifact_path,
        model_resolver=_model_resolver,
        heldout_count=2,
        now=lambda: 100.0,
        write=True,
    )

    assert artifact["self_learning_loop_executed"] is True
    assert artifact["near_miss_count"] > 0
    assert artifact["verified_trace_count"] == artifact["near_miss_count"]
    assert artifact["update_type"] == "replay_memory_insertion"
    assert artifact["pre_update_accuracy"] == 0.0
    assert artifact["post_update_accuracy"] == 0.5
    assert artifact["heldout_delta"] == 0.5
    assert artifact["genuine_tuned_sc_accuracy"] == 0.5
    assert artifact["contamination_guard_passed"] is True
    assert artifact["fr11_evidence"]["heldout_labels_used_only_for_evaluation"] is True
    assert artifact["model_specs"]["trace_generation_model"]["hf_id"] == mod.MODEL_SPECS[
        "verifier_or_judge_model"
    ]
    assert set(artifact["split_ids"]["train_ids"]).isdisjoint(artifact["split_ids"]["heldout_ids"])
    memory_path = Path(artifact["checkpoint_or_memory_path"])
    assert memory_path.is_file()
    assert json.loads(memory_path.read_text(encoding="utf-8"))["support_row_ids"]
    assert json.loads(artifact_path.read_text(encoding="utf-8")) == artifact
    assert mod.artifact_schema_errors(artifact) == []
