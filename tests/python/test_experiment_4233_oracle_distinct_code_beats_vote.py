"""Tests for Exp 4233 oracle-distinct code pass-predictor gate.

Spec refs: REQ-VERIFY-4233, SCENARIO-VERIFY-4233,
SCENARIO-VERIFY-4233-NO-HEADROOM, SCENARIO-VERIFY-4233-BLOCKED.
"""

from __future__ import annotations

import inspect
import json
from pathlib import Path

import pytest

from carnot.reporting import oracle_distinct_code_beats_vote_4233 as mod


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _adversarial_clean(_path: Path) -> dict:
    return {
        "returncode": 0,
        "reports": [
            {
                "flags": [],
                "flag_count": 0,
                "max_severity": 0,
            }
        ],
    }


def _pool_spec(rel: str = "results/fixture_code_pool.jsonl") -> mod.PoolSpec:
    return mod.PoolSpec("fixture_code_pool", (Path(rel),))


def _write_code_gate_fixture(
    root: Path,
    *,
    correct_majority: bool,
    task_count: int = 6,
) -> Path:
    rows: list[dict] = []
    for task_index in range(task_count):
        task_id = f"HumanEval/{task_index}"
        function_name = f"solve_{task_index}"
        correct = (
            f"def {function_name}(x):\n"
            "    # PASS_PATTERN stable invariant candidate\n"
            "    return x + 1\n"
        )
        wrong = (
            f"def {function_name}(x):\n"
            "    # WRONG_PATTERN brittle shortcut candidate\n"
            "    return x - 1\n"
        )
        if correct_majority:
            candidates = [(correct, True), (correct, True), (wrong, False)]
        else:
            candidates = [(wrong, False), (wrong, False), (correct, True)]
        for candidate_index, (completion, hidden_pass) in enumerate(candidates):
            rows.append(
                {
                    "task_id": task_id,
                    "completion": completion,
                    "prompt": f"Complete the Python function for {task_id}.",
                    "hidden_pass": hidden_pass,
                    "source_draw_index": candidate_index,
                }
            )
    _write_jsonl(root / "results" / "fixture_code_pool.jsonl", rows)
    return root


def test_req_4233_spec_declares_oracle_distinct_code_gate_contract() -> None:
    """REQ-VERIFY-4233: OpenSpec declares the code beats-vote contract."""

    spec = Path("openspec/capabilities/verification/spec.md").read_text(encoding="utf-8")
    for marker in (
        "REQ-VERIFY-4233",
        "SCENARIO-VERIFY-4233",
        "SCENARIO-VERIFY-4233-NO-HEADROOM",
        "SCENARIO-VERIFY-4233-BLOCKED",
        "python/carnot/reporting/oracle_distinct_code_beats_vote_4233.py",
        "results/experiment_4233_oracle_distinct_code_beats_vote.py",
        "blocked_code_candidate_pool_missing",
        "code_oracle_distinct_beats_vote",
        "code_predictor_minus_vote_delta",
        "code_predictor_minus_vote_ci95",
        "oracle_at_k",
        "held_out_task_n",
        "disambiguation_read",
        "verifier_is_oracle=false",
        "calibrated imbalance-aware",
    ):
        assert marker in spec
    for principle in mod.FIELD_PRINCIPLES.values():
        assert principle in spec


def test_code_predictor_beats_vote_without_execution(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-4233: learned OOF code scores beat vote with headroom."""

    _write_code_gate_fixture(tmp_path, correct_majority=False, task_count=6)

    artifact = mod.run(
        tmp_path,
        pool_specs=(_pool_spec(),),
        adversarial_runner=_adversarial_clean,
    )

    mod.validate_artifact(artifact)
    assert artifact["honest_verdict"] == "complete: code_oracle_distinct_beats_vote"
    assert artifact["code_oracle_distinct_beats_vote"] is True
    assert artifact["disambiguation_read"] == "ARC_null_is_data_sparsity"
    assert artifact["verifier_is_oracle"] is False
    assert artifact["held_out_task_n"] == 6
    assert artifact["oracle_at_k"] == pytest.approx(1.0)
    assert artifact["pass_rates"]["vote_at_1"] == pytest.approx(0.0)
    assert artifact["pass_rates"]["predictor_at_1"] == pytest.approx(1.0)
    assert artifact["code_predictor_minus_vote_delta"] == pytest.approx(1.0)
    assert artifact["code_predictor_minus_vote_ci95"] == [1.0, 1.0]
    assert artifact["off_fold_auroc"] > 0.9
    assert artifact["bootstrap_resamples"] >= 2000
    assert artifact["candidate_pool"]["source_id"] == "fixture_code_pool"
    assert artifact["vote_signature_source"] == "normalized_code_text_signature"
    assert artifact["model_specs"]["verifier_is_oracle"] is False
    assert "hidden_pass" not in artifact["model_specs"]["feature_names"]
    assert "visible_perfect" not in artifact["model_specs"]["feature_names"]
    assert "arm" not in artifact["model_specs"]["feature_names"]
    assert artifact["adversarial_verify"]["circular_moat_overclaim_clean"] is True
    assert (tmp_path / "results" / "experiment_4233_oracle_distinct_code_beats_vote.json").exists()

    feature_source = inspect.getsource(mod.build_feature_matrix)
    for forbidden in ("safe_exec", "pytest", "HumanEval check", "EvalPlus"):
        assert forbidden not in feature_source


def test_code_no_headroom_positive_control_is_uninformative(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-4233-NO-HEADROOM: oracle@K ~= vote blocks a false null."""

    _write_code_gate_fixture(tmp_path, correct_majority=True, task_count=5)

    artifact = mod.run(
        tmp_path,
        pool_specs=(_pool_spec(),),
        adversarial_runner=_adversarial_clean,
    )

    mod.validate_artifact(artifact)
    assert artifact["honest_verdict"] == "complete: code_no_headroom"
    assert artifact["code_oracle_distinct_beats_vote"] is False
    assert artifact["disambiguation_read"] == "code_no_headroom"
    assert artifact["oracle_at_k"] == artifact["pass_rates"]["vote_at_1"] == pytest.approx(1.0)
    assert artifact["headroom_exists"] is False
    assert "selection_thesis_bounded" not in artifact["honest_verdict"]


def test_missing_code_candidate_pool_blocks_honestly(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-4233-BLOCKED: missing pass-labeled pool stops training."""

    _write_jsonl(
        tmp_path / "results" / "no_labels.jsonl",
        [
            {"task_id": "HumanEval/0", "completion": "def f():\n    return 1\n"},
            {"task_id": "HumanEval/0", "completion": "def f():\n    return 2\n"},
        ],
    )

    artifact = mod.run(
        tmp_path,
        pool_specs=(mod.PoolSpec("no_labels", (Path("results/no_labels.jsonl"),)),),
        adversarial_runner=_adversarial_clean,
    )

    mod.validate_artifact(artifact)
    assert artifact["honest_verdict"] == "blocked_code_candidate_pool_missing"
    assert artifact["code_oracle_distinct_beats_vote"] is False
    assert artifact["code_predictor_minus_vote_delta"] == 0.0
    assert artifact["code_predictor_minus_vote_ci95"] == [0.0, 0.0]
    assert artifact["oracle_at_k"] == 0.0
    assert artifact["held_out_task_n"] == 0
    assert artifact["disambiguation_read"] == "blocked_code_candidate_pool_missing"
    assert artifact["verifier_is_oracle"] is False
    assert artifact["acceptance_gate"] is True
    assert artifact["candidate_pool"]["source_id"] == ""
    assert artifact["attempted_candidate_sources"][0]["source_id"] == "no_labels"
