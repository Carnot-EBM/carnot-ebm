"""Tests for Exp 4246 second-corpus code oracle-distinct replication.

Spec refs: REQ-VERIFY-4246, SCENARIO-VERIFY-4246,
SCENARIO-VERIFY-4246-BLOCKED.
"""

from __future__ import annotations

import inspect
import json
from pathlib import Path

import pytest

from carnot.reporting import code_oracle_distinct_replication_4246 as mod


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


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


def _write_exp4233_source(root: Path, rows: list[dict] | None = None) -> Path:
    source_rel = Path("results/exp4233_source_fixture.jsonl")
    source_path = root / source_rel
    _write_jsonl(
        source_path,
        rows
        or [
            {
                "task_id": "HumanEval/source",
                "completion": "def source_fixture(x):\n    return x\n",
                "hidden_pass": True,
                "source_draw_index": 0,
            }
        ],
    )
    _write_json(
        root / "results" / "experiment_4233_oracle_distinct_code_beats_vote.json",
        {
            "honest_verdict": "complete: code_oracle_distinct_beats_vote",
            "candidate_pool": {
                "source_id": "exp4233_fixture",
                "source_paths": [str(source_rel)],
            },
        },
    )
    return source_path


def _write_code_gate_fixture(
    root: Path,
    rel: str,
    *,
    correct_majority: bool,
    task_count: int = 6,
) -> Path:
    rows: list[dict] = []
    for task_index in range(task_count):
        task_id = f"SecondCorpus/{task_index}"
        function_name = f"solve_second_{task_index}"
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
        candidates = (
            [(correct, True), (correct, True), (wrong, False)]
            if correct_majority
            else [(wrong, False), (wrong, False), (correct, True)]
        )
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
    out = root / rel
    _write_jsonl(out, rows)
    return out


def test_req_4246_spec_declares_second_corpus_replication_contract() -> None:
    """REQ-VERIFY-4246: OpenSpec declares the second-corpus replication gate."""

    spec = Path("openspec/capabilities/verification/spec.md").read_text(encoding="utf-8")
    for marker in (
        "REQ-VERIFY-4246",
        "SCENARIO-VERIFY-4246",
        "SCENARIO-VERIFY-4246-BLOCKED",
        "python/carnot/reporting/code_oracle_distinct_replication_4246.py",
        "results/experiment_4246_code_oracle_distinct_replication.py",
        "blocked_code_second_corpus_missing",
        "code_replication_beats_vote",
        "code_predictor_minus_vote_delta",
        "code_predictor_minus_vote_ci95",
        "oracle_at_k",
        "held_out_task_n",
        "replication_read",
        "verifier_is_oracle=false",
        "calibrated imbalance-aware",
    ):
        assert marker in spec
    for principle in mod.FIELD_PRINCIPLES.values():
        assert principle in spec


def test_distinct_second_corpus_replicates_without_execution(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-4246: distinct corpus can replicate the learned win."""

    _write_exp4233_source(tmp_path)
    _write_code_gate_fixture(
        tmp_path,
        "results/second_corpus_fixture.jsonl",
        correct_majority=False,
        task_count=6,
    )

    artifact = mod.run(
        tmp_path,
        pool_specs=(
            mod.PoolSpec("second_corpus_fixture", (Path("results/second_corpus_fixture.jsonl"),)),
        ),
        adversarial_runner=_adversarial_clean,
    )

    mod.validate_artifact(artifact)
    assert artifact["honest_verdict"] == "complete: code_oracle_distinct_replication_replicates"
    assert artifact["code_replication_beats_vote"] is True
    assert artifact["replication_read"] == "replicates"
    assert artifact["verifier_is_oracle"] is False
    assert artifact["held_out_task_n"] == 6
    assert artifact["oracle_at_k"] == pytest.approx(1.0)
    assert artifact["pass_rates"]["vote_at_1"] == pytest.approx(0.0)
    assert artifact["pass_rates"]["predictor_at_1"] == pytest.approx(1.0)
    assert artifact["code_predictor_minus_vote_delta"] == pytest.approx(1.0)
    assert artifact["code_predictor_minus_vote_ci95"] == [1.0, 1.0]
    assert artifact["off_fold_auroc"] > 0.9
    assert artifact["bootstrap_resamples"] >= 2000
    assert artifact["candidate_pool"]["source_id"] == "second_corpus_fixture"
    assert artifact["model_specs"]["second_corpus_id"] == "second_corpus_fixture"
    assert artifact["model_specs"]["verifier_is_oracle"] is False
    assert "hidden_pass" not in artifact["model_specs"]["feature_names"]
    assert artifact["adversarial_verify"]["circular_moat_overclaim_clean"] is True
    assert (
        tmp_path / "results" / "experiment_4246_code_oracle_distinct_replication.json"
    ).exists()

    feature_source = inspect.getsource(mod.build_feature_matrix)
    for forbidden in ("safe_exec", "pytest", "HumanEval check", "EvalPlus"):
        assert forbidden not in feature_source


def test_no_headroom_second_corpus_is_not_a_false_null(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-4246: oracle@K ~= vote gives no_headroom read."""

    _write_exp4233_source(tmp_path)
    _write_code_gate_fixture(
        tmp_path,
        "results/second_no_headroom.jsonl",
        correct_majority=True,
        task_count=5,
    )

    artifact = mod.run(
        tmp_path,
        pool_specs=(
            mod.PoolSpec("second_no_headroom", (Path("results/second_no_headroom.jsonl"),)),
        ),
        adversarial_runner=_adversarial_clean,
    )

    mod.validate_artifact(artifact)
    assert artifact["honest_verdict"] == "complete: code_oracle_distinct_replication_no_headroom"
    assert artifact["code_replication_beats_vote"] is False
    assert artifact["replication_read"] == "no_headroom"
    assert artifact["oracle_at_k"] == artifact["pass_rates"]["vote_at_1"] == pytest.approx(1.0)
    assert artifact["headroom_exists"] is False


def test_byte_identical_exp4233_duplicate_blocks_honestly(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-4246-BLOCKED: non-distinct duplicate source is skipped."""

    duplicate_rows = [
        {
            "task_id": "HumanEval/0",
            "completion": "def f(x):\n    return x + 1\n",
            "hidden_pass": True,
            "source_draw_index": 0,
        },
        {
            "task_id": "HumanEval/0",
            "completion": "def f(x):\n    return x - 1\n",
            "hidden_pass": False,
            "source_draw_index": 1,
        },
    ]
    _write_exp4233_source(tmp_path, duplicate_rows)
    _write_jsonl(
        tmp_path / "results" / "no_labels.jsonl",
        [
            {"task_id": "HumanEval/0", "completion": "def f():\n    return 1\n"},
            {"task_id": "HumanEval/0", "completion": "def f():\n    return 2\n"},
        ],
    )
    _write_jsonl(tmp_path / "results" / "duplicate_source.jsonl", duplicate_rows)

    artifact = mod.run(
        tmp_path,
        pool_specs=(
            mod.PoolSpec("no_labels", (Path("results/no_labels.jsonl"),)),
            mod.PoolSpec("duplicate_source", (Path("results/duplicate_source.jsonl"),)),
        ),
        adversarial_runner=_adversarial_clean,
    )

    mod.validate_artifact(artifact)
    assert artifact["honest_verdict"] == "blocked_code_second_corpus_missing"
    assert artifact["code_replication_beats_vote"] is False
    assert artifact["code_predictor_minus_vote_delta"] == 0.0
    assert artifact["code_predictor_minus_vote_ci95"] == [0.0, 0.0]
    assert artifact["oracle_at_k"] == 0.0
    assert artifact["held_out_task_n"] == 0
    assert artifact["replication_read"] == "blocked_code_second_corpus_missing"
    assert artifact["verifier_is_oracle"] is False
    assert artifact["acceptance_gate"] is True
    assert artifact["candidate_pool"]["source_id"] == ""
    assert artifact["attempted_candidate_sources"][0]["source_id"] == "no_labels"
    assert (
        artifact["attempted_candidate_sources"][0]["skip_reason"]
        == "missing_viable_multicandidate_hidden_label_rows"
    )
    assert artifact["attempted_candidate_sources"][1]["source_id"] == "duplicate_source"
    assert artifact["attempted_candidate_sources"][1]["distinct_from_exp4233"] is False


def test_missing_exp4233_source_blocks_before_training(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-4246-BLOCKED: source-distinctness needs Exp 4233 hashes."""

    _write_code_gate_fixture(
        tmp_path,
        "results/second_corpus_fixture.jsonl",
        correct_majority=False,
        task_count=6,
    )

    artifact = mod.run(
        tmp_path,
        pool_specs=(
            mod.PoolSpec("second_corpus_fixture", (Path("results/second_corpus_fixture.jsonl"),)),
        ),
        adversarial_runner=_adversarial_clean,
    )

    mod.validate_artifact(artifact)
    assert artifact["honest_verdict"] == "blocked_code_second_corpus_missing"
    assert artifact["attempted_candidate_sources"] == []
    assert artifact["exp4233_source"]["artifact_exists"] is False


def test_corpus_specific_read_is_available_for_nonwinning_headroom() -> None:
    """REQ-VERIFY-4246: headroom without a significant win is corpus_specific."""

    read, verdict = mod._replication_read(
        {"headroom_exists": True, "code_oracle_distinct_beats_vote": False}
    )

    assert read == "corpus_specific"
    assert verdict == "complete: code_oracle_distinct_replication_corpus_specific"


def test_validate_artifact_rejects_schema_drift() -> None:
    """REQ-VERIFY-4246: artifact schema rejects non-bare or inconsistent fields."""

    valid = {
        "honest_verdict": "blocked_code_second_corpus_missing",
        "code_replication_beats_vote": False,
        "code_predictor_minus_vote_delta": 0.0,
        "code_predictor_minus_vote_ci95": [0.0, 0.0],
        "oracle_at_k": 0.0,
        "held_out_task_n": 0,
        "replication_read": "blocked_code_second_corpus_missing",
        "verifier_is_oracle": False,
        "model_specs": {"verifier_is_oracle": False},
        "random_seed": 4246,
        "reproducibility_checksum": "checksum",
        "field_principles": mod.FIELD_PRINCIPLES,
        "spec_refs": mod.SPEC_REFS,
        "acceptance_gate": True,
    }
    mod.validate_artifact(valid)
    cases = [
        ("missing", lambda d: d.pop("honest_verdict")),
        ("bad_verdict", lambda d: d.__setitem__("honest_verdict", "pending")),
        ("bad_bool", lambda d: d.__setitem__("code_replication_beats_vote", 1)),
        ("bad_float", lambda d: d.__setitem__("code_predictor_minus_vote_delta", True)),
        ("bad_ci", lambda d: d.__setitem__("code_predictor_minus_vote_ci95", [0.0])),
        ("bad_n", lambda d: d.__setitem__("held_out_task_n", 0.0)),
        ("bad_read", lambda d: d.__setitem__("replication_read", "maybe")),
        (
            "win_read_mismatch",
            lambda d: (
                d.__setitem__("code_replication_beats_vote", True),
                d.__setitem__("replication_read", "corpus_specific"),
            ),
        ),
        ("oracle_true", lambda d: d.__setitem__("verifier_is_oracle", True)),
        ("bad_seed", lambda d: d.__setitem__("random_seed", 4246.0)),
        ("missing_specs", lambda d: d.__setitem__("model_specs", None)),
        ("specs_oracle_true", lambda d: d.__setitem__("model_specs", {"verifier_is_oracle": True})),
        ("bad_principles", lambda d: d.__setitem__("field_principles", {})),
        ("bad_refs", lambda d: d.__setitem__("spec_refs", [])),
    ]
    for _name, mutate in cases:
        drifted = dict(valid)
        mutate(drifted)
        with pytest.raises(ValueError):
            mod.validate_artifact(drifted)
