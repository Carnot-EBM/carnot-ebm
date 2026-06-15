"""Tests for Exp 4263 verifier-as-reward out-of-band package prep.

Spec refs: REQ-LEARN-4263, SCENARIO-LEARN-4263-READY,
SCENARIO-LEARN-4263-BLOCKED.
"""

from __future__ import annotations

import json
from pathlib import Path

from carnot import experiment_4263_verifier_as_reward_out_of_band_or_retire as exp4263


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "self-learning" / "spec.md"


def _write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    path.write_text(payload, encoding="utf-8")
    return path


def _rows(arm: str, *, n: int = 3, hidden_pass: bool | None = None) -> list[dict]:
    by_arm_hidden = {"A": True, "B": False, "C": True}
    rows: list[dict] = []
    for index in range(n):
        passes = by_arm_hidden[arm] if hidden_pass is None else hidden_pass
        arm_name = {
            "A": "A_certified",
            "B": "B_random_same_generator",
            "C": "C_hidden_gold",
        }[arm]
        rows.append(
            {
                "arm": arm_name,
                "completion": f"def f_{arm.lower()}_{index}(x):\n    return x + {index}\n",
                "hidden_pass": passes,
                "prompt": f"Complete HumanEval fixture {index}.",
                "source_draw_index": index,
                "task_id": f"HumanEval/{index}",
                "visible_perfect": passes,
            }
        )
    return rows


def _stable_checkpoint(tmp_path: Path, *, b_hidden_pass: bool | None = None) -> Path:
    root = tmp_path / "code_verifier_reward_lora_rft_a83b52882c198954"
    _write_jsonl(root / "corpora" / "arm_A.jsonl", _rows("A"))
    _write_jsonl(root / "corpora" / "arm_B.jsonl", _rows("B", hidden_pass=b_hidden_pass))
    _write_jsonl(root / "corpora" / "arm_C.jsonl", _rows("C", n=2))
    return root


def test_req_learn_4263_spec_declares_out_of_band_contract() -> None:
    """REQ-LEARN-4263: OpenSpec declares the terminal package schema."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-LEARN-4263" in spec
    assert "SCENARIO-LEARN-4263-READY" in spec
    assert "SCENARIO-LEARN-4263-BLOCKED" in spec
    for field in exp4263.REQUIRED_ARTIFACT_FIELDS:
        assert field in spec
        assert field in exp4263.FIELD_PRINCIPLES


def test_req_learn_4263_precomputes_deterministic_reward_weighted_corpus(tmp_path: Path) -> None:
    """REQ-LEARN-4263: A/B/C rows are persisted with deterministic reward weights."""

    stable = _stable_checkpoint(tmp_path)
    bundle = exp4263.load_abc_corpora(stable)
    weighted = exp4263.build_reward_weighted_corpus(bundle, random_seed=exp4263.RANDOM_SEED)
    output = tmp_path / "weighted.jsonl"
    checksum = exp4263.write_jsonl_with_checksum(weighted.rows, output)
    repeated_checksum = exp4263.write_jsonl_with_checksum(weighted.rows, tmp_path / "weighted-again.jsonl")

    assert bundle.ready is True
    assert bundle.corpus_sizes == {"A": 3, "B": 3, "C": 2}
    assert weighted.supports_clean_avsb is True
    assert len(weighted.rows) == 8
    assert {row["reward_source"] for row in weighted.rows} == {
        "verifier_certified",
        "same_generator_random_label_control",
        "hidden_gold_positive_control",
    }
    assert [row["reward_weight"] for row in weighted.rows[:3]] == [1.0, 0.25, 1.0]
    assert checksum == repeated_checksum
    assert checksum.startswith("sha256:")


def test_scenario_learn_4263_ready_writes_runner_and_terminal_artifact(tmp_path: Path) -> None:
    """SCENARIO-LEARN-4263-READY: prep writes corpus, runner, invocation, and no training."""

    stable = _stable_checkpoint(tmp_path)
    artifact = exp4263.run(
        output_path=tmp_path / "experiment_4263.json",
        stable_checkpoint_path=stable,
        package_dir=tmp_path / "package",
    )
    persisted = json.loads((tmp_path / "experiment_4263.json").read_text(encoding="utf-8"))
    runner_path = Path(artifact["out_of_band_runner_path"])
    corpus_path = Path(artifact["model_specs"]["reward_weighted_corpus_path"])
    runner_text = runner_path.read_text(encoding="utf-8")

    assert persisted == artifact
    assert artifact["honest_verdict"] == "complete: ready_for_out_of_band_verifier_reward_training"
    assert artifact["ready_for_out_of_band"] is True
    assert artifact["verifier_as_reward_retired"] is False
    assert artifact["verifier_is_oracle"] is True
    assert artifact["acceptance_gate"]["satisfied"] is True
    assert artifact["preconditions"]["trm_runs_touched"] is False
    assert artifact["model_specs"]["trainable_base_is_non_qwen"] is True
    assert artifact["model_specs"]["qwen_train_base_forbidden"] is True
    assert "Qwen" not in artifact["model_specs"]["trainable_base"]
    assert corpus_path.is_file()
    assert artifact["reproducibility_checksum"] == exp4263.sha256_file(corpus_path)
    assert runner_path.is_file()
    assert "AutoModelForCausalLM" in runner_text
    assert "get_peft_model" in runner_text
    assert "MIN_OPTIMIZER_STEPS = 20" in runner_text
    assert "loss_final < loss_initial" in runner_text
    assert artifact["one_command_invocation"].startswith("python3 ")
    assert str(runner_path) in artifact["one_command_invocation"]
    assert not (tmp_path / "package" / "training_result.json").exists()


def test_scenario_learn_4263_missing_corpora_blocks_without_runner(tmp_path: Path) -> None:
    """SCENARIO-LEARN-4263-BLOCKED: missing A/B/C corpora stop before prep artifacts."""

    stable = tmp_path / "missing_corpora"
    _write_jsonl(stable / "corpora" / "arm_A.jsonl", _rows("A"))

    artifact = exp4263.run(
        output_path=tmp_path / "blocked.json",
        stable_checkpoint_path=stable,
        package_dir=tmp_path / "package",
    )

    assert artifact["honest_verdict"] == "blocked_abc_corpora_missing"
    assert artifact["ready_for_out_of_band"] is False
    assert artifact["verifier_as_reward_retired"] is False
    assert artifact["out_of_band_runner_path"] == ""
    assert artifact["acceptance_gate"]["satisfied"] is True
    assert not (tmp_path / "package" / exp4263.RUNNER_FILENAME).exists()
    assert not (tmp_path / "package" / exp4263.WEIGHTED_CORPUS_FILENAME).exists()


def test_req_learn_4263_infeasible_clean_avsb_retires_axis(tmp_path: Path) -> None:
    """REQ-LEARN-4263: loaded but non-contrastive corpora retire the in-loop axis."""

    stable = _stable_checkpoint(tmp_path, b_hidden_pass=True)

    artifact = exp4263.run(
        output_path=tmp_path / "retired.json",
        stable_checkpoint_path=stable,
        package_dir=tmp_path / "package",
    )

    assert artifact["honest_verdict"] == "complete: verifier_as_reward_retired_fover_memory_ablation_stands"
    assert artifact["ready_for_out_of_band"] is False
    assert artifact["verifier_as_reward_retired"] is True
    assert artifact["retirement_evidence"]["experiment_id"] == "exp2837"
    assert artifact["retirement_evidence"]["fover_memory_ablation_delta"] == 0.0185
    assert artifact["out_of_band_runner_path"] == ""
    assert artifact["acceptance_gate"]["satisfied"] is True


def test_req_learn_4263_edge_branches_keep_blocking_honest(tmp_path: Path) -> None:
    """REQ-LEARN-4263: malformed, empty, and non-contrastive inputs fail closed."""

    root = tmp_path / "bad"
    _write_jsonl(root / "corpora" / "arm_A.jsonl", _rows("A"))
    _write_jsonl(root / "corpora" / "arm_B.jsonl", _rows("B"))
    (root / "corpora" / "arm_C.jsonl").write_text("[]\n", encoding="utf-8")

    malformed = exp4263.load_abc_corpora(root)
    empty_root = tmp_path / "empty"
    _write_jsonl(empty_root / "corpora" / "arm_A.jsonl", _rows("A"))
    _write_jsonl(empty_root / "corpora" / "arm_B.jsonl", _rows("B"))
    (empty_root / "corpora" / "arm_C.jsonl").parent.mkdir(parents=True, exist_ok=True)
    (empty_root / "corpora" / "arm_C.jsonl").write_text("\n", encoding="utf-8")

    empty = exp4263.load_abc_corpora(empty_root)

    assert exp4263._jsonable(tmp_path) == str(tmp_path)
    assert exp4263._jsonable(
        exp4263.CorpusBundle(
            ready=False,
            rows_by_arm={},
            corpus_paths={"A": str(tmp_path)},
            corpus_sizes={},
            missing=["B"],
        )
    )["corpus_paths"] == {"A": str(tmp_path)}
    assert exp4263._load_jsonl(root / "corpora" / "arm_A.jsonl") == _rows("A")
    assert malformed.ready is False
    assert malformed.error is not None
    assert "ValueError" in malformed.error
    assert empty.ready is False
    assert empty.error == "empty_C"
    assert exp4263._truthy_label({"hidden_pass": "yes"}, "hidden_pass") is None
    assert exp4263._supports_clean_avsb({"A": [], "B": [], "C": []}) == (False, "one_or_more_arms_empty")
    assert exp4263._supports_clean_avsb(
        {
            "A": [{"hidden_pass": False, "visible_perfect": False, "arm": "A_certified"}],
            "B": _rows("B"),
            "C": _rows("C"),
        }
    ) == (False, "arm_a_has_no_positive_verifier_certified_rows")
    assert exp4263._supports_clean_avsb(
        {
            "A": _rows("A"),
            "B": [{"hidden_pass": False, "visible_perfect": False, "arm": "B_control"}],
            "C": _rows("C"),
        }
    ) == (False, "arm_b_not_marked_same_generator_random_label")
