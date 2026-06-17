"""Tests for Exp 4325 second-corpus in-generation moat replication.

REQ-VERIFY-4325 / SCENARIO-VERIFY-4325: the runner must repeat the Exp 4315
reward-guided step-stitching harness on a second oracle-distinct corpus with
at least 80 measured tasks per arm and at least three seeds.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import pytest

from carnot import experiment_4325_in_generation_moat_replicate_second_corpus as exp


class TinyTokenizer:
    vocab = {
        "<unk>": 0,
        "4": 4,
        "A": 20,
        "B": 21,
        "C": 22,
        "D": 23,
        " A": 24,
        " B": 25,
        " C": 26,
        " D": 27,
    }

    def tokenize(self, data: bytes) -> list[int]:
        text = data.decode("utf-8", errors="replace")
        if text in self.vocab:
            return [self.vocab[text]]
        return [self.vocab.get(piece, 0) for piece in text.split()] or [0]


class KeywordScorer:
    mask_token_id = exp.MASK_TOKEN_ID

    def score_partial_state(self, canvas_ids: Sequence[int], step: int) -> float:
        del step
        text = "".join(
            chr(int(token_id) - 10)
            for token_id in canvas_ids
            if int(token_id) != self.mask_token_id and 0 <= int(token_id) - 10 <= 0x10FFFF
        )
        if "verified" in text or "valid" in text or "coherent" in text:
            return 0.05
        return 5.0


def _binary(tmp_path: Path, payload: bytes = b"binary") -> Path:
    path = tmp_path / "llama-diffusion-gemma-eval"
    path.write_bytes(payload)
    path.chmod(0o755)
    return path


def _cache_root_with_repo(tmp_path: Path) -> Path:
    repo_dir = tmp_path / exp.CACHE_REPO_DIRNAME
    repo_dir.mkdir(parents=True, exist_ok=True)
    (repo_dir / "refs").mkdir(exist_ok=True)
    return tmp_path


def _loader_result() -> exp.VocabLoadResult:
    return exp.VocabLoadResult(
        ok=True,
        backend="test",
        mode="embedded_vocab_metadata",
        elapsed_s=0.001,
        token_count=1,
        token_ids=(exp.MASK_TOKEN_ID,),
        detail="test loader",
        tokenizer=TinyTokenizer(),
    )


def _second_corpus_items(
    *,
    correct_n: int = 120,
    incorrect_n: int = 360,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for index in range(correct_n):
        rows.append(
            {
                "question_id": f"prmbench_correct_{index}",
                "corpus_item_id": f"pc_{index}",
                "label": "correct",
                "question": f"PRMBench validation question {index}",
                "step_text": (
                    f"The verified PRMBench step is coherent and valid: "
                    f"<<{index}+3={index + 3}>>{index + 3}."
                ),
                "source": "prmbench",
            }
        )
    for index in range(incorrect_n):
        rows.append(
            {
                "question_id": f"prmbench_incorrect_{index}",
                "corpus_item_id": f"pi_{index}",
                "label": "incorrect",
                "question": f"PRMBench adversarial question {index}",
                "step_text": (
                    f"The unsupported shortcut contradicts the premise: "
                    f"<<{index}+3={index + 9}>>{index + 9}."
                ),
                "source": "prmbench",
            }
        )
    return rows


def _scorer_artifact(path: Path, scorer_path: Path, *, built: bool = True) -> None:
    payload = {
        "partial_state_scorer_built": built,
        "partial_state_leak_free": True,
        "partial_state_auroc": 0.966143 if built else 0.0,
        "leak_ablation_auroc": 0.937365 if built else 0.0,
        "verifier_is_oracle": False,
        "scorer_path": str(scorer_path),
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    scorer_path.write_bytes(b"test scorer placeholder")


def _passing_leak_recheck() -> dict[str, object]:
    return {
        "status": "measured",
        "fresh_heldout_n": 160,
        "unmasked_auroc": 0.9,
        "answer_masked_auroc": 0.82,
        "scorer_leak_recheck_passed": True,
    }


def _failing_leak_recheck() -> dict[str, object]:
    return {
        "status": "measured",
        "fresh_heldout_n": 160,
        "unmasked_auroc": 0.9,
        "answer_masked_auroc": 0.5,
        "scorer_leak_recheck_passed": False,
    }


def _prior_for_replication(
    *, task: exp.ChoiceTask, tokenizer: object, **_: object
) -> dict[str, object]:
    del tokenizer
    index = int(task.task_id.rsplit("_", 1)[-1])
    wrong = next(choice for choice in task.choices if not choice.label)
    return {
        "status": "extracted",
        "option_logits": {
            choice.option: (
                4.4
                if choice.option == task.correct_option
                else 5.0
                if choice.option == wrong.option
                else 1.0
            )
            for choice in task.choices
        },
        "intrinsic_confidence": {
            choice.option: (
                0.98
                if choice.option == task.correct_option and index < 20
                else 0.98
                if choice.option == wrong.option and index >= 20
                else 0.1
            )
            for choice in task.choices
        },
        "mask_entropy": 1.2 if index < 10 else 0.2,
        "prompt_ids_count": 8,
        "score_shape": [exp.CANVAS_LEN, exp.VOCAB_SIZE],
    }


def _prior_for_degenerate(
    *, task: exp.ChoiceTask, tokenizer: object, **_: object
) -> dict[str, object]:
    del tokenizer
    return {
        "status": "extracted",
        "option_logits": {
            choice.option: (5.0 if choice.option == task.correct_option else 1.0)
            for choice in task.choices
        },
        "intrinsic_confidence": {choice.option: 0.5 for choice in task.choices},
        "mask_entropy": 0.0,
        "prompt_ids_count": 8,
        "score_shape": [exp.CANVAS_LEN, exp.VOCAB_SIZE],
    }


def _clean_adversarial_verify(_path: Path) -> dict[str, object]:
    return {"status": "clean", "critical_flags": [], "warn_flags": [], "returncode": 0}


def _prior_failure(*, task: exp.ChoiceTask, **_: object) -> dict[str, object]:
    return {"status": "blocked_pr_binary_eval_failed", "task_id": task.task_id}


def _common_run_kwargs(tmp_path: Path) -> dict[str, object]:
    cache_root = _cache_root_with_repo(tmp_path)
    gguf_path = tmp_path / "diffusiongemma-Q4_K_M.gguf"
    gguf_path.write_bytes(b"GGUF")
    scorer_artifact = tmp_path / "experiment_4292.json"
    scorer_path = tmp_path / "partial_state_scorer.pkl"
    _scorer_artifact(scorer_artifact, scorer_path)
    return {
        "pr_binary_path": _binary(tmp_path),
        "cache_root": cache_root,
        "scorer_artifact_path": scorer_artifact,
        "scorer_path": scorer_path,
        "resolve_gguf_fn": lambda **_: str(gguf_path),
        "vocab_loader_fn": lambda _path, _probe: _loader_result(),
        "process_rows_fn": lambda: [],
        "scorer_loader_fn": lambda _path: KeywordScorer(),
        "leak_recheck_fn": lambda **_: _passing_leak_recheck(),
        "second_corpus_items_fn": _second_corpus_items,
        "max_tasks_per_seed": 80,
        "seeds": (4325, 4326, 4327),
        "minimum_duration_s": 0.0,
        "adversarial_verify_fn": _clean_adversarial_verify,
    }


def test_req_verify_4325_spec_declares_second_corpus_replication_contract() -> None:
    """REQ-VERIFY-4325: OpenSpec declares the second-corpus replication gate."""

    spec = Path("openspec/capabilities/verification/spec.md").read_text(encoding="utf-8")
    for marker in (
        "REQ-VERIFY-4325",
        "SCENARIO-VERIFY-4325",
        "experiment_4325_in_generation_moat_replicate_second_corpus.py",
        "blocked_second_corpus_unavailable",
        "scorer_leaky_on_second_corpus",
        "in_generation_moat_replicates",
        "replication_ci95",
        "verifier_is_oracle=false",
    ):
        assert marker in spec


def test_scenario_4325_missing_pr_binary_blocks_before_second_corpus(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-4325: missing PR binary stops before corpus/scorer work."""

    def fail_resolve(**_: object) -> str:
        raise AssertionError("GGUF cache should not be inspected without PR binary")

    artifact = exp.run(
        artifact_path=tmp_path / "blocked.json",
        pr_binary_path=tmp_path / "missing-binary",
        cache_root=tmp_path,
        scorer_artifact_path=tmp_path / "missing-4292.json",
        scorer_path=tmp_path / "missing-scorer.pkl",
        resolve_gguf_fn=fail_resolve,
        second_corpus_items_fn=pytest.fail,
        minimum_duration_s=0.0,
        adversarial_verify_fn=_clean_adversarial_verify,
    )

    assert artifact["honest_verdict"] == "blocked_pr_binary"
    assert artifact["in_generation_moat_replicates"] is False
    assert artifact["controls_differentiated"] is False
    assert artifact["preconditions_checked"][0]["resource"] == "pr_binary"
    assert artifact["preconditions_checked"][1]["skipped"] is True


def test_scenario_4325_scorer_must_be_built_and_loadable(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-4325: unavailable Exp 4292 scorer blocks the run."""

    common = _common_run_kwargs(tmp_path)
    scorer_artifact = Path(common["scorer_artifact_path"])
    scorer_path = Path(common["scorer_path"])
    _scorer_artifact(scorer_artifact, scorer_path, built=False)
    artifact = exp.run(
        artifact_path=tmp_path / "blocked-scorer.json",
        option_prior_fn=pytest.fail,
        **common,
    )

    assert artifact["honest_verdict"] == "blocked_partial_state_scorer_unavailable"
    assert artifact["preconditions_checked"][-1]["resource"] == "partial_state_scorer_gate"
    assert artifact["preconditions_checked"][-1]["partial_state_scorer_built"] is False
    assert artifact["benchmark_n"] == 0


def test_scenario_4325_second_corpus_unavailable_blocks_before_leak_check(
    tmp_path: Path,
) -> None:
    """SCENARIO-VERIFY-4325: unavailable second corpus is terminal and honest."""

    common = _common_run_kwargs(tmp_path)
    common["second_corpus_items_fn"] = lambda: _second_corpus_items(
        correct_n=79,
        incorrect_n=360,
    )
    common["leak_recheck_fn"] = pytest.fail
    artifact = exp.run(
        artifact_path=tmp_path / "blocked-corpus.json",
        option_prior_fn=pytest.fail,
        **common,
    )

    assert artifact["honest_verdict"] == "blocked_second_corpus_unavailable"
    assert artifact["benchmark_n"] == 0
    assert artifact["preconditions_checked"][-1]["resource"] == "second_oracle_distinct_corpus"
    assert artifact["preconditions_checked"][-1]["ok"] is False
    assert artifact["scorer_leak_recheck_passed"] is False

    raised_common = _common_run_kwargs(tmp_path)
    raised_common["second_corpus_items_fn"] = lambda: (_ for _ in ()).throw(
        RuntimeError("missing corpus"),
    )
    raised_common["leak_recheck_fn"] = pytest.fail
    raised = exp.run(
        artifact_path=tmp_path / "blocked-corpus-error.json",
        option_prior_fn=pytest.fail,
        **raised_common,
    )
    assert raised["honest_verdict"] == "blocked_second_corpus_unavailable"
    assert "RuntimeError" in raised["preconditions_checked"][-1]["error"]


def test_scenario_4325_leak_recheck_blocks_on_second_corpus(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-4325: answer-masked collapse on corpus two stops the run."""

    common = _common_run_kwargs(tmp_path)
    common["leak_recheck_fn"] = lambda **_: _failing_leak_recheck()
    artifact = exp.run(
        artifact_path=tmp_path / "leaky.json",
        option_prior_fn=pytest.fail,
        **common,
    )

    assert artifact["honest_verdict"] == "scorer_leaky_on_second_corpus"
    assert artifact["scorer_leak_recheck_passed"] is False
    assert artifact["condition_accuracy"] == {}


def test_scenario_4325_partial_prior_failures_are_terminal(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-4325: all prior failures produce a partial verdict."""

    artifact = exp.run(
        artifact_path=tmp_path / "partial.json",
        option_prior_fn=_prior_failure,
        **_common_run_kwargs(tmp_path),
    )

    assert artifact["honest_verdict"] == "partial: second_corpus_replication_prior_eval_incomplete"
    assert artifact["benchmark_n"] == 0
    assert artifact["benchmark_failures"]


def test_scenario_4325_controls_not_differentiable_guard_fires(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-4325: bit-identical arms reject replication reporting."""

    artifact = exp.run(
        artifact_path=tmp_path / "controls.json",
        option_prior_fn=_prior_for_degenerate,
        **_common_run_kwargs(tmp_path),
    )

    assert artifact["honest_verdict"] == "controls_not_differentiable"
    assert artifact["controls_differentiated"] is False
    assert artifact["in_generation_moat_replicates"] is False
    assert artifact["control_noop_guard"]["bit_identical_accuracy_pairs"]
    assert artifact["control_noop_guard"]["bit_identical_selection_pairs"]


def test_scenario_4325_powered_non_replication_is_distinct(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-4325: powered null does not claim the moat replicated."""

    original_summary = exp.summarize_replication_rows
    monkeypatch = pytest.MonkeyPatch()

    def force_powered_null(
        rows: Sequence[dict[str, object]],
        **kwargs: object,
    ) -> dict[str, object]:
        summary = original_summary(rows, **kwargs)
        summary["carnot_minus_best_control_delta"] = -0.1
        summary["carnot_minus_self_reward_smc_delta"] = -0.1
        summary["replication_ci95"] = [-0.2, 0.0]
        summary["in_generation_moat_replicates"] = False
        return summary

    monkeypatch.setattr(exp, "summarize_replication_rows", force_powered_null)
    try:
        artifact = exp.run(
            artifact_path=tmp_path / "nonrep.json",
            option_prior_fn=_prior_for_replication,
            **_common_run_kwargs(tmp_path),
        )
    finally:
        monkeypatch.undo()

    assert artifact["honest_verdict"] == "complete: powered_non_replication_second_corpus"
    assert artifact["in_generation_moat_replicates"] is False
    assert artifact["carnot_minus_best_control_delta"] < 0.0
    assert artifact["controls_differentiated"] is True


def test_scenario_4325_complete_path_reports_replicated_moat(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-4325: complete run reports the second-corpus replication."""

    artifact_path = tmp_path / "artifact.json"
    artifact = exp.run(
        artifact_path=artifact_path,
        option_prior_fn=_prior_for_replication,
        bootstrap_resamples=2500,
        **_common_run_kwargs(tmp_path),
    )

    exp.validate_artifact(artifact)
    assert json.loads(artifact_path.read_text(encoding="utf-8")) == artifact
    assert artifact["honest_verdict"] == "complete: in_generation_moat_replicates"
    assert artifact["in_generation_moat_replicates"] is True
    assert artifact["controls_differentiated"] is True
    assert artifact["scorer_leak_recheck_passed"] is True
    assert artifact["verifier_is_oracle"] is False
    assert artifact["seed_count"] == 3
    assert artifact["benchmark_n_per_seed"] == 80
    assert artifact["benchmark_n"] == 240
    assert artifact["condition_pass_counts"]["unguided"] == 0
    assert artifact["condition_pass_counts"]["entrgi"] == 30
    assert artifact["condition_pass_counts"]["self_reward_smc"] == 60
    assert artifact["condition_pass_counts"]["carnot_stitched"] == 240
    assert artifact["carnot_minus_best_control_delta"] == pytest.approx(0.875)
    assert artifact["carnot_minus_self_reward_smc_delta"] == pytest.approx(0.75)
    assert artifact["carnot_minus_unguided_delta"] == pytest.approx(1.0)
    assert artifact["replication_ci95"][0] > 0.0
    assert artifact["model_specs"]["second_corpus"]["name"] == "step_error_balanced_v2"
    assert artifact["model_specs"]["self_reward_smc_baseline"]["uses_external_scorer"] is False
    assert artifact["adversarial_verify"]["status"] == "clean"


def test_req_verify_4325_loader_validation_and_artifact_guards(tmp_path: Path) -> None:
    """REQ-VERIFY-4325: loaders and validators enforce bare decision fields."""

    corpus_path = tmp_path / "second.json"
    corpus_path.write_text(json.dumps({"items": _second_corpus_items()}), encoding="utf-8")
    loaded = exp.load_second_corpus_items(corpus_path)
    assert len(loaded) == 480
    check = exp.check_second_corpus_available(
        items=loaded,
        corpus_path=corpus_path,
        min_tasks_per_seed=80,
        seeds=(4325, 4326, 4327),
        baseline_corpus_checksum="different",
    )
    assert check["ok"] is True
    assert check["label_counts"] == {"correct": 120, "incorrect": 360}

    object_path = tmp_path / "object.json"
    object_path.write_text(json.dumps({"items": {"not": "a list"}}), encoding="utf-8")
    with pytest.raises(ValueError, match="second corpus must be a list"):
        exp.load_second_corpus_items(object_path)

    empty_path = tmp_path / "empty.json"
    empty_path.write_text(json.dumps({"items": [{"label": "correct"}]}), encoding="utf-8")
    with pytest.raises(ValueError, match="second corpus contains no labeled step_text rows"):
        exp.load_second_corpus_items(empty_path)
    missing_check = exp.check_second_corpus_available(
        items=_second_corpus_items(correct_n=80, incorrect_n=0),
        corpus_path=tmp_path / "missing.json",
        min_tasks_per_seed=80,
        seeds=(4325, 4326, 4327),
        baseline_corpus_checksum="different",
    )
    assert missing_check["ok"] is False

    summary = exp.summarize_replication_rows(
        [
            {
                "task_id": f"t{i}",
                "unguided": False,
                "entrgi": i < 10,
                "self_reward_smc": i < 20,
                "carnot_stitched": i < 40,
            }
            for i in range(80)
        ],
        resamples=2500,
        seed=4325,
    )
    assert summary["benchmark_n"] == 80
    assert summary["replication_ci95"][0] > 0.0

    preconditions = {
        "ordered_checks": [
            {"resource": "pr_binary", "ok": True, "path": str(tmp_path / "bin")},
            {"resource": "diffusiongemma_cache", "ok": True, "gguf_path": str(tmp_path / "m.gguf")},
            {"resource": "trm_training_stand_down", "ok": True},
            {"resource": "gguf_vocab_loader", "ok": True},
            {"resource": "partial_state_scorer_gate", "ok": True, "scorer_path": "s.pkl"},
            {"resource": "second_oracle_distinct_corpus", "ok": True},
        ]
    }
    controls = exp.assess_replication_control_differentiation(
        [
            {
                "task_id": "t0",
                "unguided": False,
                "entrgi": True,
                "self_reward_smc": True,
                "carnot_stitched": True,
            },
            {
                "task_id": "t1",
                "unguided": False,
                "entrgi": False,
                "self_reward_smc": True,
                "carnot_stitched": True,
            },
        ],
        [
                {
                    "unguided_option": "A",
                    "entrgi_option": "B",
                    "self_reward_smc_option": "C",
                    "carnot_stitched_option": "C",
                },
                {
                    "unguided_option": "A",
                    "entrgi_option": "A",
                    "self_reward_smc_option": "C",
                    "carnot_stitched_option": "C",
                },
            ],
        )
    assert controls["controls_differentiated"] is False
    assert controls["bit_identical_selection_pairs"] == [["self_reward_smc", "carnot_stitched"]]

    artifact = exp.build_artifact(
        honest_verdict="complete: powered_non_replication_second_corpus",
        preconditions=preconditions,
        duration_s=1.0,
        summary=summary | {"in_generation_moat_replicates": False},
        leak_recheck=_passing_leak_recheck(),
        controls=controls | {"controls_differentiated": True, "bit_identical_selection_pairs": []},
        dynamics=exp.guidance_dynamics_diagnostic([]),
        scorer_gate={"scorer_path": "s.pkl"},
        corpus_check=check,
        corpus_items=loaded,
        adversarial_verify={"status": "clean", "critical_flags": []},
    )
    exp.validate_artifact(artifact)

    corruptions = [
        ("missing required fields", lambda a: a.pop("honest_verdict")),
        ("honest_verdict", lambda a: a.update({"honest_verdict": ""})),
        ("in_generation_moat_replicates", lambda a: a.update({"in_generation_moat_replicates": 1})),
        ("controls_differentiated", lambda a: a.update({"controls_differentiated": "true"})),
        (
            "carnot_minus_best_control_delta",
            lambda a: a.update({"carnot_minus_best_control_delta": "0"}),
        ),
        (
            "carnot_minus_self_reward_smc_delta",
            lambda a: a.update({"carnot_minus_self_reward_smc_delta": "0"}),
        ),
        ("replication_ci95", lambda a: a.update({"replication_ci95": [0.0]})),
        ("scorer_leak_recheck_passed", lambda a: a.update({"scorer_leak_recheck_passed": 1})),
        ("benchmark_n", lambda a: a.update({"benchmark_n": "80"})),
        ("verifier_is_oracle", lambda a: a.update({"verifier_is_oracle": True})),
        ("preconditions_checked", lambda a: a.update({"preconditions_checked": []})),
        ("field_principles", lambda a: a.update({"field_principles": {}})),
        ("spec_refs", lambda a: a.update({"spec_refs": []})),
        ("adversarial_verify", lambda a: a.update({"adversarial_verify": {}})),
        (
            "replication cannot be true",
            lambda a: a.update(
                {"in_generation_moat_replicates": True, "replication_ci95": [0.0, 0.2]}
            ),
        ),
    ]
    for message, mutate in corruptions:
        broken = json.loads(json.dumps(artifact))
        mutate(broken)
        with pytest.raises(ValueError, match=message):
            exp.validate_artifact(broken)

    assert exp.assess_replication_control_differentiation([], [])[
        "controls_differentiated"
    ] is False
    with pytest.raises(ValueError, match="at least one condition row"):
        exp.summarize_replication_rows([])

    sleeps: list[float] = []
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(exp.time, "sleep", lambda seconds: sleeps.append(seconds))
    try:
        exp._maybe_sleep_for_live_floor(exp.time.perf_counter(), 1.0)
    finally:
        monkeypatch.undo()
    assert sleeps and sleeps[0] > 0.0
