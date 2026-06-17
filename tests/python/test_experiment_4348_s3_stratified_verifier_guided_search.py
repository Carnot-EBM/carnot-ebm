"""Tests for Exp 4348 S3 fixed-NFE verifier-guided diffusion search.

REQ-VERIFY-4348 / SCENARIO-VERIFY-4348: the runner must put the Exp 4337
leak-robust scorer inside an S3-style denoising search loop and compare it to
compute-matched best-of-K plus intrinsic self-reward SMC controls at fixed NFE.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import pytest

from carnot import experiment_4348_s3_stratified_verifier_guided_search as exp


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


def _scorer_artifact(path: Path, scorer_path: Path, *, audit_passed: bool = True) -> None:
    payload = {
        "honest_verdict": (
            "complete: leak_robust_partial_state_scorer_built"
            if audit_passed
            else "complete_no_leak_robust_partial_state_scorer"
        ),
        "scorer_leak_audit_passed": audit_passed,
        "masked_answer_recovery_auroc": 0.559682 if audit_passed else 0.9,
        "process_ranking_auroc": 0.704633 if audit_passed else 0.5,
        "scorer_module_path": str(scorer_path),
        "verifier_is_oracle": False,
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


def _prior_for_s3(*, task: exp.ChoiceTask, tokenizer: object, **_: object) -> dict[str, object]:
    del tokenizer
    index = int(task.task_id.rsplit("_", 1)[-1])
    wrong = next(choice for choice in task.choices if not choice.label)
    best_of_k_bonus = {
        choice.option: (1.0 if choice.option == task.correct_option and index < 15 else 0.0)
        for choice in task.choices
    }
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
        "best_of_k_bonus": best_of_k_bonus,
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
        "mask_entropy": 1.1 if index < 10 else 0.2,
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
        "best_of_k_bonus": {choice.option: 0.0 for choice in task.choices},
        "intrinsic_confidence": {choice.option: 0.5 for choice in task.choices},
        "mask_entropy": 0.0,
        "prompt_ids_count": 8,
        "score_shape": [exp.CANVAS_LEN, exp.VOCAB_SIZE],
    }


def _prior_failure(*, task: exp.ChoiceTask, **_: object) -> dict[str, object]:
    return {"status": "blocked_pr_binary_eval_failed", "task_id": task.task_id}


def _clean_adversarial_verify(_path: Path) -> dict[str, object]:
    return {"status": "clean", "critical_flags": [], "warn_flags": [], "returncode": 0}


def _common_run_kwargs(tmp_path: Path) -> dict[str, object]:
    cache_root = _cache_root_with_repo(tmp_path)
    gguf_path = tmp_path / "diffusiongemma-Q4_K_M.gguf"
    gguf_path.write_bytes(b"GGUF")
    scorer_artifact = tmp_path / "experiment_4337.json"
    scorer_path = tmp_path / "dina_lrm_partial_state_scorer.pkl"
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
        "search_corpus_items_fn": _second_corpus_items,
        "max_tasks_per_seed": 80,
        "seeds": (4348, 4349, 4350),
        "minimum_duration_s": 0.0,
        "adversarial_verify_fn": _clean_adversarial_verify,
    }


def test_req_verify_4348_spec_declares_fixed_nfe_s3_contract() -> None:
    """REQ-VERIFY-4348: OpenSpec declares the fixed-NFE S3 search gate."""

    spec = Path("openspec/capabilities/verification/spec.md").read_text(encoding="utf-8")
    for marker in (
        "REQ-VERIFY-4348",
        "SCENARIO-VERIFY-4348",
        "experiment_4348_s3_stratified_verifier_guided_search.py",
        "blocked_leak_robust_scorer_unavailable",
        "scorer_leaky_in_search_corpus",
        "s3_guided_beats_control",
        "verifier_is_oracle=false",
    ):
        assert marker in spec


def test_scenario_4348_missing_pr_binary_blocks_before_scorer(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-4348: missing PR binary stops before scorer/corpus work."""

    def fail_resolve(**_: object) -> str:
        raise AssertionError("GGUF cache should not be inspected without PR binary")

    artifact = exp.run(
        artifact_path=tmp_path / "blocked.json",
        pr_binary_path=tmp_path / "missing-binary",
        cache_root=tmp_path,
        scorer_artifact_path=tmp_path / "missing-4337.json",
        scorer_path=tmp_path / "missing-scorer.pkl",
        resolve_gguf_fn=fail_resolve,
        search_corpus_items_fn=pytest.fail,
        minimum_duration_s=0.0,
        adversarial_verify_fn=_clean_adversarial_verify,
    )

    assert artifact["honest_verdict"] == "blocked_pr_binary"
    assert artifact["s3_guided_beats_control"] is False
    assert artifact["preconditions_checked"][0]["resource"] == "pr_binary"
    assert artifact["preconditions_checked"][1]["skipped"] is True


def test_scenario_4348_leak_robust_scorer_must_pass_and_load(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-4348: failed Exp 4337 audit blocks S3 search."""

    missing_gate, missing_scorer = exp.check_leak_robust_scorer_loadable_gate(
        scorer_artifact_path=tmp_path / "missing-4337.json",
        scorer_path=tmp_path / "missing-scorer.pkl",
        scorer_loader_fn=lambda _path: KeywordScorer(),
    )
    assert missing_gate["ok"] is False
    assert missing_gate["error"] == "exp4337 artifact missing"
    assert missing_scorer is None

    common = _common_run_kwargs(tmp_path)
    _scorer_artifact(
        Path(common["scorer_artifact_path"]),
        Path(common["scorer_path"]),
        audit_passed=False,
    )
    artifact = exp.run(
        artifact_path=tmp_path / "blocked-scorer.json",
        option_prior_fn=pytest.fail,
        **common,
    )

    assert artifact["honest_verdict"] == "blocked_leak_robust_scorer_unavailable"
    assert artifact["preconditions_checked"][-1]["resource"] == "leak_robust_scorer_gate"
    assert artifact["preconditions_checked"][-1]["scorer_leak_audit_passed"] is False
    assert artifact["benchmark_n"] == 0


def test_scenario_4348_search_corpus_unavailable_blocks_before_recheck(
    tmp_path: Path,
) -> None:
    """SCENARIO-VERIFY-4348: unavailable search corpus is terminal and honest."""

    common = _common_run_kwargs(tmp_path)
    common["search_corpus_items_fn"] = lambda: _second_corpus_items(
        correct_n=79,
        incorrect_n=360,
    )
    common["leak_recheck_fn"] = pytest.fail
    artifact = exp.run(
        artifact_path=tmp_path / "blocked-corpus.json",
        option_prior_fn=pytest.fail,
        **common,
    )

    assert artifact["honest_verdict"] == "blocked_search_corpus_unavailable"
    assert artifact["benchmark_n"] == 0
    assert artifact["preconditions_checked"][-1]["resource"] == "s3_search_corpus"
    assert artifact["preconditions_checked"][-1]["ok"] is False
    assert artifact["scorer_leak_recheck_passed"] is False


def test_scenario_4348_leak_recheck_blocks_search_corpus(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-4348: answer-masked collapse stops before benchmarking."""

    common = _common_run_kwargs(tmp_path)
    common["leak_recheck_fn"] = lambda **_: _failing_leak_recheck()
    artifact = exp.run(
        artifact_path=tmp_path / "leaky.json",
        option_prior_fn=pytest.fail,
        **common,
    )

    assert artifact["honest_verdict"] == "scorer_leaky_in_search_corpus"
    assert artifact["scorer_leak_recheck_passed"] is False
    assert artifact["condition_accuracy"] == {}


def test_scenario_4348_partial_prior_failures_are_terminal(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-4348: all prior failures produce a partial verdict."""

    artifact = exp.run(
        artifact_path=tmp_path / "partial.json",
        option_prior_fn=_prior_failure,
        **_common_run_kwargs(tmp_path),
    )

    assert artifact["honest_verdict"] == "partial: s3_search_prior_eval_incomplete"
    assert artifact["benchmark_n"] == 0
    assert artifact["benchmark_failures"]


def test_scenario_4348_controls_not_differentiable_guard_fires(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-4348: bit-identical arms reject utility reporting."""

    artifact = exp.run(
        artifact_path=tmp_path / "controls.json",
        option_prior_fn=_prior_for_degenerate,
        **_common_run_kwargs(tmp_path),
    )

    assert artifact["honest_verdict"] == "controls_not_differentiable"
    assert artifact["controls_differentiated"] is False
    assert artifact["s3_guided_beats_control"] is False
    assert artifact["control_noop_guard"]["bit_identical_accuracy_pairs"]
    assert artifact["control_noop_guard"]["bit_identical_selection_pairs"]


def test_scenario_4348_powered_null_is_distinct(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-4348: fixed-NFE null is reported as decision-grade."""

    original_summary = exp.summarize_s3_rows
    monkeypatch = pytest.MonkeyPatch()

    def force_powered_null(
        rows: Sequence[dict[str, object]],
        **kwargs: object,
    ) -> dict[str, object]:
        summary = original_summary(rows, **kwargs)
        summary["s3_minus_best_of_k_delta"] = -0.1
        summary["s3_minus_self_reward_smc_delta"] = -0.1
        summary["s3_gain_ci95"] = [-0.2, 0.0]
        summary["s3_guided_beats_control"] = False
        return summary

    monkeypatch.setattr(exp, "summarize_s3_rows", force_powered_null)
    try:
        artifact = exp.run(
            artifact_path=tmp_path / "null.json",
            option_prior_fn=_prior_for_s3,
            **_common_run_kwargs(tmp_path),
        )
    finally:
        monkeypatch.undo()

    assert artifact["honest_verdict"] == "complete: powered_null_s3_guided_search"
    assert artifact["s3_guided_beats_control"] is False
    assert artifact["controls_differentiated"] is True
    assert artifact["s3_minus_best_of_k_delta"] < 0.0


def test_scenario_4348_complete_path_reports_fixed_nfe_s3_gain(
    tmp_path: Path,
) -> None:
    """SCENARIO-VERIFY-4348: complete run reports S3 utility at fixed NFE."""

    artifact_path = tmp_path / "artifact.json"
    artifact = exp.run(
        artifact_path=artifact_path,
        option_prior_fn=_prior_for_s3,
        bootstrap_resamples=2500,
        **_common_run_kwargs(tmp_path),
    )

    exp.validate_artifact(artifact)
    assert json.loads(artifact_path.read_text(encoding="utf-8")) == artifact
    assert artifact["honest_verdict"] == "complete: s3_guided_beats_control"
    assert artifact["s3_guided_beats_control"] is True
    assert artifact["controls_differentiated"] is True
    assert artifact["scorer_leak_recheck_passed"] is True
    assert artifact["verifier_is_oracle"] is False
    assert artifact["seed_count"] == 3
    assert artifact["benchmark_n_per_seed"] == 80
    assert artifact["benchmark_n"] == 240
    assert artifact["nfe_budget"] == 16
    assert artifact["condition_pass_counts"]["unguided"] == 0
    assert artifact["condition_pass_counts"]["best_of_k"] == 45
    assert artifact["condition_pass_counts"]["self_reward_smc"] == 60
    assert artifact["condition_pass_counts"]["s3_carnot"] == 240
    assert artifact["s3_minus_best_of_k_delta"] == pytest.approx(0.8125)
    assert artifact["s3_minus_self_reward_smc_delta"] == pytest.approx(0.75)
    assert artifact["s3_gain_ci95"][0] > 0.0
    assert artifact["model_specs"]["s3_config"]["best_of_k"] == 4
    assert artifact["model_specs"]["s3_config"]["nfe_budget"] == 16
    assert artifact["model_specs"]["partial_state_scorer"]["source_experiment"] == 4337
    assert artifact["model_specs"]["self_reward_smc_baseline"]["uses_external_scorer"] is False
    assert artifact["adversarial_verify"]["status"] == "clean"


def test_req_verify_4348_validation_and_artifact_guards(tmp_path: Path) -> None:
    """REQ-VERIFY-4348: validators enforce bare decision fields."""

    with pytest.raises(ValueError, match="at least one condition row"):
        exp.summarize_s3_rows([])

    task = exp.build_seeded_second_corpus_tasks(
        _second_corpus_items(),
        max_tasks=1,
        seed=4348,
    )[0]
    default_proxy_selections = exp.select_s3_conditions(
        task=task,
        option_logits={choice.option: float(index) for index, choice in enumerate(task.choices)},
        intrinsic_confidence={},
        best_of_k_bonus={},
        scorer=KeywordScorer(),
        config=exp.S3SearchConfig(),
        mask_entropy=0.4,
    )
    assert default_proxy_selections["best_of_k"]["nfe_budget"] == 16
    assert default_proxy_selections["self_reward_smc"]["uses_external_scorer"] is False
    assert default_proxy_selections["s3_carnot"]["frontier_preview"]

    common = _common_run_kwargs(tmp_path)
    scorer_gate, scorer = exp.check_leak_robust_scorer_loadable_gate(
        scorer_artifact_path=Path(common["scorer_artifact_path"]),
        scorer_path=Path(common["scorer_path"]),
        scorer_loader_fn=lambda _path: KeywordScorer(),
    )
    assert scorer_gate["ok"] is True
    assert scorer is not None

    preconditions = {
        "ordered_checks": [
            {"resource": "pr_binary", "ok": True, "path": str(tmp_path / "bin")},
            {"resource": "diffusiongemma_cache", "ok": True, "gguf_path": str(tmp_path / "m.gguf")},
            {"resource": "trm_training_stand_down", "ok": True},
            {"resource": "gguf_vocab_loader", "ok": True},
            scorer_gate,
            {"resource": "s3_search_corpus", "ok": True},
        ]
    }
    controls = exp.assess_s3_control_differentiation(
        [
            {
                "task_id": "t0",
                "unguided": False,
                "best_of_k": True,
                "self_reward_smc": True,
                "s3_carnot": True,
            },
            {
                "task_id": "t1",
                "unguided": False,
                "best_of_k": False,
                "self_reward_smc": True,
                "s3_carnot": True,
            },
        ],
        [
            {
                "unguided_option": "A",
                "best_of_k_option": "B",
                "self_reward_smc_option": "C",
                "s3_carnot_option": "D",
            },
            {
                "unguided_option": "A",
                "best_of_k_option": "A",
                "self_reward_smc_option": "C",
                "s3_carnot_option": "D",
            },
        ],
    )
    assert controls["controls_differentiated"] is False

    summary = exp.summarize_s3_rows(
        [
            {
                "task_id": f"t{i}",
                "unguided": False,
                "best_of_k": i < 10,
                "self_reward_smc": i < 20,
                "s3_carnot": i < 40,
            }
            for i in range(80)
        ],
        resamples=2500,
        seed=4348,
    )
    artifact = exp.build_artifact(
        honest_verdict="complete: powered_null_s3_guided_search",
        preconditions=preconditions,
        duration_s=1.0,
        summary=summary | {"s3_guided_beats_control": False},
        leak_recheck=_passing_leak_recheck(),
        controls=controls | {"controls_differentiated": True, "bit_identical_selection_pairs": []},
        scorer_gate=scorer_gate,
        corpus_check={"ok": True, "name": "step_error_balanced_v2", "checksum": "abc"},
        corpus_items=_second_corpus_items(),
        adversarial_verify={"status": "clean", "critical_flags": []},
    )
    exp.validate_artifact(artifact)

    corruptions = [
        ("missing required fields", lambda a: a.pop("honest_verdict")),
        ("honest_verdict", lambda a: a.update({"honest_verdict": ""})),
        ("s3_guided_beats_control", lambda a: a.update({"s3_guided_beats_control": 1})),
        ("controls_differentiated", lambda a: a.update({"controls_differentiated": "true"})),
        ("s3_minus_best_of_k_delta", lambda a: a.update({"s3_minus_best_of_k_delta": "0"})),
        (
            "s3_minus_self_reward_smc_delta",
            lambda a: a.update({"s3_minus_self_reward_smc_delta": "0"}),
        ),
        ("s3_gain_ci95", lambda a: a.update({"s3_gain_ci95": [0.0]})),
        ("scorer_leak_recheck_passed", lambda a: a.update({"scorer_leak_recheck_passed": 1})),
        ("nfe_budget", lambda a: a.update({"nfe_budget": "16"})),
        ("benchmark_n", lambda a: a.update({"benchmark_n": "80"})),
        ("verifier_is_oracle", lambda a: a.update({"verifier_is_oracle": True})),
        ("preconditions_checked", lambda a: a.update({"preconditions_checked": []})),
        ("field_principles", lambda a: a.update({"field_principles": {}})),
        ("spec_refs", lambda a: a.update({"spec_refs": []})),
        ("adversarial_verify", lambda a: a.update({"adversarial_verify": {}})),
        (
            "S3 fixed-NFE gain cannot be true",
            lambda a: a.update({"s3_guided_beats_control": True, "s3_gain_ci95": [0.0, 0.2]}),
        ),
    ]
    for message, mutate in corruptions:
        broken = json.loads(json.dumps(artifact))
        mutate(broken)
        with pytest.raises(ValueError, match=message):
            exp.validate_artifact(broken)

    sleeps: list[float] = []
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(exp.time, "sleep", lambda seconds: sleeps.append(seconds))
    try:
        exp._maybe_sleep_for_live_floor(exp.time.perf_counter(), 1.0)
    finally:
        monkeypatch.undo()
    assert sleeps and sleeps[0] > 0.0
