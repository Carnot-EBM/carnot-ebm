"""Tests for Exp 4326 adaptive guided generation scale-up.

REQ-VERIFY-4326 / SCENARIO-VERIFY-4326: the runner must turn the Exp 4315
external reward into a bounded adaptive loop, keep no-adaptation and engaged
controls, re-check scorer leakage, and emit a decision-grade artifact.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import pytest

from carnot import experiment_4326_adaptive_guided_generation_scaleup as exp


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


def _reasoning_items() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for index in range(52):
        rows.append(
            {
                "question_id": f"correct_{index}",
                "corpus_item_id": f"c_{index}",
                "label": "correct",
                "step_text": (
                    f"The verified arithmetic relation is coherent and valid: "
                    f"<<{index}+2={index + 2}>>{index + 2}."
                ),
            }
        )
    for index in range(140):
        rows.append(
            {
                "question_id": f"incorrect_{index}",
                "corpus_item_id": f"i_{index}",
                "label": "incorrect",
                "step_text": (
                    f"The unsupported shortcut contradicts the premise and guesses "
                    f"<<{index}+2={index + 9}>>{index + 9}."
                ),
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
        "fresh_heldout_n": 96,
        "unmasked_auroc": 0.91,
        "answer_masked_auroc": 0.84,
        "scorer_leak_recheck_passed": True,
    }


def _failing_leak_recheck() -> dict[str, object]:
    return {
        "status": "measured",
        "fresh_heldout_n": 96,
        "unmasked_auroc": 0.91,
        "answer_masked_auroc": 0.5,
        "scorer_leak_recheck_passed": False,
    }


def _prior_for_adaptive_win(
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


def _prior_for_bounded_null(
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
                if choice.option == wrong.option and index < 10
                else 30.0
                if choice.option == wrong.option
                else 1.0
            )
            for choice in task.choices
        },
        "intrinsic_confidence": {
            choice.option: (0.98 if choice.option == task.correct_option and index < 10 else 0.1)
            for choice in task.choices
        },
        "mask_entropy": 1.2 if index < 10 else 0.2,
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
        "reasoning_items_fn": _reasoning_items,
        "option_prior_fn": _prior_for_adaptive_win,
        "max_tasks": 40,
        "bootstrap_resamples": 2500,
        "minimum_duration_s": 0.0,
        "adversarial_verify_fn": _clean_adversarial_verify,
    }


def test_req_verify_4326_spec_declares_adaptive_scaleup_contract() -> None:
    """REQ-VERIFY-4326: OpenSpec declares the adaptive scale-up gate."""

    spec = Path("openspec/capabilities/verification/spec.md").read_text(encoding="utf-8")
    for marker in (
        "REQ-VERIFY-4326",
        "SCENARIO-VERIFY-4326",
        "experiment_4326_adaptive_guided_generation_scaleup.py",
        "adaptive_guidance_beats_control",
        "adaptive_ci95",
        "verified_citation",
        "domain_used",
        "controls_not_differentiable",
        "scorer_leaky_rebuild_needed",
        "verifier_is_oracle=false",
    ):
        assert marker in spec


def test_scenario_4326_missing_pr_binary_blocks_before_cache_or_scorer(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-4326: missing PR binary stops before cache/scorer work."""

    def fail_resolve(**_: object) -> str:
        raise AssertionError("GGUF cache should not be inspected without PR binary")

    artifact = exp.run(
        artifact_path=tmp_path / "blocked.json",
        pr_binary_path=tmp_path / "missing-binary",
        cache_root=tmp_path,
        scorer_artifact_path=tmp_path / "missing-4292.json",
        scorer_path=tmp_path / "missing-scorer.pkl",
        resolve_gguf_fn=fail_resolve,
        minimum_duration_s=0.0,
        adversarial_verify_fn=_clean_adversarial_verify,
    )

    assert artifact["honest_verdict"] == "blocked_pr_binary"
    assert artifact["adaptive_guidance_beats_control"] is False
    assert artifact["preconditions_checked"][0]["resource"] == "pr_binary"
    assert any(row["resource"] == "verified_citation" for row in artifact["preconditions_checked"])


def test_scenario_4326_missing_verified_citation_blocks_before_scorer(tmp_path: Path) -> None:
    """REQ-VERIFY-4326: the adaptive-method citation is a precondition."""

    common = _common_run_kwargs(tmp_path)
    common["verified_citation"] = ""
    common["scorer_loader_fn"] = lambda _path: pytest.fail("scorer should not load")

    artifact = exp.run(artifact_path=tmp_path / "citation.json", **common)

    assert artifact["honest_verdict"] == "blocked_verified_citation"
    assert artifact["preconditions_checked"][-1]["resource"] == "verified_citation"
    assert artifact["preconditions_checked"][-1]["ok"] is False


def test_scenario_4326_leak_recheck_blocks_before_benchmark(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-4326: answer-masked scorer collapse stops the run."""

    common = _common_run_kwargs(tmp_path)
    common["leak_recheck_fn"] = lambda **_: _failing_leak_recheck()
    common["option_prior_fn"] = pytest.fail

    artifact = exp.run(artifact_path=tmp_path / "leaky.json", **common)

    assert artifact["honest_verdict"] == "scorer_leaky_rebuild_needed"
    assert artifact["scorer_leak_recheck_passed"] is False
    assert artifact["adaptive_guidance_beats_control"] is False
    assert artifact["condition_accuracy"] == {}


def test_scenario_4326_scorer_must_be_built_and_loadable(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-4326: Exp 4292 must provide a built loadable scorer."""

    common = _common_run_kwargs(tmp_path)
    scorer_artifact = Path(common["scorer_artifact_path"])
    scorer_path = Path(common["scorer_path"])
    _scorer_artifact(scorer_artifact, scorer_path, built=False)

    artifact = exp.run(artifact_path=tmp_path / "blocked-scorer.json", **common)

    assert artifact["honest_verdict"] == "blocked_partial_state_scorer_unavailable"
    assert artifact["preconditions_checked"][-1]["resource"] == "partial_state_scorer_gate"
    assert artifact["preconditions_checked"][-1]["partial_state_scorer_built"] is False


def test_scenario_4326_controls_not_differentiable_guard_fires(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-4326: bit-identical adaptive/no-adaptation arms are rejected."""

    common = _common_run_kwargs(tmp_path)
    common["option_prior_fn"] = _prior_for_degenerate

    artifact = exp.run(artifact_path=tmp_path / "controls.json", **common)

    assert artifact["honest_verdict"] == "controls_not_differentiable"
    assert artifact["controls_differentiated"] is False
    assert artifact["control_noop_guard"]["adaptive_vs_no_adaptation_bit_identical"] is True
    assert artifact["adaptive_guidance_beats_control"] is False


def test_scenario_4326_partial_and_bounded_null_verdicts(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-4326: incomplete and bounded-null outcomes are distinct."""

    partial_common = _common_run_kwargs(tmp_path)
    partial_common["option_prior_fn"] = _prior_failure
    partial = exp.run(artifact_path=tmp_path / "partial.json", **partial_common)
    assert partial["honest_verdict"] == "partial: adaptive_guided_generation_prior_eval_incomplete"
    assert partial["benchmark_n"] == 0

    bounded_common = _common_run_kwargs(tmp_path)
    bounded_common["option_prior_fn"] = _prior_for_bounded_null
    bounded = exp.run(artifact_path=tmp_path / "bounded.json", **bounded_common)
    assert bounded["honest_verdict"] == "complete: adaptive_guidance_bounded_to_stitching_null"
    assert bounded["controls_differentiated"] is True
    assert bounded["adaptive_guidance_beats_control"] is False


def test_scenario_4326_complete_path_reports_adaptive_win(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-4326: adaptive guidance beats no-adaptation and engaged controls."""

    artifact_path = tmp_path / "artifact.json"
    artifact = exp.run(artifact_path=artifact_path, **_common_run_kwargs(tmp_path))

    exp.validate_artifact(artifact)
    assert json.loads(artifact_path.read_text(encoding="utf-8")) == artifact
    assert artifact["honest_verdict"] == "complete: adaptive_guidance_moat_scaled"
    assert artifact["adaptive_guidance_beats_control"] is True
    assert artifact["controls_differentiated"] is True
    assert artifact["scorer_leak_recheck_passed"] is True
    assert artifact["domain_used"] == "reasoning_corpus_fallback"
    assert artifact["verified_citation"].endswith("2603.12554")
    assert artifact["verifier_is_oracle"] is False
    assert artifact["condition_pass_counts"]["unguided"] == 0
    assert artifact["condition_pass_counts"]["entrgi"] == 10
    assert artifact["condition_pass_counts"]["carnot_adaptive"] == 40
    assert artifact["best_control"] == "entrgi"
    assert artifact["carnot_minus_best_control_delta"] == pytest.approx(0.75)
    assert artifact["adaptive_ci95"][0] > 0.0
    assert artifact["guidance_dynamics_diagnostic"]["adaptive_step_count_mean"] == 4.0
    assert artifact["model_specs"]["adaptive_method"]["uses_external_scorer"] is True
    assert artifact["model_specs"]["control_construction"]["no_adaptation"] == "unguided"
    assert artifact["adversarial_verify"]["status"] == "clean"


def test_req_verify_4326_benchmark_continues_after_prior_failures() -> None:
    """REQ-VERIFY-4326: benchmark keeps trying until it has 40 measured rows."""

    tasks = exp.build_choice_tasks(_reasoning_items(), max_tasks=48, seed=4326)

    def flaky_prior(
        *, task: exp.ChoiceTask, tokenizer: object, **kwargs: object
    ) -> dict[str, object]:
        index = int(task.task_id.rsplit("_", 1)[-1])
        if index < 8:
            return {"status": "blocked_pr_binary_eval_failed", "task_id": task.task_id}
        return _prior_for_adaptive_win(task=task, tokenizer=tokenizer, **kwargs)

    benchmark = exp.run_adaptive_benchmark(
        tasks=tasks,
        scorer=KeywordScorer(),
        tokenizer=TinyTokenizer(),
        pr_binary_path=Path("/tmp/pr-binary"),
        gguf_path="/tmp/diffusiongemma.gguf",
        config=exp._default_adaptive_config(),
        option_prior_fn=flaky_prior,
        target_successes=40,
    )

    assert len(benchmark["failures"]) == 8
    assert len(benchmark["rows"]) == 40
    assert benchmark["rows"][0]["task_id"] == "fover_math_choice_008"
    assert benchmark["rows"][-1]["task_id"] == "fover_math_choice_047"


def test_req_verify_4326_validation_and_helper_edges(tmp_path: Path) -> None:
    """REQ-VERIFY-4326: validation enforces bare fields and gate constraints."""

    rows = [
        {
            "task_id": f"t{i}",
            "unguided": False,
            "entrgi": i < 10,
            "carnot_adaptive": i < 30,
        }
        for i in range(40)
    ]
    summary = exp.summarize_adaptive_rows(rows, resamples=2500, seed=4326)
    assert summary["best_control"] == "entrgi"
    assert summary["carnot_minus_best_control_delta"] == pytest.approx(0.5)

    preconditions = {
        "ordered_checks": [
            {"resource": "pr_binary", "ok": True, "path": str(tmp_path / "bin")},
            {"resource": "diffusiongemma_cache", "ok": True, "gguf_path": str(tmp_path / "m.gguf")},
            {"resource": "trm_training_stand_down", "ok": True},
            {"resource": "gguf_vocab_loader", "ok": True},
            {"resource": "verified_citation", "ok": True, "citation": exp.VERIFIED_CITATION},
            {"resource": "partial_state_scorer_gate", "ok": True},
        ]
    }
    artifact = exp.build_artifact(
        honest_verdict="complete: adaptive_guidance_bounded_to_stitching_null",
        preconditions=preconditions,
        duration_s=1.0,
        summary=summary | {"adaptive_guidance_beats_control": False},
        leak_recheck=_passing_leak_recheck(),
        controls=exp.assess_adaptive_control_differentiation(
            rows,
            [
                {
                    "unguided_option": "A",
                    "entrgi_option": "B",
                    "carnot_adaptive_option": "C",
                    "adaptive_step_count": 4,
                    "adaptive_changed_steps": 2,
                }
            ],
        ),
        dynamics=exp.guidance_dynamics_diagnostic([]),
        scorer_gate={"scorer_path": str(tmp_path / "scorer.pkl")},
        domain_check=exp.default_domain_check(),
        corpus_items=[],
        adversarial_verify={"status": "clean", "critical_flags": []},
    )
    exp.validate_artifact(artifact)

    corruptions = [
        ("missing required fields", lambda a: a.pop("honest_verdict")),
        ("honest_verdict", lambda a: a.update({"honest_verdict": ""})),
        (
            "adaptive_guidance_beats_control",
            lambda a: a.update({"adaptive_guidance_beats_control": 1}),
        ),
        (
            "carnot_minus_best_control_delta",
            lambda a: a.update({"carnot_minus_best_control_delta": "0"}),
        ),
        ("adaptive_ci95", lambda a: a.update({"adaptive_ci95": [0.0]})),
        ("controls_differentiated", lambda a: a.update({"controls_differentiated": "true"})),
        ("scorer_leak_recheck_passed", lambda a: a.update({"scorer_leak_recheck_passed": 1})),
        ("domain_used", lambda a: a.update({"domain_used": "arc"})),
        ("verified_citation", lambda a: a.update({"verified_citation": ""})),
        ("verifier_is_oracle", lambda a: a.update({"verifier_is_oracle": True})),
        ("preconditions_checked", lambda a: a.update({"preconditions_checked": []})),
        ("field_principles", lambda a: a.update({"field_principles": {}})),
        ("spec_refs", lambda a: a.update({"spec_refs": []})),
        ("guidance_dynamics_diagnostic", lambda a: a.update({"guidance_dynamics_diagnostic": {}})),
        ("adversarial_verify", lambda a: a.update({"adversarial_verify": {}})),
        (
            "adaptive win cannot be true",
            lambda a: a.update(
                {"adaptive_guidance_beats_control": True, "adaptive_ci95": [0.0, 0.2]}
            ),
        ),
    ]
    for message, mutate in corruptions:
        broken = json.loads(json.dumps(artifact))
        mutate(broken)
        with pytest.raises(ValueError, match=message):
            exp.validate_artifact(broken)

    assert exp.assess_adaptive_control_differentiation([], [])["controls_differentiated"] is False
    with pytest.raises(ValueError, match="at least one condition row"):
        exp.summarize_adaptive_rows([])
    assert exp.default_domain_check()["domain_used"] == "reasoning_corpus_fallback"
    assert exp.record_verified_citation("")["ok"] is False
    assert exp.guidance_dynamics_diagnostic([])["status"] == "not_run"
