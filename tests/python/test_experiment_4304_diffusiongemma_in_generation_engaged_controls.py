"""Tests for Exp 4304 DiffusionGemma engaged-control guidance rerun.

REQ-VERIFY-4304 / SCENARIO-VERIFY-4304: the runner must re-check the Exp 4292
partial-state scorer for leakage, require differentiated engaged controls, and
report the moat only against the best genuinely engaged non-Carnot control.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import pytest

from carnot import experiment_4304_diffusiongemma_in_generation_engaged_controls as exp


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
        "valid": 40,
        "unsupported": 41,
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

    def predict_correct_proba(self, canvas_ids: Sequence[int], step: int) -> float:
        return 0.95 if self.score_partial_state(canvas_ids, step) < 1.0 else 0.05


class MisleadingScorer:
    mask_token_id = exp.MASK_TOKEN_ID

    def score_partial_state(self, canvas_ids: Sequence[int], step: int) -> float:
        del step
        text = "".join(
            chr(int(token_id) - 10)
            for token_id in canvas_ids
            if int(token_id) != self.mask_token_id and 0 <= int(token_id) - 10 <= 0x10FFFF
        )
        if "unsupported" in text:
            return 0.05
        return 5.0


def _binary(tmp_path: Path, payload: bytes = b"binary") -> Path:
    path = tmp_path / "llama-diffusion-gemma-eval"
    path.write_bytes(payload)
    path.chmod(0o755)
    return path


def _cache_root_with_repo(tmp_path: Path) -> Path:
    repo_dir = tmp_path / exp.CACHE_REPO_DIRNAME
    repo_dir.mkdir(parents=True)
    (repo_dir / "refs").mkdir()
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
    for index in range(36):
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
    for index in range(96):
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
        "partial_state_leak_free": False,
        "partial_state_auroc": 0.966143 if built else 0.0,
        "leak_ablation_auroc": 0.51,
        "verifier_is_oracle": False,
        "scorer_path": str(scorer_path),
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    scorer_path.write_bytes(b"test scorer placeholder")


def _passing_leak_recheck() -> dict[str, object]:
    return {
        "status": "measured",
        "fresh_heldout_n": 72,
        "unmasked_auroc": 0.91,
        "answer_masked_auroc": 0.84,
        "scorer_leak_recheck_passed": True,
    }


def _failing_leak_recheck() -> dict[str, object]:
    return {
        "status": "measured",
        "fresh_heldout_n": 72,
        "unmasked_auroc": 0.91,
        "answer_masked_auroc": 0.5,
        "scorer_leak_recheck_passed": False,
    }


def _prior_for_moat(*, task: exp.ChoiceTask, tokenizer: object, **_: object) -> dict[str, object]:
    del tokenizer
    index = int(task.task_id.rsplit("_", 1)[-1])
    wrong = next(choice for choice in task.choices if not choice.label)
    correct_gap = 4.4 if index < 15 else 2.0
    return {
        "status": "extracted",
        "option_logits": {
            choice.option: (
                correct_gap
                if choice.option == task.correct_option
                else 5.0
                if choice.option == wrong.option
                else 1.0
            )
            for choice in task.choices
        },
        "mask_entropy": 1.2,
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
        "mask_entropy": 0.0,
        "prompt_ids_count": 8,
        "score_shape": [exp.CANVAS_LEN, exp.VOCAB_SIZE],
    }


def _prior_for_bounded(*, task: exp.ChoiceTask, tokenizer: object, **_: object) -> dict[str, object]:
    del tokenizer
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
        "mask_entropy": 1.2,
        "prompt_ids_count": 8,
        "score_shape": [exp.CANVAS_LEN, exp.VOCAB_SIZE],
    }


def _prior_failure(*, task: exp.ChoiceTask, **_: object) -> dict[str, object]:
    return {"status": "blocked_pr_binary_eval_failed", "task_id": task.task_id}


def test_req_verify_4304_spec_declares_engaged_control_contract() -> None:
    """REQ-VERIFY-4304: OpenSpec declares leak re-check and no-op guard fields."""

    spec = Path("openspec/capabilities/verification/spec.md").read_text(encoding="utf-8")
    for marker in (
        "REQ-VERIFY-4304",
        "SCENARIO-VERIFY-4304",
        "results/experiment_4304_diffusiongemma_in_generation_engaged_controls.py",
        "controls_not_differentiable",
        "scorer_leaky_rebuild_needed",
        "carnot_minus_best_control_delta",
        "controls_differentiated",
        "scorer_leak_recheck_passed",
        "verifier_is_oracle=false",
    ):
        assert marker in spec


def test_scenario_4304_missing_pr_binary_blocks_before_cache_or_scorer(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-4304: missing PR binary stops before cache/scorer work."""

    calls: list[str] = []

    def fail_resolve(**_: object) -> str:
        calls.append("resolve")
        raise AssertionError("GGUF cache should not be inspected without PR binary")

    artifact = exp.run(
        artifact_path=tmp_path / "blocked.json",
        pr_binary_path=tmp_path / "missing-binary",
        cache_root=tmp_path,
        scorer_artifact_path=tmp_path / "missing-4292.json",
        scorer_path=tmp_path / "missing-scorer.pkl",
        resolve_gguf_fn=fail_resolve,
        minimum_duration_s=0.0,
    )

    assert calls == []
    assert artifact["honest_verdict"] == "blocked_pr_binary"
    assert artifact["diffusiongemma_guidance_moat"] is False
    assert artifact["controls_differentiated"] is False
    assert artifact["preconditions_checked"][0]["resource"] == "pr_binary"
    assert artifact["preconditions_checked"][1]["skipped"] is True


def test_scenario_4304_scorer_must_be_built_loadable_and_oracle_distinct(
    tmp_path: Path,
) -> None:
    """SCENARIO-VERIFY-4304: Exp 4292 must provide a built loadable scorer."""

    cache_root = _cache_root_with_repo(tmp_path)
    gguf_path = tmp_path / "diffusiongemma-Q4_K_M.gguf"
    gguf_path.write_bytes(b"GGUF")
    scorer_artifact = tmp_path / "experiment_4292.json"
    scorer_path = tmp_path / "partial_state_scorer.pkl"
    _scorer_artifact(scorer_artifact, scorer_path, built=False)

    artifact = exp.run(
        artifact_path=tmp_path / "blocked-scorer.json",
        pr_binary_path=_binary(tmp_path),
        cache_root=cache_root,
        scorer_artifact_path=scorer_artifact,
        scorer_path=scorer_path,
        resolve_gguf_fn=lambda **_: str(gguf_path),
        vocab_loader_fn=lambda _path, _probe: _loader_result(),
        process_rows_fn=lambda: [],
        scorer_loader_fn=lambda _path: KeywordScorer(),
        option_prior_fn=_prior_for_moat,
        reasoning_items_fn=_reasoning_items,
        minimum_duration_s=0.0,
    )

    assert artifact["honest_verdict"] == "blocked_partial_state_scorer_unavailable"
    assert artifact["preconditions_checked"][-1]["resource"] == "partial_state_scorer_gate"
    assert artifact["preconditions_checked"][-1]["partial_state_scorer_built"] is False
    assert artifact["scorer_leak_recheck_passed"] is False


def test_scenario_4304_scorer_gate_error_edges(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-4304: missing, unreadable, and unloadable scorer gates block."""

    missing, scorer = exp.check_scorer_loadable_gate(
        scorer_artifact_path=tmp_path / "missing.json",
        scorer_path=tmp_path / "scorer.pkl",
        scorer_loader_fn=lambda _path: KeywordScorer(),
    )
    assert missing["ok"] is False
    assert "missing" in missing["error"]
    assert scorer is None

    unreadable_path = tmp_path / "bad.json"
    unreadable_path.write_text("{", encoding="utf-8")
    unreadable, scorer = exp.check_scorer_loadable_gate(
        scorer_artifact_path=unreadable_path,
        scorer_path=tmp_path / "scorer.pkl",
        scorer_loader_fn=lambda _path: KeywordScorer(),
    )
    assert unreadable["ok"] is False
    assert "unreadable" in unreadable["error"]
    assert scorer is None

    artifact_path = tmp_path / "experiment_4292.json"
    scorer_path = tmp_path / "scorer.pkl"
    _scorer_artifact(artifact_path, scorer_path)
    unloadable, scorer = exp.check_scorer_loadable_gate(
        scorer_artifact_path=artifact_path,
        scorer_path=scorer_path,
        scorer_loader_fn=lambda _path: (_ for _ in ()).throw(RuntimeError("bad pickle")),
    )
    assert unloadable["ok"] is False
    assert "RuntimeError" in unloadable["load_error"]
    assert scorer is None


def test_scenario_4304_independent_leak_recheck_helper_edges() -> None:
    """SCENARIO-VERIFY-4304: fresh answer masking is measured and fails closed."""

    measured = exp.independent_leak_recheck(scorer=KeywordScorer(), items=_reasoning_items())
    assert measured["status"] == "measured"
    assert measured["fresh_heldout_n"] > 0
    assert measured["answer_masked_auroc"] > exp.AUROC_FLOOR
    assert measured["scorer_leak_recheck_passed"] is True

    failed = exp.independent_leak_recheck(
        scorer=KeywordScorer(),
        items=[{"question_id": "only", "label": "correct", "step_text": "valid"}],
    )
    assert failed["status"] == "failed"
    assert failed["scorer_leak_recheck_passed"] is False


def test_scenario_4304_leak_recheck_blocks_before_benchmark(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-4304: answer-masked scorer collapse retires the run."""

    cache_root = _cache_root_with_repo(tmp_path)
    gguf_path = tmp_path / "diffusiongemma-Q4_K_M.gguf"
    gguf_path.write_bytes(b"GGUF")
    scorer_artifact = tmp_path / "experiment_4292.json"
    scorer_path = tmp_path / "partial_state_scorer.pkl"
    _scorer_artifact(scorer_artifact, scorer_path)

    artifact = exp.run(
        artifact_path=tmp_path / "leaky.json",
        pr_binary_path=_binary(tmp_path),
        cache_root=cache_root,
        scorer_artifact_path=scorer_artifact,
        scorer_path=scorer_path,
        resolve_gguf_fn=lambda **_: str(gguf_path),
        vocab_loader_fn=lambda _path, _probe: _loader_result(),
        process_rows_fn=lambda: [],
        scorer_loader_fn=lambda _path: KeywordScorer(),
        leak_recheck_fn=lambda **_: _failing_leak_recheck(),
        option_prior_fn=pytest.fail,
        reasoning_items_fn=_reasoning_items,
        minimum_duration_s=0.0,
    )

    assert artifact["honest_verdict"] == "scorer_leaky_rebuild_needed"
    assert artifact["scorer_leak_recheck_passed"] is False
    assert artifact["condition_accuracy"] == {}


def test_scenario_4304_controls_not_differentiable_guard_fires(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-4304: bit-identical control accuracy rejects moat reporting."""

    cache_root = _cache_root_with_repo(tmp_path)
    gguf_path = tmp_path / "diffusiongemma-Q4_K_M.gguf"
    gguf_path.write_bytes(b"GGUF")
    scorer_artifact = tmp_path / "experiment_4292.json"
    scorer_path = tmp_path / "partial_state_scorer.pkl"
    _scorer_artifact(scorer_artifact, scorer_path)

    artifact = exp.run(
        artifact_path=tmp_path / "controls.json",
        pr_binary_path=_binary(tmp_path),
        cache_root=cache_root,
        scorer_artifact_path=scorer_artifact,
        scorer_path=scorer_path,
        resolve_gguf_fn=lambda **_: str(gguf_path),
        vocab_loader_fn=lambda _path, _probe: _loader_result(),
        process_rows_fn=lambda: [],
        scorer_loader_fn=lambda _path: KeywordScorer(),
        leak_recheck_fn=lambda **_: _passing_leak_recheck(),
        option_prior_fn=_prior_for_degenerate,
        reasoning_items_fn=_reasoning_items,
        max_tasks=30,
        minimum_duration_s=0.0,
    )

    assert artifact["honest_verdict"] == "controls_not_differentiable"
    assert artifact["controls_differentiated"] is False
    assert artifact["diffusiongemma_guidance_moat"] is False
    assert artifact["control_noop_guard"]["bit_identical_accuracy_pairs"]


def test_scenario_4304_partial_bounded_and_overguided_verdicts(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-4304: partial, bounded-null, and over-guided outcomes are distinct."""

    cache_root = _cache_root_with_repo(tmp_path)
    gguf_path = tmp_path / "diffusiongemma-Q4_K_M.gguf"
    gguf_path.write_bytes(b"GGUF")
    scorer_artifact = tmp_path / "experiment_4292.json"
    scorer_path = tmp_path / "partial_state_scorer.pkl"
    _scorer_artifact(scorer_artifact, scorer_path)

    common = {
        "pr_binary_path": _binary(tmp_path),
        "cache_root": cache_root,
        "scorer_artifact_path": scorer_artifact,
        "scorer_path": scorer_path,
        "resolve_gguf_fn": lambda **_: str(gguf_path),
        "vocab_loader_fn": lambda _path, _probe: _loader_result(),
        "process_rows_fn": lambda: [],
        "leak_recheck_fn": lambda **_: _passing_leak_recheck(),
        "reasoning_items_fn": _reasoning_items,
        "max_tasks": 30,
        "minimum_duration_s": 0.0,
    }
    partial = exp.run(
        artifact_path=tmp_path / "partial.json",
        scorer_loader_fn=lambda _path: KeywordScorer(),
        option_prior_fn=_prior_failure,
        **common,
    )
    assert partial["honest_verdict"] == "partial: diffusiongemma_guidance_prior_eval_incomplete"
    assert partial["benchmark_failures"]

    bounded = exp.run(
        artifact_path=tmp_path / "bounded.json",
        scorer_loader_fn=lambda _path: KeywordScorer(),
        option_prior_fn=_prior_for_bounded,
        **common,
    )
    assert (
        bounded["honest_verdict"]
        == "complete: diffusiongemma_guidance_bounded_null_vs_engaged_control"
    )
    assert bounded["diffusiongemma_guidance_moat"] is False

    overguided = exp.run(
        artifact_path=tmp_path / "overguided.json",
        scorer_loader_fn=lambda _path: MisleadingScorer(),
        option_prior_fn=_prior_for_moat,
        **common,
    )
    assert overguided["honest_verdict"] == "complete: diffusiongemma_guidance_over_guided_diagnostic"
    assert overguided["guidance_dynamics_diagnostic"]["over_guided_finding"] is True


def test_scenario_4304_complete_path_reports_moat_against_best_control(
    tmp_path: Path,
) -> None:
    """SCENARIO-VERIFY-4304: complete run compares Carnot to best engaged control."""

    cache_root = _cache_root_with_repo(tmp_path)
    gguf_path = tmp_path / "diffusiongemma-Q4_K_M.gguf"
    gguf_path.write_bytes(b"GGUF")
    scorer_artifact = tmp_path / "experiment_4292.json"
    scorer_path = tmp_path / "partial_state_scorer.pkl"
    artifact_path = tmp_path / "artifact.json"
    _scorer_artifact(scorer_artifact, scorer_path)

    artifact = exp.run(
        artifact_path=artifact_path,
        pr_binary_path=_binary(tmp_path),
        cache_root=cache_root,
        scorer_artifact_path=scorer_artifact,
        scorer_path=scorer_path,
        resolve_gguf_fn=lambda **_: str(gguf_path),
        vocab_loader_fn=lambda _path, _probe: _loader_result(),
        process_rows_fn=lambda: [],
        scorer_loader_fn=lambda _path: KeywordScorer(),
        leak_recheck_fn=lambda **_: _passing_leak_recheck(),
        option_prior_fn=_prior_for_moat,
        reasoning_items_fn=_reasoning_items,
        max_tasks=30,
        bootstrap_resamples=2500,
        minimum_duration_s=0.0,
    )

    exp.validate_artifact(artifact)
    assert json.loads(artifact_path.read_text(encoding="utf-8")) == artifact
    assert artifact["honest_verdict"] == "complete: diffusiongemma_guidance_moat_won"
    assert artifact["diffusiongemma_guidance_moat"] is True
    assert artifact["controls_differentiated"] is True
    assert artifact["scorer_leak_recheck_passed"] is True
    assert artifact["verifier_is_oracle"] is False
    assert artifact["best_engaged_control"] == "entrgi"
    assert artifact["condition_pass_counts"]["unguided"] == 0
    assert artifact["condition_pass_counts"]["entrgi"] == 15
    assert artifact["condition_pass_counts"]["carnot"] == 30
    assert artifact["carnot_minus_best_control_delta"] == pytest.approx(0.5)
    assert artifact["carnot_minus_unguided_delta"] == pytest.approx(1.0)
    assert artifact["guidance_moat_ci95"][0] > 0.0
    assert artifact["guidance_changes_selection"]["entrgi"] is True
    assert artifact["guidance_dynamics_diagnostic"]["trajectory_stability"] >= 0.0
    assert artifact["model_specs"]["control_construction"]["engaged_controls"] == ["entrgi"]


def test_req_verify_4304_validation_and_helper_edges(tmp_path: Path) -> None:
    """REQ-VERIFY-4304: validation enforces bare fields and no-op constraints."""

    rows = [
        {"task_id": f"t{i}", "unguided": False, "entrgi": i < 10, "carnot": i < 15}
        for i in range(30)
    ]
    summary = exp.summarize_engaged_rows(rows, resamples=2500, seed=4304)
    assert summary["best_engaged_control"] == "entrgi"
    assert summary["carnot_minus_best_control_delta"] == pytest.approx(0.166667)

    preconditions = {
        "ordered_checks": [
            {"resource": "pr_binary", "ok": True, "path": str(tmp_path / "bin")},
            {"resource": "diffusiongemma_cache", "ok": True, "gguf_path": str(tmp_path / "m.gguf")},
            {"resource": "trm_training_stand_down", "ok": True},
            {"resource": "gguf_vocab_loader", "ok": True},
            {"resource": "partial_state_scorer_gate", "ok": True},
        ]
    }
    artifact = exp.build_artifact(
        honest_verdict="complete: diffusiongemma_guidance_bounded_null_vs_engaged_control",
        preconditions=preconditions,
        duration_s=1.0,
        summary=summary | {"diffusiongemma_guidance_moat": False},
        leak_recheck=_passing_leak_recheck(),
        controls=exp.assess_control_differentiation(
            rows,
            [{"unguided_option": "A", "entrgi_option": "B", "carnot_option": "B"}],
        ),
        dynamics=exp.guidance_dynamics_diagnostic([]),
        scorer_gate={"scorer_path": str(tmp_path / "scorer.pkl")},
        corpus_items=[],
    )
    exp.validate_artifact(artifact)

    corruptions = [
        ("missing required fields", lambda a: a.pop("honest_verdict")),
        ("honest_verdict", lambda a: a.update({"honest_verdict": ""})),
        ("diffusiongemma_guidance_moat", lambda a: a.update({"diffusiongemma_guidance_moat": 1})),
        ("controls_differentiated", lambda a: a.update({"controls_differentiated": "true"})),
        (
            "carnot_minus_best_control_delta",
            lambda a: a.update({"carnot_minus_best_control_delta": "0"}),
        ),
        ("scorer_leak_recheck_passed", lambda a: a.update({"scorer_leak_recheck_passed": 1})),
        ("guidance_moat_ci95", lambda a: a.update({"guidance_moat_ci95": [0.0]})),
        (
            "guidance_dynamics_diagnostic",
            lambda a: a.update({"guidance_dynamics_diagnostic": {}}),
        ),
        ("verifier_is_oracle", lambda a: a.update({"verifier_is_oracle": True})),
        ("preconditions_checked", lambda a: a.update({"preconditions_checked": []})),
        ("field_principles", lambda a: a.update({"field_principles": {}})),
        ("spec_refs", lambda a: a.update({"spec_refs": []})),
        (
            "moat cannot be true",
            lambda a: a.update(
                {"diffusiongemma_guidance_moat": True, "guidance_moat_ci95": [0.0, 0.2]}
            ),
        ),
    ]
    for message, mutate in corruptions:
        broken = json.loads(json.dumps(artifact))
        mutate(broken)
        with pytest.raises(ValueError, match=message):
            exp.validate_artifact(broken)

    assert exp.assess_control_differentiation([], [])["controls_differentiated"] is False
    with pytest.raises(ValueError, match="at least one condition row"):
        exp.summarize_engaged_rows([])
    assert exp._entrgi_candidate_prior("ambiguous neutral text") == 0.0
    assert exp._entropy_from_logits([]) == 0.0
    assert exp._covariance([1.0], [1.0, 2.0]) == 0.0

    sleeps: list[float] = []
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(exp.time, "sleep", lambda seconds: sleeps.append(seconds))
    try:
        exp._maybe_sleep_for_live_floor(exp.time.perf_counter(), 1.0)
    finally:
        monkeypatch.undo()
    assert sleeps and sleeps[0] > 0.0
