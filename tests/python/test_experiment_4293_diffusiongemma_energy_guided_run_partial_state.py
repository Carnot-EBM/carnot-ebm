"""Tests for Exp 4293 DiffusionGemma partial-state guided benchmark.

REQ-VERIFY-4293 / SCENARIO-VERIFY-4293: the runner must gate on the leak-free
Exp 4292 partial-state scorer, wire it into per-step token guidance, compare
unguided/RFG/EntRGi/Carnot conditions, and report bare moat fields plus the
guidance-dynamics diagnostic without using an executable oracle.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import pytest

from carnot import experiment_4293_diffusiongemma_energy_guided_run_partial_state as exp


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
        "return": 42,
        "pass": 43,
        "raise": 44,
        "5": 5,
        "3": 3,
        "9": 9,
    }

    def tokenize(self, data: bytes) -> list[int]:
        text = data.decode("utf-8", errors="replace")
        if text in self.vocab:
            return [self.vocab[text]]
        return [self.vocab.get(piece, 0) for piece in text.split()] or [0]

    def detokenize(self, token_ids: Sequence[int]) -> bytes:
        inverse = {value: key for key, value in self.vocab.items()}
        return " ".join(inverse.get(int(token_id), "<unk>") for token_id in token_ids).encode(
            "utf-8"
        )


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
        return 4.5


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
        return 10.0


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


def _scorer_artifact(
    path: Path, scorer_path: Path, *, built: bool = True, leak_free: bool = True
) -> None:
    payload = {
        "partial_state_scorer_built": built,
        "partial_state_leak_free": leak_free,
        "partial_state_auroc": 0.91 if built else 0.0,
        "leak_ablation_auroc": 0.87 if leak_free else 0.51,
        "verifier_is_oracle": False,
        "scorer_path": str(scorer_path),
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    scorer_path.write_bytes(b"test scorer placeholder")


def _guidance_smoke() -> dict[str, object]:
    return {
        "status": "measured",
        "examples": 2,
        "guidance_changes_selection": True,
        "guidance_selection_change_count": 2,
        "guidance_reweighted_token_count": 8,
    }


def _prior_for_task(*, task: exp.ChoiceTask, tokenizer: object, **_: object) -> dict[str, object]:
    del tokenizer
    wrong = next(choice for choice in task.choices if not choice.label)
    return {
        "status": "extracted",
        "option_logits": {
            choice.option: (5.0 if choice.option == wrong.option else 1.0)
            for choice in task.choices
        },
        "mask_entropy": 1.2,
        "prompt_ids_count": 8,
        "score_shape": [exp.CANVAS_LEN, exp.VOCAB_SIZE],
    }


def _prior_for_correct_task(
    *, task: exp.ChoiceTask, tokenizer: object, **_: object
) -> dict[str, object]:
    del tokenizer
    return {
        "status": "extracted",
        "option_logits": {
            choice.option: (5.0 if choice.option == task.correct_option else 1.0)
            for choice in task.choices
        },
        "mask_entropy": 0.9,
        "prompt_ids_count": 8,
        "score_shape": [exp.CANVAS_LEN, exp.VOCAB_SIZE],
    }


def _prior_failure(*, task: exp.ChoiceTask, **_: object) -> dict[str, object]:
    return {"status": "blocked_pr_binary_eval_failed", "task_id": task.task_id}


def test_req_verify_4293_spec_declares_partial_state_guidance_contract() -> None:
    """REQ-VERIFY-4293: OpenSpec declares the guided benchmark fields."""

    spec = Path("openspec/capabilities/verification/spec.md").read_text(encoding="utf-8")
    for marker in (
        "REQ-VERIFY-4293",
        "SCENARIO-VERIFY-4293",
        "results/experiment_4293_diffusiongemma_energy_guided_run_partial_state.py",
        "blocked_partial_state_scorer_not_leak_free",
        "diffusiongemma_guidance_moat",
        "carnot_minus_rfg_delta",
        "guidance_dynamics_diagnostic",
        "verifier_is_oracle=false",
        "partial_state_scorer_built=true",
        "partial_state_leak_free=true",
    ):
        assert marker in spec


def test_scenario_4293_missing_pr_binary_blocks_before_cache_or_scorer(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-4293: missing PR binary stops before cache/scorer work."""

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
    assert artifact["preconditions_checked"][0]["resource"] == "pr_binary"
    assert artifact["preconditions_checked"][1]["skipped"] is True


def test_scenario_4293_scorer_gate_blocks_unleak_free_artifact(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-4293: Exp 4292 must be built and leak-free."""

    cache_root = _cache_root_with_repo(tmp_path)
    gguf_path = tmp_path / "diffusiongemma-Q4_K_M.gguf"
    gguf_path.write_bytes(b"GGUF")
    scorer_artifact = tmp_path / "experiment_4292.json"
    scorer_path = tmp_path / "partial_state_scorer.pkl"
    _scorer_artifact(scorer_artifact, scorer_path, built=True, leak_free=False)

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
        option_prior_fn=_prior_for_task,
        reasoning_items_fn=_reasoning_items,
        minimum_duration_s=0.0,
    )

    assert artifact["honest_verdict"] == "blocked_partial_state_scorer_not_leak_free"
    assert artifact["preconditions_checked"][-1]["resource"] == "partial_state_scorer_gate"
    assert artifact["preconditions_checked"][-1]["partial_state_leak_free"] is False
    assert artifact["condition_accuracy"] == {}


def test_scenario_4293_scorer_gate_error_edges(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-4293: missing, unreadable, and unloadable scorer gates block."""

    missing, scorer = exp.check_scorer_gate(
        scorer_artifact_path=tmp_path / "missing.json",
        scorer_path=tmp_path / "scorer.pkl",
        scorer_loader_fn=lambda _path: KeywordScorer(),
    )
    assert missing["ok"] is False
    assert "missing" in missing["error"]
    assert scorer is None

    unreadable_path = tmp_path / "bad.json"
    unreadable_path.write_text("{", encoding="utf-8")
    unreadable, scorer = exp.check_scorer_gate(
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
    unloadable, scorer = exp.check_scorer_gate(
        scorer_artifact_path=artifact_path,
        scorer_path=scorer_path,
        scorer_loader_fn=lambda _path: (_ for _ in ()).throw(RuntimeError("bad pickle")),
    )
    assert unloadable["ok"] is False
    assert "RuntimeError" in unloadable["load_error"]
    assert scorer is None


def test_scenario_4293_complete_path_reports_moat_and_diagnostics(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-4293: complete run reports deltas, CI95, and diagnostics."""

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
        guidance_smoke_fn=lambda **_: _guidance_smoke(),
        option_prior_fn=_prior_for_task,
        reasoning_items_fn=_reasoning_items,
        max_tasks=30,
        bootstrap_resamples=2500,
        minimum_duration_s=0.0,
    )

    exp.validate_artifact(artifact)
    assert json.loads(artifact_path.read_text(encoding="utf-8")) == artifact
    assert artifact["honest_verdict"] == "complete: diffusiongemma_guidance_moat_won"
    assert artifact["diffusiongemma_guidance_moat"] is True
    assert artifact["verifier_is_oracle"] is False
    assert artifact["condition_pass_counts"]["carnot"] == 30
    assert artifact["condition_pass_counts"]["rfg"] == 0
    assert artifact["carnot_minus_rfg_delta"] == pytest.approx(1.0)
    assert artifact["carnot_minus_unguided_delta"] == pytest.approx(1.0)
    assert artifact["guidance_moat_ci95"][0] > 0.0
    assert artifact["guidance_dynamics_diagnostic"]["token_change_covariance"] >= 0.0
    assert artifact["guidance_dynamics_diagnostic"]["trajectory_stability"] >= 0.0
    assert artifact["model_specs"]["partial_state_scorer"]["verifier_is_oracle"] is False
    assert artifact["model_specs"]["denoising"]["conditions"] == [
        "unguided",
        "RFG",
        "EntRGi",
        "Carnot-partial-state-guided",
    ]


def test_scenario_4293_blocks_when_smoke_does_not_change_selection(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-4293: full benchmark waits for a real guidance hook change."""

    cache_root = _cache_root_with_repo(tmp_path)
    gguf_path = tmp_path / "diffusiongemma-Q4_K_M.gguf"
    gguf_path.write_bytes(b"GGUF")
    scorer_artifact = tmp_path / "experiment_4292.json"
    scorer_path = tmp_path / "partial_state_scorer.pkl"
    _scorer_artifact(scorer_artifact, scorer_path)

    artifact = exp.run(
        artifact_path=tmp_path / "blocked-smoke.json",
        pr_binary_path=_binary(tmp_path),
        cache_root=cache_root,
        scorer_artifact_path=scorer_artifact,
        scorer_path=scorer_path,
        resolve_gguf_fn=lambda **_: str(gguf_path),
        vocab_loader_fn=lambda _path, _probe: _loader_result(),
        process_rows_fn=lambda: [],
        scorer_loader_fn=lambda _path: KeywordScorer(),
        guidance_smoke_fn=lambda **_: {"status": "measured", "guidance_changes_selection": False},
        option_prior_fn=_prior_for_task,
        reasoning_items_fn=_reasoning_items,
        minimum_duration_s=0.0,
    )

    assert artifact["honest_verdict"] == "blocked_guidance_selection_not_changed"
    assert artifact["guidance_changes_selection"] is False


def test_scenario_4293_partial_bounded_and_overguided_verdicts(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-4293: partial, bounded-null, and over-guided verdicts are distinct."""

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
        "guidance_smoke_fn": lambda **_: _guidance_smoke(),
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
        option_prior_fn=_prior_for_correct_task,
        **common,
    )
    assert bounded["honest_verdict"] == "complete: diffusiongemma_guidance_bounded_null_vs_rfg"
    assert bounded["diffusiongemma_guidance_moat"] is False

    overguided = exp.run(
        artifact_path=tmp_path / "overguided.json",
        scorer_loader_fn=lambda _path: MisleadingScorer(),
        option_prior_fn=_prior_for_correct_task,
        config=exp.GuidanceConfig(steps=4, guidance_lambda=5.0, candidate_count=4),
        **common,
    )
    assert (
        overguided["honest_verdict"] == "complete: diffusiongemma_guidance_over_guided_diagnostic"
    )
    assert overguided["guidance_dynamics_diagnostic"]["over_guided_finding"] is True


def test_req_verify_4293_bounded_null_and_validation_edges(tmp_path: Path) -> None:
    """REQ-VERIFY-4293: moat bool stays false when CI95 includes zero."""

    rows = [
        {"task_id": f"t{i}", "unguided": i < 10, "rfg": i < 15, "entrgi": i < 15, "carnot": i < 15}
        for i in range(30)
    ]
    summary = exp.summarize_condition_rows(rows, resamples=2500, seed=4293)
    assert summary["diffusiongemma_guidance_moat"] is False
    assert summary["carnot_minus_rfg_delta"] == pytest.approx(0.0)

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
        honest_verdict="complete: diffusiongemma_guidance_bounded_null_vs_rfg",
        preconditions=preconditions,
        duration_s=1.0,
        summary=summary,
        guidance_smoke=_guidance_smoke(),
        dynamics=exp.guidance_dynamics_diagnostic(
            [
                {
                    "mask_entropy": 1.0,
                    "unguided_option": "A",
                    "rfg_option": "A",
                    "carnot_option": "A",
                    "rfg_correct": True,
                    "carnot_correct": True,
                }
            ]
        ),
        scorer_gate={"scorer_path": str(tmp_path / "scorer.pkl")},
        benchmark_records=[],
    )
    exp.validate_artifact(artifact)

    corruptions = [
        ("missing required fields", lambda a: a.pop("honest_verdict")),
        ("honest_verdict", lambda a: a.update({"honest_verdict": ""})),
        ("diffusiongemma_guidance_moat", lambda a: a.update({"diffusiongemma_guidance_moat": 1})),
        ("carnot_minus_rfg_delta", lambda a: a.update({"carnot_minus_rfg_delta": "0"})),
        (
            "carnot_minus_unguided_delta",
            lambda a: a.update({"carnot_minus_unguided_delta": None}),
        ),
        ("guidance_moat_ci95", lambda a: a.update({"guidance_moat_ci95": [0.0]})),
        ("guidance_dynamics_diagnostic", lambda a: a.update({"guidance_dynamics_diagnostic": {}})),
        ("verifier_is_oracle", lambda a: a.update({"verifier_is_oracle": True})),
        ("preconditions_checked", lambda a: a.update({"preconditions_checked": []})),
        ("field_principles", lambda a: a.update({"field_principles": {}})),
        ("spec_refs", lambda a: a.update({"spec_refs": []})),
        ("moat cannot be true", lambda a: a.update({"diffusiongemma_guidance_moat": True})),
    ]
    for message, mutate in corruptions:
        broken = json.loads(json.dumps(artifact))
        mutate(broken)
        with pytest.raises(ValueError, match=message):
            exp.validate_artifact(broken)


def test_req_verify_4293_task_building_and_diagnostic_edges() -> None:
    """REQ-VERIFY-4293: task construction and diagnostics reject bad inputs."""

    with pytest.raises(ValueError, match="at least 30"):
        exp.build_choice_tasks(_reasoning_items()[:10], max_tasks=30, seed=4293)

    task = exp.build_choice_tasks(_reasoning_items(), max_tasks=1, seed=4293)[0]
    assert len(task.choices) == 4
    assert sum(1 for choice in task.choices if choice.label) == 1
    assert "Return only" in task.prompt

    with pytest.raises(ValueError, match="at least one condition row"):
        exp.summarize_condition_rows([])
    with pytest.raises(ValueError, match="same length"):
        exp.bootstrap_delta_ci([True], [False, True], resamples=10, seed=1)

    assert exp._option_token_id(TinyTokenizer(), "A") in {20, 24}
    assert (
        exp._option_token_id(type("EmptyTokenizer", (), {"tokenize": lambda self, data: []})(), "Z")
        == 0
    )
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

    diagnostic = exp.guidance_dynamics_diagnostic(
        [
            {
                "mask_entropy": 1.0,
                "unguided_option": "A",
                "rfg_option": "A",
                "carnot_option": "B",
                "rfg_correct": False,
                "carnot_correct": True,
            },
            {
                "mask_entropy": 0.5,
                "unguided_option": "C",
                "rfg_option": "C",
                "carnot_option": "C",
                "rfg_correct": True,
                "carnot_correct": True,
            },
        ]
    )
    assert diagnostic["mask_entropy_mean"] == pytest.approx(0.75)
    assert diagnostic["token_change_rate"] == pytest.approx(0.5)
    assert diagnostic["trajectory_stability"] == pytest.approx(0.5)
