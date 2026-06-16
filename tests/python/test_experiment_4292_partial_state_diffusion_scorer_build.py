"""Tests for Exp 4292 partial-state DiffusionGemma scorer build.

REQ-VERIFY-4292 / SCENARIO-VERIFY-4292: the runner must build a
loadable learned value head for masked partial canvases, report held-out AUROC,
and prove the signal survives masking answer-bearing cells before Exp 4293 can
use it as an oracle-distinct guidance scorer.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import pytest

from carnot import experiment_4292_partial_state_diffusion_scorer_build as exp
from carnot.verify.partial_state_diffusion_scorer import (
    ByteCanvasEncoder,
    PartialStateDiffusionScorer,
    _rank_auroc,
    build_partial_state_records,
    find_answer_spans,
    mask_answer_bearing_cells,
    partial_state_auroc,
    split_items_task_disjoint,
)


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


def _energy_prior() -> dict[str, object]:
    return {
        "status": "extracted",
        "eval_rc": 0,
        "score_shape": [exp.CANVAS_LEN, exp.VOCAB_SIZE],
        "score_finite_sample": True,
        "logits_file_size_bytes": exp.CANVAS_LEN * exp.VOCAB_SIZE * 4,
        "expected_logits_file_size_bytes": exp.CANVAS_LEN * exp.VOCAB_SIZE * 4,
        "prompt_ids_count": 5,
    }


def _loader_result() -> exp.VocabLoadResult:
    return exp.VocabLoadResult(
        ok=True,
        backend="test",
        mode="embedded_vocab_metadata",
        elapsed_s=0.001,
        token_count=1,
        token_ids=(exp.MASK_TOKEN_ID,),
        detail="test loader",
        tokenizer=object(),
    )


def _reasoning_items() -> list[dict[str, object]]:
    positives = [
        "First compute the verified equation {a}+{b}. The balanced arithmetic check matches the premise, so the step is valid: <<{a}+{b}={c}>>{c}.",
        "Because both quantities are accounted for, the derivation is supported by the equation {a}*{b}. Therefore the final step is correct: <<{a}*{b}={c}>>{c}.",
        "Next apply the conservation relation and verify each term before concluding. The consistency check succeeds, giving boxed{{{c}}}.",
        "The scratch work repeats the premise, checks the operation, and preserves units. Hence the answer follows from a verified calculation: <<{a}+{b}={c}>>{c}.",
        "Since the intermediate total agrees with the stated constraint, the reasoning is coherent. Therefore the result is boxed{{{c}}}.",
        "The step cites the relevant numbers, performs the operation once, and keeps the same unit. The verified value is <<{a}+{b}={c}>>{c}.",
    ]
    negatives = [
        "This step guesses a value without using the premise and contradicts the earlier total. However it still prints an answer: <<{a}+{b}={c}>>{c}.",
        "The derivation changes units midstream and the unsupported shortcut skips the required equation. The final number is boxed{{{c}}}.",
        "Instead of checking the relationship, the step copies a plausible number after an unrelated sentence. This contradiction yields <<{a}*{b}={c}>>{c}.",
        "The scratch work omits the condition, introduces an extra quantity, and cannot justify the answer. It ends with boxed{{{c}}}.",
        "Although the text sounds confident, the arithmetic relation is unsupported and conflicts with the premise. It claims <<{a}+{b}={c}>>{c}.",
        "A wrong shortcut replaces the required operation with a guess, so the partial reasoning is inconsistent before the final answer boxed{{{c}}}.",
    ]
    rows: list[dict[str, object]] = []
    for copy in range(4):
        for index, template in enumerate(positives):
            a, b = index + 2 + copy, index + 4
            rows.append(
                {
                    "question_id": f"pos_{copy}_{index}",
                    "step_text": template.format(a=a, b=b, c=a + b),
                    "label": "correct",
                }
            )
        for index, template in enumerate(negatives):
            a, b = index + 5 + copy, index + 3
            rows.append(
                {
                    "question_id": f"neg_{copy}_{index}",
                    "step_text": template.format(a=a, b=b, c=a + b + 7),
                    "label": "incorrect",
                }
            )
    return rows


def test_req_verify_4292_spec_declares_partial_state_scorer_contract() -> None:
    """REQ-VERIFY-4292: OpenSpec declares scorer, AUROC, and leak fields."""

    spec = Path("openspec/capabilities/verification/spec.md").read_text(encoding="utf-8")
    for marker in (
        "REQ-VERIFY-4292",
        "SCENARIO-VERIFY-4292",
        "results/experiment_4292_partial_state_diffusion_scorer_build.py",
        "score_partial_state(canvas, step) -> energy",
        "partial_state_scorer_built",
        "partial_state_leak_free",
        "partial_state_auroc",
        "leak_ablation_auroc",
        "verifier_is_oracle=false",
        "llama-diffusion-gemma-eval",
    ):
        assert marker in spec


def test_req_verify_4292_value_head_scores_partial_states_and_survives_leak_audit(
    tmp_path: Path,
) -> None:
    """REQ-VERIFY-4292: held-out partial canvases clear AUROC after answer masking."""

    train_items, heldout_items = split_items_task_disjoint(
        _reasoning_items(), heldout_fraction=0.34, seed=4292
    )
    encoder = ByteCanvasEncoder(canvas_len=160, mask_token_id=exp.MASK_TOKEN_ID)
    train_records = build_partial_state_records(train_items, encoder=encoder)
    heldout_records = build_partial_state_records(heldout_items, encoder=encoder)

    scorer = PartialStateDiffusionScorer(random_seed=4292, max_features=512)
    scorer.fit(train_records)
    heldout_auroc = partial_state_auroc(scorer, heldout_records)
    leak_auroc = partial_state_auroc(scorer, heldout_records, mask_answer_cells=True)

    assert heldout_auroc > 0.6
    assert leak_auroc > 0.6
    assert any(record.answer_cell_indices for record in heldout_records)
    revealed_answer_record = next(
        record
        for record in heldout_records
        if any(record.canvas_ids[index] != exp.MASK_TOKEN_ID for index in record.answer_cell_indices)
    )
    masked = mask_answer_bearing_cells(revealed_answer_record)
    assert masked.canvas_ids != revealed_answer_record.canvas_ids
    assert all(masked.canvas_ids[index] == exp.MASK_TOKEN_ID for index in masked.answer_cell_indices)

    scorer_path = tmp_path / "partial_state_scorer.pkl"
    scorer.save(scorer_path)
    loaded = PartialStateDiffusionScorer.load(scorer_path)
    assert loaded.score_partial_state(
        revealed_answer_record.canvas_ids, revealed_answer_record.step
    ) == pytest.approx(
        scorer.score_partial_state(revealed_answer_record.canvas_ids, revealed_answer_record.step)
    )


def test_scenario_4292_missing_pr_binary_blocks_before_cache(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-4292: missing PR binary stops before GGUF inspection."""

    calls: list[str] = []

    def fail_resolve(**_: object) -> str:
        calls.append("resolve")
        raise AssertionError("GGUF cache should not be inspected without PR binary")

    artifact = exp.run(
        artifact_path=tmp_path / "blocked.json",
        scorer_path=tmp_path / "scorer.pkl",
        pr_binary_path=tmp_path / "missing-binary",
        cache_root=tmp_path,
        resolve_gguf_fn=fail_resolve,
        minimum_duration_s=0.0,
    )

    assert calls == []
    assert artifact["honest_verdict"] == "blocked_pr_binary"
    assert artifact["partial_state_scorer_built"] is False
    assert artifact["partial_state_leak_free"] is False
    assert artifact["preconditions_checked"][0]["resource"] == "pr_binary"
    assert artifact["preconditions_checked"][1]["skipped"] is True


def test_scenario_4292_complete_path_writes_loadable_leak_free_artifact(
    tmp_path: Path,
) -> None:
    """SCENARIO-VERIFY-4292: complete run persists scorer and bare gate fields."""

    cache_root = _cache_root_with_repo(tmp_path)
    gguf_path = tmp_path / "diffusiongemma-Q4_K_M.gguf"
    gguf_path.write_bytes(b"GGUF")
    artifact_path = tmp_path / "artifact.json"
    scorer_path = tmp_path / "partial_state_scorer.pkl"

    artifact = exp.run(
        artifact_path=artifact_path,
        scorer_path=scorer_path,
        pr_binary_path=_binary(tmp_path),
        cache_root=cache_root,
        resolve_gguf_fn=lambda **_: str(gguf_path),
        vocab_loader_fn=lambda _path, _probe: _loader_result(),
        process_rows_fn=lambda: [],
        energy_prior_fn=lambda **_: _energy_prior(),
        reasoning_items_fn=_reasoning_items,
        minimum_duration_s=0.0,
    )

    exp.validate_artifact(artifact)
    assert json.loads(artifact_path.read_text(encoding="utf-8")) == artifact
    assert artifact["honest_verdict"] == "complete: partial_state_diffusion_scorer_built_leak_free"
    assert artifact["partial_state_scorer_built"] is True
    assert artifact["partial_state_leak_free"] is True
    assert artifact["partial_state_auroc"] > 0.6
    assert artifact["leak_ablation_auroc"] > 0.6
    assert artifact["verifier_is_oracle"] is False
    assert artifact["heldout_n"] > 0
    assert artifact["leak_audit"]["answer_masked_cells"] > 0
    assert Path(artifact["scorer_path"]).exists()
    assert PartialStateDiffusionScorer.load(Path(artifact["scorer_path"])).is_fitted


def test_scenario_4292_blocked_cache_and_trm_training(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-4292: cache and TRM preconditions block before training."""

    cache_blocked = exp.run(
        artifact_path=tmp_path / "cache-blocked.json",
        scorer_path=tmp_path / "unused-cache.pkl",
        pr_binary_path=_binary(tmp_path),
        cache_root=tmp_path,
        resolve_gguf_fn=lambda **_: None,
        process_rows_fn=lambda: [],
        minimum_duration_s=0.0,
    )
    assert cache_blocked["honest_verdict"] == "blocked_diffusiongemma_not_cached"
    assert cache_blocked["partial_state_scorer_built"] is False

    cache_root = _cache_root_with_repo(tmp_path / "trm")
    gguf_path = tmp_path / "model.gguf"
    gguf_path.write_bytes(b"GGUF")
    trm_blocked = exp.run(
        artifact_path=tmp_path / "trm-blocked.json",
        scorer_path=tmp_path / "unused-trm.pkl",
        pr_binary_path=_binary(tmp_path / "trm"),
        cache_root=cache_root,
        resolve_gguf_fn=lambda **_: str(gguf_path),
        vocab_loader_fn=lambda _path, _probe: _loader_result(),
        process_rows_fn=lambda: [{"pid": 123, "command": "torchrun train_trm.py"}],
        minimum_duration_s=0.0,
    )
    assert trm_blocked["honest_verdict"] == "blocked_trm_training_active"
    assert trm_blocked["preconditions_checked"][2]["active_training_processes"]


def test_req_verify_4292_validation_rejects_schema_drift(tmp_path: Path) -> None:
    """REQ-VERIFY-4292: validator enforces bare gate fields and methodology."""

    cache_root = _cache_root_with_repo(tmp_path)
    gguf_path = tmp_path / "model.gguf"
    gguf_path.write_bytes(b"GGUF")
    artifact = exp.run(
        artifact_path=tmp_path / "valid.json",
        scorer_path=tmp_path / "scorer.pkl",
        pr_binary_path=_binary(tmp_path),
        cache_root=cache_root,
        resolve_gguf_fn=lambda **_: str(gguf_path),
        vocab_loader_fn=lambda _path, _probe: _loader_result(),
        process_rows_fn=lambda: [],
        energy_prior_fn=lambda **_: _energy_prior(),
        reasoning_items_fn=_reasoning_items,
        minimum_duration_s=0.0,
    )

    corruptions = [
        ("missing required fields", lambda a: a.pop("partial_state_scorer_built")),
        ("honest_verdict", lambda a: a.update({"honest_verdict": ""})),
        ("partial_state_scorer_built", lambda a: a.update({"partial_state_scorer_built": "true"})),
        ("partial_state_leak_free", lambda a: a.update({"partial_state_leak_free": 1})),
        ("partial_state_auroc", lambda a: a.update({"partial_state_auroc": "0.7"})),
        ("leak_ablation_auroc", lambda a: a.update({"leak_ablation_auroc": None})),
        ("verifier_is_oracle", lambda a: a.update({"verifier_is_oracle": True})),
        ("preconditions_checked", lambda a: a.update({"preconditions_checked": []})),
        ("field_principles", lambda a: a.update({"field_principles": {}})),
        ("spec_refs", lambda a: a.update({"spec_refs": []})),
        ("built scorer requires", lambda a: a.update({"partial_state_scorer_built": True, "partial_state_auroc": 0.6})),
        ("leak-free scorer requires", lambda a: a.update({"partial_state_leak_free": True, "leak_ablation_auroc": 0.6})),
    ]
    for message, mutate in corruptions:
        broken = json.loads(json.dumps(artifact))
        mutate(broken)
        with pytest.raises(ValueError, match=message):
            exp.validate_artifact(broken)


def test_req_verify_4292_scorer_edges_and_error_paths(tmp_path: Path) -> None:
    """REQ-VERIFY-4292: scorer helpers reject malformed partial-state inputs."""

    with pytest.raises(ValueError, match="canvas_len"):
        ByteCanvasEncoder(canvas_len=0)
    encoder = ByteCanvasEncoder(canvas_len=32, mask_token_id=exp.MASK_TOKEN_ID)
    with pytest.raises(ValueError, match="visible_fraction"):
        encoder.encode("bad", visible_fraction=1.5)
    canvas, _answers = encoder.encode("A <<1+1=2>>2", visible_fraction=1.0)
    assert encoder.decode_visible([*canvas, 9999999]).endswith(" token_9999999 ")

    assert build_partial_state_records(
        [{"question_id": "empty", "step_text": "", "label": "correct"}], encoder=encoder
    ) == []
    assert build_partial_state_records(
        [{"question_id": "bad", "step_text": "x", "label": "maybe"}], encoder=encoder
    ) == []

    scorer = PartialStateDiffusionScorer(random_seed=4292)
    with pytest.raises(ValueError, match="not fitted"):
        scorer.score_partial_state(canvas, 0)
    with pytest.raises(ValueError, match="at least one"):
        scorer.fit([])
    one_class = build_partial_state_records(
        [{"question_id": "p", "step_text": "verified <<1+1=2>>2", "label": "correct"}],
        encoder=encoder,
    )
    with pytest.raises(ValueError, match="both labels"):
        scorer.fit(one_class)

    wrong_pickle = tmp_path / "wrong.pkl"
    with wrong_pickle.open("wb") as handle:
        pickle.dump({"not": "a scorer"}, handle)
    with pytest.raises(TypeError, match="PartialStateDiffusionScorer"):
        PartialStateDiffusionScorer.load(wrong_pickle)

    assert find_answer_spans("no answer here") == ()
    assert find_answer_spans("boxed{1} and \\boxed{2}")
    records = build_partial_state_records(
        [
            {
                "question_id": "p",
                "step_text": "verified supported <<1+1=2>>2",
                "label": "correct",
            },
            {
                "question_id": "n",
                "step_text": "unsupported contradiction boxed{9}",
                "label": "incorrect",
            },
        ],
        encoder=encoder,
    )
    fitted = PartialStateDiffusionScorer(random_seed=4292, max_features=64).fit(records)
    with pytest.raises(ValueError, match="at least one held-out"):
        partial_state_auroc(fitted, [])
    with pytest.raises(ValueError, match="both positive and negative"):
        partial_state_auroc(fitted, records[:1])
    with pytest.raises(ValueError, match="both positive and negative"):
        _rank_auroc([0.1, 0.2], [True, True])
    with pytest.raises(ValueError, match="heldout_fraction"):
        split_items_task_disjoint(_reasoning_items(), heldout_fraction=1.0)
    with pytest.raises(ValueError, match="empty train or heldout"):
        split_items_task_disjoint([], heldout_fraction=0.25)


def test_req_verify_4292_corpus_loader_and_blocked_eval_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-VERIFY-4292: loader errors and non-green verdict branches are explicit."""

    list_corpus = tmp_path / "list.json"
    list_corpus.write_text(
        json.dumps([{"question_id": "q", "step_text": "verified <<1+1=2>>2", "label": "correct"}]),
        encoding="utf-8",
    )
    assert exp.load_reasoning_items(list_corpus)[0]["question_id"] == "q"

    bad_corpus = tmp_path / "bad.json"
    bad_corpus.write_text(json.dumps({"items": "not-a-list"}), encoding="utf-8")
    with pytest.raises(ValueError, match="list"):
        exp.load_reasoning_items(bad_corpus)

    empty_corpus = tmp_path / "empty.json"
    empty_corpus.write_text(json.dumps({"items": [{"label": "correct"}]}), encoding="utf-8")
    with pytest.raises(ValueError, match="no labeled"):
        exp.load_reasoning_items(empty_corpus)

    scalar_corpus = tmp_path / "scalar.json"
    scalar_corpus.write_text(json.dumps(7), encoding="utf-8")
    with pytest.raises(ValueError, match="no labeled"):
        exp.load_reasoning_items(scalar_corpus)

    cache_root = _cache_root_with_repo(tmp_path / "run")
    gguf_path = tmp_path / "run" / "model.gguf"
    gguf_path.write_bytes(b"GGUF")
    blocked = exp.run(
        artifact_path=tmp_path / "eval-blocked.json",
        scorer_path=tmp_path / "eval-blocked.pkl",
        pr_binary_path=_binary(tmp_path / "run"),
        cache_root=cache_root,
        resolve_gguf_fn=lambda **_: str(gguf_path),
        vocab_loader_fn=lambda _path, _probe: _loader_result(),
        process_rows_fn=lambda: [],
        energy_prior_fn=lambda **_: {"status": "blocked_pr_binary_eval_failed", "eval_rc": 2},
        reasoning_items_fn=_reasoning_items,
        minimum_duration_s=0.0,
    )
    assert blocked["honest_verdict"] == "blocked_pr_binary_eval_failed"

    def run_with_eval(eval_result: dict[str, object], name: str) -> dict[str, object]:
        monkeypatch.setattr(exp, "train_evaluate_and_save", lambda **_: eval_result)
        return exp.run(
            artifact_path=tmp_path / f"{name}.json",
            scorer_path=tmp_path / f"{name}.pkl",
            pr_binary_path=_binary(tmp_path),
            cache_root=cache_root,
            resolve_gguf_fn=lambda **_: str(gguf_path),
            vocab_loader_fn=lambda _path, _probe: _loader_result(),
            process_rows_fn=lambda: [],
            energy_prior_fn=lambda **_: _energy_prior(),
            reasoning_items_fn=_reasoning_items,
            minimum_duration_s=0.0,
        )

    leaky = run_with_eval(
        {
            "scorer_loadable": True,
            "partial_state_auroc": 0.8,
            "leak_ablation_auroc": 0.5,
            "heldout_n": 6,
            "leak_audit": {"answer_masked_cells": 3, "leak_free": False},
        },
        "leaky",
    )
    assert leaky["honest_verdict"] == "complete: partial_state_diffusion_scorer_built_but_leaky"
    assert leaky["partial_state_leak_free"] is False

    weak = run_with_eval(
        {
            "scorer_loadable": True,
            "partial_state_auroc": 0.5,
            "leak_ablation_auroc": 0.8,
            "heldout_n": 6,
            "leak_audit": {"answer_masked_cells": 3, "leak_free": True},
        },
        "weak",
    )
    assert weak["honest_verdict"].endswith("cannot_build_non_degenerate_signal")

    sleeps: list[float] = []
    monkeypatch.setattr(exp.time, "sleep", lambda seconds: sleeps.append(seconds))
    exp._maybe_sleep_for_live_floor(exp.time.perf_counter(), 1.0)
    assert sleeps and sleeps[0] > 0.0
