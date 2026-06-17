"""Tests for Exp 4337 leak-robust partial-state scorer build.

REQ-VERIFY-4337 / SCENARIO-VERIFY-4337: the runner must build a
timestep-conditioned reward head on noisy answer-masked states and prove, on
two corpora, that answer identity is not recoverable while process-quality
ranking remains non-degenerate.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_4337_leak_robust_partial_state_scorer_build as exp
from carnot.verify.dina_lrm_partial_state_scorer import (
    ANSWER_RECOVERY_CEILING,
    PROCESS_RANKING_FLOOR,
    DinaLRMCanvasEncoder,
    DinaLRMPartialStateScorer,
    _rank_auroc,
    build_dina_lrm_records,
    masked_answer_recovery_auroc,
    process_ranking_auroc,
    split_corpus_items,
)


class TinyTokenizer:
    vocab = {"<unk>": 0, "4": 4, "valid": 20, "invalid": 21}

    def tokenize(self, data: bytes) -> list[int]:
        text = data.decode("utf-8", errors="replace")
        return [self.vocab.get(piece, 0) for piece in text.split()] or [0]

    def detokenize(self, token_ids: list[int]) -> bytes:
        reverse = {value: key for key, value in self.vocab.items()}
        return " ".join(reverse.get(int(token_id), "<unk>") for token_id in token_ids).encode()


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


def _reasoning_items(prefix: str, *, n_per_label: int = 48) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for index in range(n_per_label):
        a = index + 2
        b = index % 7 + 3
        rows.append(
            {
                "question_id": f"{prefix}_correct_{index}",
                "corpus_item_id": f"{prefix}_c_{index}",
                "label": "correct",
                "step_text": (
                    "The verified derivation cites the premise, preserves units, and remains "
                    f"coherent. The checked answer is <<{a}+{b}={a + b}>>{a + b}."
                ),
            }
        )
        rows.append(
            {
                "question_id": f"{prefix}_incorrect_{index}",
                "corpus_item_id": f"{prefix}_i_{index}",
                "label": "incorrect",
                "step_text": (
                    "The unsupported shortcut contradicts the premise, changes units, and is "
                    f"invalid. The hidden answer is <<{a}+{b}={a + b + 5}>>{a + b + 5}."
                ),
            }
        )
    return rows


def _corpora() -> list[dict[str, Any]]:
    return [
        {"name": "corpus_one", "path": "corpus_one.json", "items": _reasoning_items("one")},
        {"name": "corpus_two", "path": "corpus_two.json", "items": _reasoning_items("two")},
    ]


def _canvas_smoke(**kwargs: object) -> dict[str, object]:
    return {
        "corpus_name": str(kwargs["corpus_name"]),
        "status": "extracted",
        "eval_rc": 0,
        "score_shape": [exp.CANVAS_LEN, exp.VOCAB_SIZE],
        "score_finite_sample": True,
        "logits_file_size_bytes": exp.CANVAS_LEN * exp.VOCAB_SIZE * 4,
        "expected_logits_file_size_bytes": exp.CANVAS_LEN * exp.VOCAB_SIZE * 4,
        "prompt_ids_count": 3,
        "canvas_non_mask_count": 12,
        "noise_level": 0.3,
    }


def _clean_adversarial_verify(_path: Path) -> dict[str, object]:
    return {"status": "clean", "critical_flags": [], "warn_flags": [], "returncode": 0}


def _common_run_kwargs(tmp_path: Path) -> dict[str, object]:
    cache_root = _cache_root_with_repo(tmp_path)
    gguf_path = tmp_path / "diffusiongemma-Q4_K_M.gguf"
    gguf_path.write_bytes(b"GGUF")
    return {
        "pr_binary_path": _binary(tmp_path),
        "cache_root": cache_root,
        "resolve_gguf_fn": lambda **_: str(gguf_path),
        "vocab_loader_fn": lambda _path, _probe: _loader_result(),
        "process_rows_fn": lambda: [],
        "corpora_loader_fn": _corpora,
        "canvas_smoke_fn": _canvas_smoke,
        "adversarial_verify_fn": _clean_adversarial_verify,
        "minimum_duration_s": 0.0,
    }


def test_req_verify_4337_spec_declares_dina_lrm_contract() -> None:
    """REQ-VERIFY-4337: OpenSpec declares the DiNa-LRM scorer contract."""

    spec = Path("openspec/capabilities/verification/spec.md").read_text(encoding="utf-8")
    for marker in (
        "REQ-VERIFY-4337",
        "SCENARIO-VERIFY-4337",
        "experiment_4337_leak_robust_partial_state_scorer_build.py",
        "score_partial_state(canvas_ids, step) -> energy",
        "scorer_leak_audit_passed",
        "masked_answer_recovery_auroc",
        "process_ranking_auroc",
        "blocked_second_corpus_unavailable",
        "verifier_is_oracle=false",
    ):
        assert marker in spec


def test_req_verify_4337_scorer_masks_answers_and_ranks_process_quality(
    tmp_path: Path,
) -> None:
    """REQ-VERIFY-4337: noisy masked states retain process signal without answer leak."""

    items = _reasoning_items("unit", n_per_label=64)
    train_items, heldout_items = split_corpus_items(items, heldout_fraction=0.25, seed=4337)
    encoder = DinaLRMCanvasEncoder(canvas_len=180, mask_token_id=exp.MASK_TOKEN_ID)
    train_records = build_dina_lrm_records(train_items, corpus_name="unit", encoder=encoder)
    heldout_records = build_dina_lrm_records(heldout_items, corpus_name="unit", encoder=encoder)

    scorer = DinaLRMPartialStateScorer(random_seed=4337, max_features=1024)
    scorer.fit(train_records)
    process_auroc = process_ranking_auroc(scorer, heldout_records)
    answer_auroc = masked_answer_recovery_auroc(scorer, heldout_records)

    assert process_auroc > PROCESS_RANKING_FLOOR
    assert answer_auroc <= ANSWER_RECOVERY_CEILING
    assert all(
        record.canvas_ids[index] == exp.MASK_TOKEN_ID
        for record in heldout_records
        for index in record.answer_cell_indices
    )
    assert encoder.decode_visible(heldout_records[0].canvas_ids)
    assert (
        scorer.score_partial_state(heldout_records[0].canvas_ids, heldout_records[0].timestep) > 0.0
    )

    scorer_path = tmp_path / "dina.pkl"
    scorer.save(scorer_path)
    loaded = DinaLRMPartialStateScorer.load(scorer_path)
    assert loaded.predict_correct_proba(
        heldout_records[0].canvas_ids,
        heldout_records[0].timestep,
        noise_level=heldout_records[0].noise_level,
    ) == pytest.approx(
        scorer.predict_correct_proba(
            heldout_records[0].canvas_ids,
            heldout_records[0].timestep,
            noise_level=heldout_records[0].noise_level,
        )
    )


def test_scenario_4337_missing_pr_binary_blocks_before_corpora(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-4337: missing PR binary stops before corpus loading."""

    calls: list[str] = []

    def fail_resolve(**_: object) -> str:
        calls.append("resolve")
        raise AssertionError("GGUF cache should not be inspected without PR binary")

    artifact = exp.run(
        artifact_path=tmp_path / "blocked.json",
        scorer_path=tmp_path / "unused.pkl",
        pr_binary_path=tmp_path / "missing-binary",
        cache_root=tmp_path,
        resolve_gguf_fn=fail_resolve,
        corpora_loader_fn=pytest.fail,
        canvas_smoke_fn=pytest.fail,
        minimum_duration_s=0.0,
    )

    assert calls == []
    assert artifact["honest_verdict"] == "blocked_pr_binary"
    assert artifact["scorer_leak_audit_passed"] is False
    assert artifact["preconditions_checked"][0]["resource"] == "pr_binary"
    assert artifact["preconditions_checked"][1]["skipped"] is True


def test_scenario_4337_second_corpus_unavailable_blocks_before_canvas(
    tmp_path: Path,
) -> None:
    """SCENARIO-VERIFY-4337: fewer than two corpora is terminal and honest."""

    common = _common_run_kwargs(tmp_path)
    common["corpora_loader_fn"] = lambda: [
        {"name": "only", "path": "only.json", "items": _reasoning_items("only")}
    ]
    common["canvas_smoke_fn"] = pytest.fail
    artifact = exp.run(
        artifact_path=tmp_path / "blocked-corpus.json",
        scorer_path=tmp_path / "unused.pkl",
        **common,
    )

    assert artifact["honest_verdict"] == "blocked_second_corpus_unavailable"
    assert artifact["scorer_leak_audit_passed"] is False
    assert (
        artifact["preconditions_checked"][-1]["resource"] == "two_oracle_distinct_reasoning_corpora"
    )
    assert artifact["preconditions_checked"][-1]["ok"] is False

    raised_common = _common_run_kwargs(tmp_path)
    raised_common["corpora_loader_fn"] = lambda: (_ for _ in ()).throw(RuntimeError("no corpus"))
    raised_common["canvas_smoke_fn"] = pytest.fail
    raised = exp.run(
        artifact_path=tmp_path / "blocked-corpus-error.json",
        scorer_path=tmp_path / "unused-error.pkl",
        **raised_common,
    )
    assert raised["honest_verdict"] == "blocked_second_corpus_unavailable"
    assert "RuntimeError" in raised["preconditions_checked"][-1]["error"]


def test_scenario_4337_complete_path_writes_loadable_clean_artifact(
    tmp_path: Path,
) -> None:
    """SCENARIO-VERIFY-4337: complete run persists scorer and clean audit fields."""

    artifact_path = tmp_path / "artifact.json"
    scorer_path = tmp_path / "dina_lrm_scorer.pkl"
    artifact = exp.run(
        artifact_path=artifact_path,
        scorer_path=scorer_path,
        **_common_run_kwargs(tmp_path),
    )

    exp.validate_artifact(artifact)
    assert json.loads(artifact_path.read_text(encoding="utf-8")) == artifact
    assert artifact["honest_verdict"] == "complete: leak_robust_partial_state_scorer_built"
    assert artifact["scorer_leak_audit_passed"] is True
    assert artifact["masked_answer_recovery_auroc"] <= ANSWER_RECOVERY_CEILING
    assert artifact["process_ranking_auroc"] > PROCESS_RANKING_FLOOR
    assert artifact["verifier_is_oracle"] is False
    assert artifact["adversarial_verify"]["status"] == "clean"
    assert artifact["model_specs"]["reward_head"]["timestep_conditioning"]
    assert set(artifact["per_corpus_audit"]) == {"corpus_one", "corpus_two"}
    assert len(artifact["canvas_generation"]) == 2
    assert Path(artifact["scorer_module_path"]).exists()
    assert DinaLRMPartialStateScorer.load(Path(artifact["scorer_module_path"])).is_fitted
    assert artifact_path.with_suffix(".checkpoint.json").exists()


def test_req_verify_4337_validation_and_blocked_eval_guards(tmp_path: Path) -> None:
    """REQ-VERIFY-4337: validators enforce bare gates and blocked eval is honest."""

    common = _common_run_kwargs(tmp_path)
    common["canvas_smoke_fn"] = lambda **kwargs: {
        **_canvas_smoke(**kwargs),
        "status": "blocked_pr_binary_eval_failed",
    }
    blocked_eval = exp.run(
        artifact_path=tmp_path / "blocked-eval.json",
        scorer_path=tmp_path / "unused.pkl",
        **common,
    )
    assert blocked_eval["honest_verdict"] == "blocked_pr_binary_eval_failed"
    assert blocked_eval["scorer_leak_audit_passed"] is False

    preconditions = {
        "ordered_checks": [
            {"resource": "pr_binary", "ok": True, "path": str(tmp_path / "bin")},
            {"resource": "diffusiongemma_cache", "ok": True, "gguf_path": str(tmp_path / "m.gguf")},
            {"resource": "trm_training_stand_down", "ok": True},
            {"resource": "gguf_vocab_loader", "ok": True, "backend": "test"},
            {"resource": "two_oracle_distinct_reasoning_corpora", "ok": True},
        ],
        "corpora": _corpora(),
    }
    artifact = exp.build_artifact(
        honest_verdict="complete: leak_robust_partial_state_scorer_built",
        preconditions=preconditions,
        duration_s=61.0,
        scorer_path=tmp_path / "scorer.pkl",
        canvas_smokes=[_canvas_smoke(corpus_name="corpus_one")],
        eval_result={
            "scorer_leak_audit_passed": True,
            "masked_answer_recovery_auroc": 0.52,
            "process_ranking_auroc": 0.82,
            "scorer_loadable": True,
            "train_records": 10,
            "per_corpus_audit": {},
        },
        adversarial_verify={"status": "clean", "critical_flags": []},
    )
    exp.validate_artifact(artifact)

    corruptions = [
        ("missing required fields", lambda a: a.pop("honest_verdict")),
        ("honest_verdict", lambda a: a.update({"honest_verdict": ""})),
        ("scorer_leak_audit_passed", lambda a: a.update({"scorer_leak_audit_passed": 1})),
        (
            "masked_answer_recovery_auroc",
            lambda a: a.update({"masked_answer_recovery_auroc": "0.5"}),
        ),
        ("process_ranking_auroc", lambda a: a.update({"process_ranking_auroc": "0.7"})),
        ("verifier_is_oracle", lambda a: a.update({"verifier_is_oracle": True})),
        ("preconditions_checked", lambda a: a.update({"preconditions_checked": []})),
        ("field_principles", lambda a: a.update({"field_principles": {}})),
        ("spec_refs", lambda a: a.update({"spec_refs": []})),
        ("adversarial_verify", lambda a: a.update({"adversarial_verify": {}})),
        (
            "passed leak audit",
            lambda a: a.update({"masked_answer_recovery_auroc": 0.9}),
        ),
    ]
    for message, mutate in corruptions:
        broken = json.loads(json.dumps(artifact))
        mutate(broken)
        with pytest.raises(ValueError, match=message):
            exp.validate_artifact(broken)

    with pytest.raises(ValueError, match="visible_fractions"):
        build_dina_lrm_records(
            _reasoning_items("bad"),
            corpus_name="bad",
            visible_fractions=(0.5,),
            noise_levels=(0.1, 0.2),
        )


def test_req_verify_4337_edge_guards_and_helpers(tmp_path: Path) -> None:
    """REQ-VERIFY-4337: helper guards fail closed and stay deterministic."""

    loaded_corpora = exp.load_required_corpora(
        corpus1_path=tmp_path / "one.json",
        corpus2_path=tmp_path / "two.json",
        corpus1_loader=lambda: _reasoning_items("one", n_per_label=2),
        corpus2_loader=lambda: _reasoning_items("two", n_per_label=2),
    )
    assert [corpus["name"] for corpus in loaded_corpora] == [
        "in_distribution_error_corpus_v1",
        "step_error_balanced_v2",
    ]
    assert exp._scrub_answer_text("No answer span") == "No answer span"
    assert "<answer_masked>" in exp._scrub_answer_text("Final <<1+1=2>>2.")
    assert exp._stable_seed(1, "a") == exp._stable_seed(1, "a")
    exp._checkpoint(None, {"ignored": True})

    sleeps: list[float] = []
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(exp.time, "sleep", lambda seconds: sleeps.append(float(seconds)))
    try:
        exp._maybe_sleep_for_live_floor(exp.time.perf_counter(), 0.5)
    finally:
        monkeypatch.undo()
    assert sleeps and sleeps[0] > 0.0

    with pytest.raises(ValueError, match="canvas_len"):
        DinaLRMCanvasEncoder(canvas_len=0)
    encoder = DinaLRMCanvasEncoder(canvas_len=64, mask_token_id=exp.MASK_TOKEN_ID)
    with pytest.raises(ValueError, match="visible_fraction"):
        encoder.encode("x", visible_fraction=-0.1, timestep=0, noise_level=0.0, seed=1)
    with pytest.raises(ValueError, match="noise_level"):
        encoder.encode("x", visible_fraction=0.5, timestep=0, noise_level=1.1, seed=1)

    scorer = DinaLRMPartialStateScorer(random_seed=4337, max_features=64)
    with pytest.raises(ValueError, match="not fitted"):
        scorer.predict_correct_proba([exp.MASK_TOKEN_ID], 0)
    with pytest.raises(ValueError, match="at least one"):
        scorer.fit([])
    correct_only = [
        item for item in _reasoning_items("one-label", n_per_label=4) if item["label"] == "correct"
    ]
    one_label_records = build_dina_lrm_records(
        correct_only,
        corpus_name="one-label",
        encoder=encoder,
    )
    with pytest.raises(ValueError, match="both process-quality labels"):
        scorer.fit(one_label_records)

    mixed_records = build_dina_lrm_records(
        [
            {"question_id": "skip_missing_text", "label": "correct"},
            {"question_id": "skip_bad_label", "label": "maybe", "step_text": "text"},
            *_reasoning_items("mixed", n_per_label=4),
            {
                "question_id": "no_answer",
                "label": "correct",
                "step_text": "A verified step with no explicit answer span.",
            },
        ],
        corpus_name="mixed",
        encoder=encoder,
    )
    fitted = DinaLRMPartialStateScorer(random_seed=4337, max_features=256).fit(mixed_records)
    assert fitted.predict_correct_proba(mixed_records[0].canvas_ids, mixed_records[0].timestep)
    assert fitted._infer_noise_level([]) == 1.0
    assert masked_answer_recovery_auroc(fitted, mixed_records[:3]) == 0.5
    with pytest.raises(ValueError, match="held-out record"):
        process_ranking_auroc(fitted, [])
    with pytest.raises(ValueError, match="both labels"):
        process_ranking_auroc(fitted, [record for record in mixed_records if record.label][:3])
    assert _rank_auroc([0.5, 0.5, 0.7, 0.2], [True, False, True, False]) > 0.5
    with pytest.raises(ValueError, match="AUROC requires"):
        _rank_auroc([0.1, 0.2], [True, True])

    wrong_pickle = tmp_path / "wrong.pkl"
    with wrong_pickle.open("wb") as handle:
        pickle.dump({"not": "a scorer"}, handle)
    with pytest.raises(TypeError, match="DinaLRMPartialStateScorer"):
        DinaLRMPartialStateScorer.load(wrong_pickle)
