"""Tests for Exp 5031 (PHASE D1) trained LoRA-EBM holistic-quality MuSR scorer.

Spec refs: REQ-VERIFY-5031, SCENARIO-VERIFY-5031.

The live GPU training/scoring paths (the B3 ``moat_trainer`` calls) are injected
as deterministic fakes, so every pure-Python line — corpus build, eval wiring,
precondition gating, the honest verdict logic, and the schema check — is
exercised without a GPU.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_5031_lora_ebm_scorer_musr_v3 as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "verification" / "spec.md"


class Clock:
    """Deterministic clock so tests can drive the >60s training gate exactly."""

    def __init__(self, values: list[float]) -> None:
        self.values = values
        self.index = 0

    def __call__(self) -> float:
        value = self.values[min(self.index, len(self.values) - 1)]
        self.index += 1
        return value


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")


def _write_checkpoint(path: Path, *, gold: str, answers: list[Any]) -> None:
    _write_json(path, {"q": int(path.stem[1:]), "gold": gold, "answers": answers})


def _write_fover(path: Path, *, correct: int = 8, incorrect: int = 3) -> None:
    records = [
        {"question_id": f"q_{i}", "step_text": f"correct step {i}", "label": "correct"}
        for i in range(correct)
    ] + [
        {"question_id": f"q_{i}", "step_text": f"incorrect step {i}", "label": "incorrect"}
        for i in range(incorrect)
    ]
    _write_json(path, records)


def _setup_root(
    tmp_path: Path,
    *,
    n_questions: int = 8,
    answers: list[Any] | None = None,
    gold: str = "GOLD",
    smoke_passed: bool = True,
    fover_correct: int = 8,
    fover_incorrect: int = 3,
    with_fover: bool = True,
) -> Path:
    """Build a self-contained repo root: B3 artifact + MuSR checkpoints + FoVer."""
    root = tmp_path / "root"
    answers = answers if answers is not None else ["WRONG", "WRONG", "GOLD"]
    _write_json(
        root / mod.B3_ARTIFACT_RELATIVE_PATH,
        {"smoke_passed": smoke_passed, "base_used": "Qwen/Qwen3.5-2B"},
    )
    ckdir = root / mod.MUSR_CHECKPOINT_RELATIVE_DIR
    for i in range(n_questions):
        _write_checkpoint(ckdir / f"q{i:04d}.json", gold=gold, answers=answers)
    if with_fover:
        _write_fover(
            root / mod.FOVER_RELATIVE_PATH, correct=fover_correct, incorrect=fover_incorrect
        )
    return root


def _gold_low_score_fn(checkpoint: Any, texts: list[str]) -> list[float]:
    """A 'perfect' trained scorer: GOLD-answer texts get the lowest energy."""
    return [0.0 if "Candidate answer: GOLD" in text else 1.0 for text in texts]


def _constant_score_fn(checkpoint: Any, texts: list[str]) -> list[float]:
    """A non-discriminating scorer: every candidate ties (a clean null)."""
    return [0.5 for _ in texts]


def _fake_trainer(n_pairs_seen: dict[str, int]) -> Any:
    def trainer(pairs: Any, *, base: Any, out_dir: Any, config: Any) -> dict[str, Any]:
        n_pairs_seen["n"] = len(list(pairs))
        return {
            "train_loss": 0.1234,
            "n_pairs": n_pairs_seen["n"],
            "base_used": base[0],
            "checkpoint_dir": str(Path(out_dir) / "epoch_1"),
            "reproducibility_checksum": "sha256:fake",
            "model_specs": {"base_model": base[0], "adapter": "LoRA"},
        }

    return trainer


def _fake_resolver() -> tuple[str, str]:
    return ("Qwen/Qwen3.5-2B", "/fake/snapshot")


# --------------------------------------------------------------------------- #
# Spec presence (REQ-/SCENARIO traceability).
# --------------------------------------------------------------------------- #
def test_spec_has_req_and_scenario() -> None:
    """REQ-VERIFY-5031 + SCENARIO-VERIFY-5031 are anchored in the spec."""
    text = SPEC_PATH.read_text(encoding="utf-8")
    assert "REQ-VERIFY-5031" in text
    assert "SCENARIO-VERIFY-5031" in text


# --------------------------------------------------------------------------- #
# Small helpers.
# --------------------------------------------------------------------------- #
def test_precondition_check_as_dict_with_and_without_path() -> None:
    with_path = mod.PreconditionCheck("cuda", True, "ok", "/p").as_dict()
    assert with_path == {"resource": "cuda", "available": True, "detail": "ok", "path": "/p"}
    without = mod.PreconditionCheck("cuda", False, "no").as_dict()
    assert "path" not in without and without["available"] is False


def test_training_config_payload_roundtrip() -> None:
    payload = mod.TrainingConfig().lora_config_payload()
    assert payload["r"] == 8 and payload["device_index"] == 0 and payload["epochs"] == 1


def test_read_json_handles_bad_path(tmp_path: Path) -> None:
    assert mod._read_json(tmp_path / "missing.json") is None
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    assert mod._read_json(bad) is None


def test_b3_module_importable_real() -> None:
    ok, detail = mod.b3_module_importable()
    assert ok is True and "moat_trainer" in detail


def test_read_b3_smoke_passed_variants(tmp_path: Path) -> None:
    passed = tmp_path / "p.json"
    _write_json(passed, {"smoke_passed": True, "base_used": "Qwen/Qwen3.5-2B"})
    assert mod.read_b3_smoke_passed(passed)[0] is True
    failed = tmp_path / "f.json"
    _write_json(failed, {"smoke_passed": False})
    assert mod.read_b3_smoke_passed(failed)[0] is False
    assert mod.read_b3_smoke_passed(tmp_path / "missing.json")[0] is False


# --------------------------------------------------------------------------- #
# Corpus construction.
# --------------------------------------------------------------------------- #
def test_load_fover_pairs_builds_and_cycles(tmp_path: Path) -> None:
    path = tmp_path / "fover.json"
    _write_fover(path, correct=5, incorrect=2)
    pairs = mod.load_fover_contrastive_pairs(path, max_pairs=6)
    assert len(pairs) == 6
    assert all(g.startswith("correct") and b.startswith("incorrect") for g, b in pairs)


def test_load_fover_pairs_edge_cases(tmp_path: Path) -> None:
    assert mod.load_fover_contrastive_pairs(tmp_path / "f.json", max_pairs=0) == []
    missing = tmp_path / "missing.json"
    assert mod.load_fover_contrastive_pairs(missing, max_pairs=4) == []
    # Not a list.
    not_list = tmp_path / "nl.json"
    _write_json(not_list, {"x": 1})
    assert mod.load_fover_contrastive_pairs(not_list, max_pairs=4) == []
    # Only correct labels -> no contrastive signal.
    only_correct = tmp_path / "oc.json"
    _write_json(
        only_correct,
        [
            {"step_text": "a", "label": "correct"},
            {"step_text": "", "label": "incorrect"},  # blank dropped
            "not-a-mapping",
        ],
    )
    assert mod.load_fover_contrastive_pairs(only_correct, max_pairs=4) == []


def test_load_fover_pairs_skips_identical_good_bad(tmp_path: Path) -> None:
    path = tmp_path / "same.json"
    _write_json(
        path,
        [
            {"step_text": "same", "label": "correct"},
            {"step_text": "same", "label": "incorrect"},
        ],
    )
    # good == bad on the only available step -> skipped, yields nothing.
    assert mod.load_fover_contrastive_pairs(path, max_pairs=3) == []


def test_musr_candidate_text_answer_first_and_no_context() -> None:
    with_ctx = mod.musr_candidate_text("Ana", "Who?", "A long narrative", context_char_cap=4)
    assert with_ctx.startswith("Candidate answer: Ana")
    assert "Narrative:\nA lo" in with_ctx  # capped to 4 chars
    no_ctx = mod.musr_candidate_text("Ana", "Who?", "")
    assert "Narrative" not in no_ctx


def test_load_musr_eval_rows_with_and_without_narratives(tmp_path: Path) -> None:
    ckdir = tmp_path / "ck"
    _write_checkpoint(ckdir / "q0000.json", gold="GOLD", answers=["GOLD", "WRONG", None, ""])
    _write_checkpoint(ckdir / "q0001.json", gold="X", answers=["X", "Y"])
    narratives = [{"question": "Q0", "context": "C0"}]
    rows = mod.load_musr_eval_rows(ckdir, narratives=narratives, limit=10)
    assert len(rows) == 2
    # q0 has 2 valid candidates (None + "" dropped) and context attached.
    assert len(rows[0]["candidates"]) == 2
    assert "C0" in rows[0]["candidates"][0]["text"]
    # q1 has no narrative (index out of range) -> answer-only text.
    assert "Narrative" not in rows[1]["candidates"][0]["text"]


def test_load_musr_eval_rows_missing_dir_and_malformed(tmp_path: Path) -> None:
    assert mod.load_musr_eval_rows(tmp_path / "nope") == []
    ckdir = tmp_path / "ck"
    ckdir.mkdir()
    (ckdir / "q0000.json").write_text("{bad", encoding="utf-8")  # unreadable -> skipped
    _write_json(ckdir / "q0001.json", {"gold": "G"})  # no answers list -> skipped
    _write_checkpoint(ckdir / "q0002.json", gold="G", answers=[None, ""])  # no valid cands
    assert mod.load_musr_eval_rows(ckdir) == []


def test_build_musr_training_pairs_dedups_and_caps(tmp_path: Path) -> None:
    rows = [
        {
            "gold": "GOLD",
            "candidates": [
                {"answer": "GOLD", "text": "Candidate answer: GOLD a"},
                {"answer": "GOLD", "text": "Candidate answer: GOLD dup"},  # dedup by answer
                {"answer": "WRONG", "text": "Candidate answer: WRONG"},
                {"answer": "", "text": ""},  # skipped
            ],
        }
    ]
    pairs = mod.build_musr_training_pairs(rows, max_pairs=10)
    assert len(pairs) == 1  # one unique (GOLD, WRONG) pair
    assert pairs[0][0].startswith("Candidate answer: GOLD")
    assert mod.build_musr_training_pairs(rows, max_pairs=0) == []


def test_build_musr_training_pairs_respects_cap() -> None:
    rows = [
        {
            "gold": "G",
            "candidates": [
                {"answer": "G", "text": "g"},
                {"answer": "A", "text": "a"},
                {"answer": "B", "text": "b"},
            ],
        }
    ]
    assert len(mod.build_musr_training_pairs(rows, max_pairs=1)) == 1


def test_build_contrastive_corpus_combines(tmp_path: Path) -> None:
    fover = tmp_path / "fover.json"
    _write_fover(fover, correct=10, incorrect=4)
    rows = [
        {"gold": "G", "candidates": [{"answer": "G", "text": "g"}, {"answer": "B", "text": "b"}]}
    ]
    corpus = mod.build_contrastive_corpus(fover, rows, max_pairs=5, fover_fraction=0.6)
    assert 1 <= len(corpus) <= 5
    # MuSR pair present plus some FoVer pairs.
    assert ("g", "b") in corpus
    assert any(g.startswith("correct") for g, _ in corpus)


# --------------------------------------------------------------------------- #
# Eval scoring lookup (oracle-distinct).
# --------------------------------------------------------------------------- #
def test_precompute_and_lookup_scorer() -> None:
    rows = [
        {
            "candidates": [
                {"candidate_id": "0/c0", "text": "t0"},
                {"candidate_id": "0/c1", "text": "t1"},
            ]
        }
    ]
    energies = mod.precompute_candidate_energies(
        "ckpt", rows, score_fn=lambda c, texts: [float(i) for i in range(len(texts))]
    )
    assert energies == {"0/c0": 0.0, "0/c1": 1.0}
    scorer = mod.make_lookup_scorer(energies)
    assert scorer({"candidate_id": "0/c1"}) == 1.0
    assert scorer({"candidate_id": "missing"}) == float("inf")


def test_precompute_empty_and_mismatch() -> None:
    assert (
        mod.precompute_candidate_energies("c", [{"candidates": []}], score_fn=lambda c, t: []) == {}
    )
    with pytest.raises(RuntimeError, match="energies"):
        mod.precompute_candidate_energies(
            "c",
            [{"candidates": [{"candidate_id": "a", "text": "x"}]}],
            score_fn=lambda c, t: [1.0, 2.0],
        )


def test_default_score_fn_builds_callable() -> None:
    fn = mod.default_score_fn(mod.TrainingConfig())
    assert callable(fn)


# --------------------------------------------------------------------------- #
# Preconditions.
# --------------------------------------------------------------------------- #
def _run_preconditions(root: Path, **overrides: Any) -> tuple[list[mod.PreconditionCheck], Any]:
    kwargs: dict[str, Any] = {
        "root": root,
        "b3_artifact_path": root / mod.B3_ARTIFACT_RELATIVE_PATH,
        "cuda_available": lambda: True,
        "b3_importable": lambda: (True, "ok"),
        "base_resolver": _fake_resolver,
        "min_questions": 2,
    }
    kwargs.update(overrides)
    return mod.check_preconditions(**kwargs)


def test_check_preconditions_all_pass(tmp_path: Path) -> None:
    root = _setup_root(tmp_path, n_questions=3)
    checks, base = _run_preconditions(root)
    assert mod.first_missing_resource(checks) is None
    assert base == ("Qwen/Qwen3.5-2B", "/fake/snapshot")


def test_check_preconditions_b3_not_importable(tmp_path: Path) -> None:
    root = _setup_root(tmp_path, n_questions=3)
    checks, base = _run_preconditions(root, b3_importable=lambda: (False, "no module"))
    assert mod.first_missing_resource(checks) == "b3_module"
    assert base is None
    # trainable_base check is recorded as skipped/unavailable.
    assert any(c.resource == "trainable_base_cached" and not c.available for c in checks)


def test_check_preconditions_smoke_not_passed(tmp_path: Path) -> None:
    root = _setup_root(tmp_path, n_questions=3, smoke_passed=False)
    checks, _ = _run_preconditions(root)
    assert mod.first_missing_resource(checks) == "b3_smoke"


def test_check_preconditions_cuda_missing(tmp_path: Path) -> None:
    root = _setup_root(tmp_path, n_questions=3)
    checks, _ = _run_preconditions(root, cuda_available=lambda: False)
    assert mod.first_missing_resource(checks) == "cuda"


def test_check_preconditions_base_resolver_raises(tmp_path: Path) -> None:
    root = _setup_root(tmp_path, n_questions=3)

    def boom() -> tuple[str, str]:
        raise RuntimeError("no_trainable_base")

    checks, base = _run_preconditions(root, base_resolver=boom)
    assert mod.first_missing_resource(checks) == "trainable_base_cached"
    assert base is None


def test_check_preconditions_few_checkpoints(tmp_path: Path) -> None:
    root = _setup_root(tmp_path, n_questions=1)
    checks, _ = _run_preconditions(root, min_questions=5)
    assert mod.first_missing_resource(checks) == "cached_musr_candidates"


def test_check_preconditions_fover_missing(tmp_path: Path) -> None:
    root = _setup_root(tmp_path, n_questions=3, with_fover=False)
    checks, _ = _run_preconditions(root)
    assert mod.first_missing_resource(checks) == "fover_pairs"


# --------------------------------------------------------------------------- #
# Reproducibility + small artifact helpers.
# --------------------------------------------------------------------------- #
def test_reproducibility_checksum_deterministic() -> None:
    cfg = mod.TrainingConfig()
    pairs = [("g", "b"), ("g2", "b2")]
    a = mod.reproducibility_checksum(
        base_used="Qwen/Qwen3.5-2B", config=cfg, pairs=pairs, candidate_source="c", seed=1
    )
    b = mod.reproducibility_checksum(
        base_used="Qwen/Qwen3.5-2B", config=cfg, pairs=pairs, candidate_source="c", seed=1
    )
    assert a == b and a.startswith("sha256:")
    c = mod.reproducibility_checksum(
        base_used="other", config=cfg, pairs=pairs, candidate_source="c", seed=1
    )
    assert c != a


def test_format_delta_and_ci_zero() -> None:
    assert mod._format_delta(0.123) == "plus_0p123"
    assert mod._format_delta(-0.2) == "minus_0p200"
    assert mod._ci_includes_zero([-0.1, 0.2]) is True
    assert mod._ci_includes_zero([0.1, 0.2]) is False
    assert mod._ci_includes_zero([0.1]) is False


def test_read_b1_baseline(tmp_path: Path) -> None:
    root = tmp_path / "root"
    assert mod._read_b1_baseline(root)["available"] is False
    _write_json(
        root / mod.B1_BASELINE_RELATIVE_PATH,
        {"honest_verdict": "complete", "genuine_tuned_sc_accuracy": 0.585, "oracle_at_k": 0.865},
    )
    ref = mod._read_b1_baseline(root)
    assert ref["available"] is True and ref["genuine_tuned_sc_accuracy"] == 0.585


# --------------------------------------------------------------------------- #
# Verdict logic in build_complete_artifact.
# --------------------------------------------------------------------------- #
def _evaluation(
    *,
    verifier_acc: float,
    sc_acc: float,
    delta: float,
    ci95: list[float],
    mcnemar_p: float,
    headroom: bool,
    predictions: list[Any] | None = None,
    n_rows: int = 20,
) -> dict[str, Any]:
    return {
        "n_rows": n_rows,
        "tuned_self_consistency": {"accuracy": sc_acc, "config": {"k": 1, "temperature": "cached"}},
        "oracle_at_k": 0.865,
        "headroom_present": headroom,
        "verifier": {
            "accuracy": verifier_acc,
            "predictions": predictions if predictions is not None else ["GOLD"] * n_rows,
        },
        "verifier_minus_tuned_sc_delta": delta,
        "verifier_minus_tuned_sc_ci95": ci95,
        "mcnemar_p": mcnemar_p,
    }


def _complete(evaluation: dict[str, Any], *, root: Path, train_loss: Any = 0.1) -> dict[str, Any]:
    return mod.build_complete_artifact(
        evaluation=evaluation,
        train_result={"train_loss": train_loss, "n_pairs": 100, "model_specs": {"x": 1}},
        config=mod.TrainingConfig(),
        pairs=[("g", "b")],
        preconditions_checked=[{"resource": "cuda", "available": True, "detail": "ok"}],
        candidate_source="ck",
        checkpoint_path="/ckpt/epoch_1",
        base_used="Qwen/Qwen3.5-2B",
        root=root,
        duration_s=120.0,
    )


def test_complete_artifact_win(tmp_path: Path) -> None:
    ev = _evaluation(
        verifier_acc=0.8, sc_acc=0.585, delta=0.215, ci95=[0.1, 0.3], mcnemar_p=0.001, headroom=True
    )
    art = _complete(ev, root=tmp_path)
    assert art["honest_verdict"].startswith("success_lora_ebm_beats_sc_musr_")
    assert art["scorer_trained"] is True and art["verifier_is_oracle"] is False
    assert art["oracle_distinctness_enforced"] is True
    assert art["degeneracy_guard"]["degeneracy_flag"] is False
    assert mod.artifact_schema_errors(art) == []


def test_complete_artifact_null_ci_includes_zero(tmp_path: Path) -> None:
    ev = _evaluation(
        verifier_acc=0.585,
        sc_acc=0.585,
        delta=0.0,
        ci95=[-0.05, 0.05],
        mcnemar_p=1.0,
        headroom=True,
    )
    art = _complete(ev, root=tmp_path)
    assert art["honest_verdict"].endswith("_ci_incl_0")
    assert mod.artifact_schema_errors(art) == []


def test_complete_artifact_gate_branch(tmp_path: Path) -> None:
    # delta>0 and CI excludes 0, but McNemar not significant -> gate (else) branch.
    ev = _evaluation(
        verifier_acc=0.62,
        sc_acc=0.585,
        delta=0.035,
        ci95=[0.01, 0.06],
        mcnemar_p=0.5,
        headroom=True,
    )
    art = _complete(ev, root=tmp_path)
    assert art["honest_verdict"].endswith("_mcnemar_or_headroom_gate")


def test_complete_artifact_degeneracy_blocks_win(tmp_path: Path) -> None:
    # Strong stats but the selector abstains on >50% -> degenerate, win blocked.
    ev = _evaluation(
        verifier_acc=0.8,
        sc_acc=0.585,
        delta=0.215,
        ci95=[0.1, 0.3],
        mcnemar_p=0.001,
        headroom=True,
        predictions=[None] * 15 + ["GOLD"] * 5,
    )
    art = _complete(ev, root=tmp_path)
    assert art["degeneracy_guard"]["degeneracy_flag"] is True
    assert not art["honest_verdict"].startswith("success_")


def test_complete_artifact_empty_predictions(tmp_path: Path) -> None:
    ev = _evaluation(
        verifier_acc=0.0,
        sc_acc=0.0,
        delta=0.0,
        ci95=[0.0, 0.0],
        mcnemar_p=1.0,
        headroom=True,
        predictions=[],
        n_rows=0,
    )
    art = _complete(ev, root=tmp_path)
    assert art["degeneracy_guard"]["abstain_rate"] == 0.0


def test_complete_artifact_skeleton_downgrades(tmp_path: Path) -> None:
    ev = _evaluation(
        verifier_acc=0.8, sc_acc=0.585, delta=0.215, ci95=[0.1, 0.3], mcnemar_p=0.001, headroom=True
    )
    # train_loss None -> not actually trained -> train_did_not_run.
    art = _complete(ev, root=tmp_path, train_loss=None)
    assert art["honest_verdict"] == "blocked_lora_ebm_train_did_not_run"
    assert art["scorer_trained"] is False


# --------------------------------------------------------------------------- #
# Blocked / train-did-not-run builders.
# --------------------------------------------------------------------------- #
def test_blocked_artifact_fields() -> None:
    art = mod.build_blocked_artifact(
        missing_resource="cuda",
        preconditions_checked=[{"resource": "cuda", "available": False, "detail": "no"}],
        duration_s=0.5,
        error="cuda down",
    )
    assert art["honest_verdict"] == "blocked_cuda"
    assert art["inference_substrate"] == "precondition_check_only"
    assert art["blocked_error"] == "cuda down"
    assert mod.artifact_schema_errors(art) == []


def test_train_did_not_run_artifact() -> None:
    art = mod.build_train_did_not_run_artifact(
        preconditions_checked=[], duration_s=12.0, base_used="Qwen/Qwen3.5-2B", error="boom"
    )
    assert art["honest_verdict"] == "blocked_lora_ebm_train_did_not_run"
    assert art["scorer_trained"] is False and art["blocked_error"] == "boom"


# --------------------------------------------------------------------------- #
# Schema validation.
# --------------------------------------------------------------------------- #
def test_schema_errors_flag_problems(tmp_path: Path) -> None:
    ev = _evaluation(
        verifier_acc=0.8, sc_acc=0.585, delta=0.2, ci95=[0.1, 0.3], mcnemar_p=0.001, headroom=True
    )
    good = _complete(ev, root=tmp_path)
    assert mod.artifact_schema_errors(good) == []

    missing = dict(good)
    del missing["train_loss"]
    assert "train_loss" in mod.artifact_schema_errors(missing)

    bad_specrefs = dict(good, spec_refs=["WRONG"])
    assert "spec_refs" in mod.artifact_schema_errors(bad_specrefs)

    bad_principles = dict(good, field_principles={})
    assert "field_principles" in mod.artifact_schema_errors(bad_principles)

    non_bool = dict(good, scorer_trained="yes")
    assert "scorer_trained" in mod.artifact_schema_errors(non_bool)

    is_oracle = dict(good, verifier_is_oracle=True)
    assert "verifier_is_oracle" in mod.artifact_schema_errors(is_oracle)

    bad_ci = dict(good, paired_ci95=[0.1])
    assert "paired_ci95" in mod.artifact_schema_errors(bad_ci)

    bad_acc = dict(good, trained_scorer_accuracy=2.0)
    assert "trained_scorer_accuracy" in mod.artifact_schema_errors(bad_acc)

    bad_delta = dict(good, delta_vs_tuned_sc="big")
    assert "delta_vs_tuned_sc" in mod.artifact_schema_errors(bad_delta)

    bad_p = dict(good, mcnemar_p=2.0)
    assert "mcnemar_p" in mod.artifact_schema_errors(bad_p)

    bad_pre = dict(good, preconditions_checked={})
    assert "preconditions_checked" in mod.artifact_schema_errors(bad_pre)

    bad_specs = dict(good, model_specs=[])
    assert "model_specs" in mod.artifact_schema_errors(bad_specs)

    bad_verdict = dict(good, honest_verdict="running_thing")
    assert "honest_verdict" in mod.artifact_schema_errors(bad_verdict)


def test_schema_errors_trained_gate(tmp_path: Path) -> None:
    ev = _evaluation(
        verifier_acc=0.8, sc_acc=0.585, delta=0.2, ci95=[0.1, 0.3], mcnemar_p=0.001, headroom=True
    )
    good = _complete(ev, root=tmp_path)
    # scorer_trained True but train_loss None / n_pairs 0 / short duration / no base.
    broken = dict(good, train_loss=None, n_pairs=0, duration_s=10.0, base_used=None)
    errors = mod.artifact_schema_errors(broken)
    for field in ("train_loss", "n_pairs", "duration_s", "base_used"):
        assert field in errors
    # scorer_trained False with a success verdict is contradictory.
    contradiction = dict(good, scorer_trained=False, honest_verdict="success_x")
    assert "scorer_trained" in mod.artifact_schema_errors(contradiction)


# --------------------------------------------------------------------------- #
# Audit glue.
# --------------------------------------------------------------------------- #
def test_audit_clean_and_flag_compaction() -> None:
    assert mod._audit_is_clean({"flagged_count": 0}) is True
    assert mod._audit_is_clean({"flag_count": 2}) is False
    assert mod._audit_is_clean({"flags": [{"x": 1}]}) is False
    assert mod._compact_adversarial_flags({"reports": [{"flags": [{"k": "v"}]}]}) == [{"k": "v"}]


def test_attach_audit_writes_and_marks_clean(tmp_path: Path) -> None:
    artifact = {"honest_verdict": "complete_x"}
    out = tmp_path / "art.json"
    updated = mod.attach_audit(
        artifact,
        artifact_path=out,
        audit_runner=lambda p: {"flags": []},
        summary_runner=lambda p: 0,
    )
    assert updated["adversarial_verify_clean"] is True
    assert updated["summarize_artifact_exit_code"] == 0
    assert out.exists()


def test_precondition_dicts() -> None:
    dicts = mod._precondition_dicts([mod.PreconditionCheck("cuda", True, "ok")])
    assert dicts == [{"resource": "cuda", "available": True, "detail": "ok"}]


# --------------------------------------------------------------------------- #
# Full orchestration (run) with injected GPU fakes.
# --------------------------------------------------------------------------- #
def _run(root: Path, tmp_path: Path, **overrides: Any) -> dict[str, Any]:
    seen: dict[str, int] = {}
    kwargs: dict[str, Any] = {
        "root": root,
        "artifact_path": tmp_path / "out.json",
        "cuda_available": lambda: True,
        "b3_importable": lambda: (True, "ok"),
        "base_resolver": _fake_resolver,
        "trainer": _fake_trainer(seen),
        "score_fn": _gold_low_score_fn,
        "narratives_loader": lambda limit: [
            {"question": "Who?", "context": f"narrative {i}"} for i in range(limit)
        ],
        "audit_runner": lambda p: {"flags": []},
        "summary_runner": lambda p: 0,
        "min_questions": 4,
        "limit": 12,
        "bootstrap_samples": 300,
        "now": Clock([0.0, 120.0]),
    }
    kwargs.update(overrides)
    return mod.run(**kwargs)


def test_run_win_path(tmp_path: Path) -> None:
    root = _setup_root(tmp_path, n_questions=12)
    art = _run(root, tmp_path)
    assert art["honest_verdict"].startswith("success_lora_ebm_beats_sc_musr_")
    assert art["scorer_trained"] is True
    assert art["base_used"] == "Qwen/Qwen3.5-2B"
    assert art["genuine_tuned_sc_accuracy"] == 0.0  # SC majority is WRONG
    assert art["trained_scorer_accuracy"] == 1.0
    assert art["headroom_present"] is True
    assert art["oracle_distinctness_enforced"] is True
    assert art["adversarial_verify_clean"] is True
    assert art["summarize_artifact_exit_code"] == 0
    assert mod.artifact_schema_errors(art) == []
    assert (tmp_path / "out.json").exists()


def test_run_null_path(tmp_path: Path) -> None:
    root = _setup_root(tmp_path, n_questions=12)
    art = _run(root, tmp_path, score_fn=_constant_score_fn)
    assert art["honest_verdict"].endswith("_ci_incl_0")
    assert art["scorer_trained"] is True
    assert art["delta_vs_tuned_sc"] == 0.0
    assert mod.artifact_schema_errors(art) == []


def test_run_no_narratives(tmp_path: Path) -> None:
    root = _setup_root(tmp_path, n_questions=12)
    art = _run(root, tmp_path, narratives_loader=lambda limit: None)
    assert art["scorer_trained"] is True  # still trains/evaluates answer-only


def test_run_blocked_on_cuda(tmp_path: Path) -> None:
    root = _setup_root(tmp_path, n_questions=12)
    art = _run(
        root, tmp_path, cuda_available=lambda: False
    )  # write=True exercises the write branch
    assert art["honest_verdict"] == "blocked_cuda"
    assert art["scorer_trained"] is False
    assert (tmp_path / "out.json").exists()


def test_run_train_did_not_run_short_duration(tmp_path: Path) -> None:
    root = _setup_root(tmp_path, n_questions=12)
    art = _run(root, tmp_path, now=Clock([0.0, 30.0]))  # elapsed 30s < 60s floor
    assert art["honest_verdict"] == "blocked_lora_ebm_train_did_not_run"


def test_run_train_did_not_run_null_loss(tmp_path: Path) -> None:
    root = _setup_root(tmp_path, n_questions=12)

    def null_trainer(pairs: Any, *, base: Any, out_dir: Any, config: Any) -> dict[str, Any]:
        return {"train_loss": None, "n_pairs": 0, "checkpoint_dir": "/x"}

    art = _run(root, tmp_path, trainer=null_trainer)
    assert art["honest_verdict"] == "blocked_lora_ebm_train_did_not_run"


def test_run_oracle_distinctness_violation(tmp_path: Path) -> None:
    root = _setup_root(tmp_path, n_questions=12)

    def peeking_score_fn(checkpoint: Any, texts: list[str]) -> list[float]:
        raise mod.OracleDistinctnessError("scorer read gold")

    art = _run(root, tmp_path, score_fn=peeking_score_fn, write=False)
    assert art["honest_verdict"] == "blocked_oracle_distinctness_violation"


def test_run_oracle_distinctness_violation_writes(tmp_path: Path) -> None:
    # write=True (the default) exercises the write branch of the
    # OracleDistinctnessError handler, so a peeking scorer still persists a
    # blocked artifact to disk rather than silently dropping the run.
    root = _setup_root(tmp_path, n_questions=12)

    def peeking_score_fn(checkpoint: Any, texts: list[str]) -> list[float]:
        raise mod.OracleDistinctnessError("scorer read gold")

    art = _run(root, tmp_path, score_fn=peeking_score_fn)
    assert art["honest_verdict"] == "blocked_oracle_distinctness_violation"
    assert (tmp_path / "out.json").exists()


def test_run_trainer_raises_generic(tmp_path: Path) -> None:
    root = _setup_root(tmp_path, n_questions=12)

    def boom_trainer(pairs: Any, *, base: Any, out_dir: Any, config: Any) -> dict[str, Any]:
        raise RuntimeError("cuda oom")

    art = _run(root, tmp_path, trainer=boom_trainer, write=False)
    assert art["honest_verdict"] == "blocked_lora_ebm_train_did_not_run"
    assert "cuda oom" in art["blocked_error"]


def test_run_trainer_raises_generic_writes(tmp_path: Path) -> None:
    # write=True (the default) exercises the write branch of the generic
    # exception handler, so a CUDA OOM still leaves a blocked artifact on disk.
    root = _setup_root(tmp_path, n_questions=12)

    def boom_trainer(pairs: Any, *, base: Any, out_dir: Any, config: Any) -> dict[str, Any]:
        raise RuntimeError("cuda oom")

    art = _run(root, tmp_path, trainer=boom_trainer)
    assert art["honest_verdict"] == "blocked_lora_ebm_train_did_not_run"
    assert "cuda oom" in art["blocked_error"]
    assert (tmp_path / "out.json").exists()


def test_run_no_contrastive_pairs(tmp_path: Path) -> None:
    # Gold matches every answer (no negatives) and FoVer has only correct steps
    # -> empty corpus -> no_contrastive_pairs -> train_did_not_run.
    root = _setup_root(tmp_path, n_questions=3, answers=["GOLD", "GOLD"], fover_incorrect=0)
    art = _run(root, tmp_path, min_questions=2, limit=3, write=False)
    assert art["honest_verdict"] == "blocked_lora_ebm_train_did_not_run"
    assert "no_contrastive_pairs" in art["blocked_error"]


def test_run_covers_default_injectables(tmp_path: Path) -> None:
    # Pass None for trainer/score_fn/narratives_loader so their default
    # assignments run; block early on CUDA so the live defaults are never called.
    root = _setup_root(tmp_path, n_questions=3)
    art = mod.run(
        root=root,
        artifact_path=tmp_path / "out.json",
        cuda_available=lambda: False,
        b3_importable=lambda: (True, "ok"),
        base_resolver=_fake_resolver,
        trainer=None,
        score_fn=None,
        narratives_loader=None,
        min_questions=2,
        write=False,
    )
    assert art["honest_verdict"] == "blocked_cuda"
