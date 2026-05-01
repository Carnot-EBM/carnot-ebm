"""Tests for Exp 1111 — ThinkPRM v2 retrain on the 7349-example PRM corpus.

These tests cover the *logic the experiment adds on top of* the existing
ThinkPRMProbe — corpus loading, label mapping, stratified split, BCE
loss, the run_experiment() orchestration end-to-end with a small
synthetic backbone, and the Zenil α_t convergence-condition check.

Heavy dependencies (the real Qwen3.5-0.8B backbone, GPU) are mocked so
the tests run in well under a second on CPU. Each test references the
spec requirement it backs.

Spec: REQ-VERIFY-098 (verifier learning), REQ-LEARN-011 (PRM head
      training), REQ-DIAG-001 (alpha_t tracking).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

_REPO = Path(__file__).resolve().parent.parent.parent
for _d in [str(_REPO / "python"), str(_REPO / "scripts"), str(_REPO)]:
    if _d not in sys.path:
        sys.path.insert(0, _d)

import importlib.util as _ilu

_spec = _ilu.spec_from_file_location(
    "exp1111",
    str(_REPO / "scripts" / "experiment_1111_thinkprm_v2_retrain_7349_prm.py"),
)
exp1111 = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(exp1111)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    """Tiny JSONL writer used to fabricate a synthetic PRM corpus on tmp_path."""
    with path.open("w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")


def _make_synthetic_prm(n: int = 60, frac_correct: float = 0.7, seed: int = 0) -> list[dict]:
    """Build a small but class-imbalanced synthetic PRM corpus.

    The exp1111 stratified split should preserve the (positive-heavy)
    class ratio rather than landing a validation slice with zero
    negatives.  Frac 0.7 is high enough to exercise that path while
    leaving enough negatives for AUROC to be defined on every split.
    """
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        label = "correct" if rng.random() < frac_correct else "wrong"
        out.append(
            {
                "question_id": str(i),
                "partial_cot": f"step {i} reasoning text",
                "step_label": label,
                "full_cot_correct": label == "correct",
                "cascade_score": float(rng.random()),
                "prefix_fraction": float(rng.random()),
            }
        )
    return out


# ---------------------------------------------------------------------------
# 1. PRM corpus loader maps `correct`/`wrong` to {1, 0}
# ---------------------------------------------------------------------------


def test_load_prm_corpus_label_mapping_correct_is_positive(tmp_path: Path) -> None:
    """REQ-LEARN-011: loader must map `correct` -> 1.0 and `wrong` -> 0.0.

    Why this matters: the LogisticProbe assumes y=1 is the positive
    class. A 1/0 swap silently inverts AUROC (the 2026-04-28 sign-error
    incident).  Pin the mapping with a unit test rather than rely on
    careful prose in the docstring.
    """
    corpus_path = tmp_path / "prm.jsonl"
    rows = [
        {"question_id": "1", "partial_cot": "good step", "step_label": "correct"},
        {"question_id": "2", "partial_cot": "bad step", "step_label": "wrong"},
        {"question_id": "3", "partial_cot": "another good step", "step_label": "correct"},
    ]
    _write_jsonl(corpus_path, rows)

    texts, y = exp1111.load_prm_corpus(corpus_path)

    assert texts == ["good step", "bad step", "another good step"]
    assert y.tolist() == [1.0, 0.0, 1.0]
    assert y.dtype == np.float32


# ---------------------------------------------------------------------------
# 2. Stratified split keeps class ratio across train/val
# ---------------------------------------------------------------------------


def test_stratified_split_preserves_class_ratio() -> None:
    """REQ-LEARN-011: the 80/20 split must not strand all negatives in one slice.

    With ~93% positives in the real PRM corpus, a uniform shuffle would
    occasionally yield a validation slice with zero wrongs — making
    AUROC degenerate to 0.5 by definition.  The stratified split must
    keep the train/val positive-fractions within a few percentage
    points of each other.
    """
    rows = _make_synthetic_prm(n=200, frac_correct=0.93, seed=1)
    texts = [r["partial_cot"] for r in rows]
    y = np.asarray([1.0 if r["step_label"] == "correct" else 0.0 for r in rows], dtype=np.float32)

    train_texts, y_train, val_texts, y_val = exp1111.stratified_split(texts, y)

    assert len(train_texts) + len(val_texts) == len(texts)
    assert y_train.sum() > 0 and y_val.sum() > 0
    # Both slices must contain at least one negative — the bug we are
    # specifically guarding against.
    assert (y_train < 0.5).sum() > 0
    assert (y_val < 0.5).sum() > 0
    # Class ratio drift between splits should be below 5 pp.
    train_frac_pos = float(y_train.mean())
    val_frac_pos = float(y_val.mean())
    assert abs(train_frac_pos - val_frac_pos) < 0.05


# ---------------------------------------------------------------------------
# 3. BCE loss helper computes correct value on a known case
# ---------------------------------------------------------------------------


def test_bce_loss_matches_hand_computation() -> None:
    """REQ-LEARN-011: val_loss helper must match a hand-computed BCE.

    The artifact reports `final_val_loss`; downstream consumers compare
    that across milestones, so a silently-wrong BCE would corrupt
    cross-experiment trend lines.  Pin the formula here.
    """
    y = np.asarray([1.0, 0.0, 1.0, 0.0])
    p = np.asarray([0.9, 0.1, 0.8, 0.2])
    expected = -np.mean(
        y * np.log(np.clip(p, 1e-7, 1 - 1e-7)) + (1 - y) * np.log(1 - np.clip(p, 1e-7, 1 - 1e-7))
    )
    assert exp1111.bce_loss(y, p) == pytest.approx(float(expected), abs=1e-6)


# ---------------------------------------------------------------------------
# 4. Trains without error on a small fabricated corpus (mocked backbone)
# ---------------------------------------------------------------------------


@pytest.fixture
def patched_corpus_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point exp1111 at a synthetic PRM + FoVer pair in tmp_path.

    Lets us exercise run_experiment() end-to-end without the real
    1.9 MB JSONL or HuggingFace model download.  The fixture also
    monkeypatches ThinkPRMProbe's hidden-state extractor so no actual
    transformer is loaded.
    """
    prm = tmp_path / "prm.jsonl"
    fover = tmp_path / "fover.json"
    _write_jsonl(prm, _make_synthetic_prm(n=80, frac_correct=0.7, seed=2))

    fover_rows = []
    rng = np.random.default_rng(3)
    for i in range(40):
        is_correct = rng.random() < 0.7
        fover_rows.append(
            {
                "question_id": str(i),
                "step_text": f"fover step {i}",
                "label": "correct" if is_correct else "incorrect",
                "confidence": 1.0,
            }
        )
    fover.write_text(json.dumps(fover_rows))

    monkeypatch.setattr(exp1111, "PRM_TRAIN_PATH", prm)
    monkeypatch.setattr(exp1111, "FOVER_PATH", fover)
    return tmp_path


def test_thinkprm_v2_trains_without_error_on_prm_corpus(
    patched_corpus_paths: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """thinkprm_v2_trains_without_error_on_prm_corpus.

    Spec: REQ-LEARN-011 — the PRM-head training pipeline must run end
    to end on a representative-shape corpus without raising.

    We mock the ThinkPRMProbe hidden-state extractor so no real model
    is loaded; the test still exercises the full data path: load,
    split, fit_features, fit_classifier, predict_proba, AUROC, α_t.
    """

    def fake_extract(self, texts, batch_size, max_length):
        # Synthesise hidden states whose first dim correlates with the
        # text-length parity — a deterministic but learnable signal.
        rng = np.random.default_rng(abs(hash(tuple(texts[:5]))) % (2**32))
        n = len(texts)
        h = rng.standard_normal(size=(n, 64)).astype(np.float32)
        for i, t in enumerate(texts):
            h[i, 0] += 2.0 * (len(t) % 2)
        self._model_used = "synthetic-mock-backbone"
        return h

    monkeypatch.setattr(
        "carnot.verify.thinkprm_probe.ThinkPRMProbe._extract_hidden_states",
        fake_extract,
    )

    payload = exp1111.run_experiment(
        n_pca_dims=4,
        classifier_epochs=50,
        fover_eval_size=20,
        backbone_model_id="synthetic-mock-backbone",
    )

    assert payload["training_examples"] == 80
    assert payload["n_train"] + payload["n_val"] == 80
    assert "thinkprm_v2_auroc" in payload
    assert payload["honest_verdict"] in {
        "auroc_above_995",
        "auroc_improved_below_995",
        "auroc_no_improvement",
    }


# ---------------------------------------------------------------------------
# 5. v2 AUROC must be at least at the v1 baseline on the FoVer eval slice
# ---------------------------------------------------------------------------


def test_thinkprm_v2_auroc_above_baseline_on_fover_eval(
    patched_corpus_paths: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """thinkprm_v2_auroc_above_baseline_on_fover_eval.

    Spec: REQ-VERIFY-098 — the retrained probe must, on a corpus where
    the true label is *learnable* from the hidden state, achieve AUROC
    ≥ the published v1 baseline (0.9885).  We construct that corpus
    deterministically: the synthetic backbone places the label in
    feature 0, so a properly-fitted logistic head should saturate.
    """

    def fake_extract(self, texts, batch_size, max_length):
        # Compute label from the same rule the corpus generator used so
        # the synthetic backbone is *capable* of separating the classes
        # perfectly when fed enough training examples.
        n = len(texts)
        h = np.random.default_rng(0).standard_normal(size=(n, 64)).astype(np.float32)
        for i, t in enumerate(texts):
            # encode the label in feature 0; for prm corpus we infer
            # from the suffix/index parity, for fover corpus likewise.
            sig = 1.0 if "fover" in t and int(t.split()[-1]) % 2 == 0 else 0.0
            sig = 1.0 if ("fover" not in t and int(t.split()[1]) % 2 == 0) else sig
            h[i, 0] = 5.0 if sig else -5.0
        self._model_used = "synthetic-mock-backbone"
        return h

    monkeypatch.setattr(
        "carnot.verify.thinkprm_probe.ThinkPRMProbe._extract_hidden_states",
        fake_extract,
    )

    # Rewrite the synthetic corpora so labels match the parity rule that
    # fake_extract uses, otherwise the probe is being asked to learn a
    # function that does not exist in its features.
    prm_path = exp1111.PRM_TRAIN_PATH
    rows = []
    for i in range(120):
        label = "correct" if i % 2 == 0 else "wrong"
        rows.append(
            {
                "question_id": str(i),
                "partial_cot": f"step {i}",
                "step_label": label,
                "full_cot_correct": label == "correct",
                "cascade_score": 1.0,
                "prefix_fraction": 0.5,
            }
        )
    _write_jsonl(prm_path, rows)

    fover_rows = []
    for i in range(40):
        label = "correct" if i % 2 == 0 else "incorrect"
        fover_rows.append(
            {
                "question_id": str(i),
                "step_text": f"fover step {i}",
                "label": label,
                "confidence": 1.0,
            }
        )
    exp1111.FOVER_PATH.write_text(json.dumps(fover_rows))

    payload = exp1111.run_experiment(
        n_pca_dims=4,
        classifier_epochs=300,
        fover_eval_size=40,
        backbone_model_id="synthetic-mock-backbone",
    )

    # AUROC must be at or above the published v1 baseline on this
    # learnable corpus.  We do NOT assert >= 0.995 here — that is the
    # real-data target captured in the experiment artifact, not a
    # universal property of the algorithm.
    assert payload["thinkprm_v2_auroc"] >= exp1111.THINKPRM_V1_AUROC


# ---------------------------------------------------------------------------
# 6. Zenil α_t > 0 — convergence condition for self-distillation
# ---------------------------------------------------------------------------


def test_alpha_t_above_zero_zenil_condition(
    patched_corpus_paths: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """alpha_t_above_zero_zenil_condition.

    Spec: REQ-DIAG-001 — Zenil's Theorem 5 requires inf_t α_t > 0 for
    a self-distillation loop to converge to a useful fixed point.
    The experiment hard-codes the live-pipeline measurement from
    exp1077 (α_t ≈ 0.38); this test guarantees the recorded value is
    strictly positive and matches the documented constant.
    """

    def fake_extract(self, texts, batch_size, max_length):
        rng = np.random.default_rng(7)
        return rng.standard_normal(size=(len(texts), 32)).astype(np.float32)

    monkeypatch.setattr(
        "carnot.verify.thinkprm_probe.ThinkPRMProbe._extract_hidden_states",
        fake_extract,
    )

    payload = exp1111.run_experiment(
        n_pca_dims=4,
        classifier_epochs=20,
        fover_eval_size=20,
        backbone_model_id="synthetic-mock-backbone",
    )

    assert payload["alpha_t_training_corpus"] > 0.0
    assert payload["alpha_t_above_zero"] is True
    # Should match the documented exp1077 measurement to two decimals.
    assert abs(payload["alpha_t_training_corpus"] - exp1111.ALPHA_T_VERIFIED_FRACTION) < 0.01
