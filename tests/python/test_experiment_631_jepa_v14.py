"""Tests for scripts/experiment_631_jepa_v14_oracle.py — JEPA v14 ORACLE calibrated retrain.

Coverage targets (new code in this experiment only):
- _load_oracle_corpus: missing file, malformed JSON, flat list, too few chains
- _build_oracle_pairs: no violations, with violations, correct chains included
- _load_flat_corpus: v5 pairs key, v4 list fallback, both missing
- _embed_entries: basic embedding shape and label assignment
- _split_entries: question_index split, OOD split
- _compute_auc: single class returns 0.5, normal case
- _compute_ece: empty returns 0.0, normal case
- _train_capo: runs without error, returns param dict with expected keys
- _select_lambda: selects lowest ECE candidate above AUC floor
- _save_model_npz: creates file with correct arrays
- main: no-corpus path, oracle path, fallback path, schema fields

Spec: REQ-VERIFY-134, REQ-VERIFY-135,
      SCENARIO-VERIFY-175, SCENARIO-VERIFY-176, SCENARIO-VERIFY-177
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import scripts.experiment_631_jepa_v14_oracle as exp631
from scripts.experiment_631_jepa_v14_oracle import (
    _build_oracle_pairs,
    _compute_auc,
    _compute_ece,
    _embed_entries,
    _load_flat_corpus,
    _load_oracle_corpus,
    _save_model_npz,
    _select_lambda,
    _split_entries,
    _train_capo,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_oracle_chain(
    question_id: str,
    is_correct: bool,
    step_labels: list[dict] | None = None,
) -> dict:
    """Build a minimal OracleChain dict for testing."""
    if step_labels is None:
        step_labels = [{"step_index": 0, "step_text": "step", "label": "correct"}]
    return {
        "question_id": question_id,
        "question": f"Q{question_id}?",
        "model_response": f"Resp{question_id}",
        "is_correct": is_correct,
        "step_labels": step_labels,
        "has_violation": any(sl["label"] == "violated" for sl in step_labels),
        "n_violated_steps": sum(1 for sl in step_labels if sl["label"] == "violated"),
    }


def _make_corpus_entry(
    question_index: int = 1,
    is_correct: bool = False,
    response: str = "resp",
) -> dict:
    """Build a flat corpus entry matching v4/v5 format."""
    return {
        "question_index": question_index,
        "question": f"Q{question_index}?",
        "response": response,
        "is_correct": is_correct,
    }


# ---------------------------------------------------------------------------
# _load_oracle_corpus
# ---------------------------------------------------------------------------


def test_load_oracle_corpus_missing_file(tmp_path: Path) -> None:
    """Returns ([], False) when oracle corpus file does not exist."""
    chains, ready = _load_oracle_corpus(tmp_path / "nonexistent.json")
    assert chains == []
    assert ready is False


def test_load_oracle_corpus_malformed_json(tmp_path: Path) -> None:
    """Returns ([], False) when oracle corpus file contains invalid JSON."""
    p = tmp_path / "oracle.json"
    p.write_text("{not valid json")
    chains, ready = _load_oracle_corpus(p)
    assert chains == []
    assert ready is False


def test_load_oracle_corpus_wrong_type(tmp_path: Path) -> None:
    """Returns ([], False) when file contains a dict instead of a list."""
    p = tmp_path / "oracle.json"
    p.write_text(json.dumps({"corpus_ready": True}))
    chains, ready = _load_oracle_corpus(p)
    assert chains == []
    assert ready is False


def test_load_oracle_corpus_too_few_chains(tmp_path: Path) -> None:
    """corpus_ready=False when n_chains < 100."""
    p = tmp_path / "oracle.json"
    p.write_text(json.dumps([_make_oracle_chain(str(i), False) for i in range(50)]))
    chains, ready = _load_oracle_corpus(p)
    assert len(chains) == 50
    assert ready is False


def test_load_oracle_corpus_enough_chains(tmp_path: Path) -> None:
    """corpus_ready=True when n_chains >= 100."""
    p = tmp_path / "oracle.json"
    p.write_text(json.dumps([_make_oracle_chain(str(i), False) for i in range(100)]))
    chains, ready = _load_oracle_corpus(p)
    assert len(chains) == 100
    assert ready is True


# ---------------------------------------------------------------------------
# _build_oracle_pairs
# ---------------------------------------------------------------------------


def test_build_oracle_pairs_no_violations_returns_empty() -> None:
    """Returns empty list when no step has label='violated'.

    This is the SCENARIO-VERIFY-177 fallback trigger: when the oracle corpus
    has n_violated_steps=0, _build_oracle_pairs returns [] so the caller
    falls back to the flat corpus.
    """
    chains = [
        _make_oracle_chain("a", False, [{"step_index": 0, "step_text": "s", "label": "correct"}]),
        _make_oracle_chain("b", True, [{"step_index": 0, "step_text": "s", "label": "correct"}]),
    ]
    pairs = _build_oracle_pairs(chains)
    assert pairs == []


def test_build_oracle_pairs_violated_step_creates_pair() -> None:
    """A violated step in an incorrect chain creates a training pair with is_correct=False."""
    chains = [
        _make_oracle_chain(
            "x",
            False,
            [{"step_index": 0, "step_text": "wrong step", "label": "violated"}],
        )
    ]
    pairs = _build_oracle_pairs(chains)
    assert len(pairs) == 1
    assert pairs[0]["is_correct"] is False
    assert "wrong step" in pairs[0]["response"]


def test_build_oracle_pairs_includes_correct_chains() -> None:
    """Correct chains are added as label=0 entries alongside violated-step pairs."""
    chains = [
        _make_oracle_chain(
            "bad",
            False,
            [{"step_index": 0, "step_text": "vs", "label": "violated"}],
        ),
        _make_oracle_chain("good", True, [{"step_index": 0, "step_text": "cs", "label": "correct"}]),
    ]
    pairs = _build_oracle_pairs(chains)
    assert len(pairs) == 2  # 1 violated + 1 correct
    is_correct_values = {p["is_correct"] for p in pairs}
    assert True in is_correct_values
    assert False in is_correct_values


def test_build_oracle_pairs_question_index_derived_from_id() -> None:
    """question_index is derived from question_id hash so split logic works."""
    chains = [
        _make_oracle_chain(
            "qid123",
            False,
            [{"step_index": 0, "step_text": "vs", "label": "violated"}],
        )
    ]
    pairs = _build_oracle_pairs(chains)
    assert "question_index" in pairs[0]
    assert isinstance(pairs[0]["question_index"], int)


# ---------------------------------------------------------------------------
# _load_flat_corpus
# ---------------------------------------------------------------------------


def test_load_flat_corpus_v5_pairs_key(tmp_path: Path) -> None:
    """Loads pairs from a corpus dict with a 'pairs' key."""
    data = {"pairs": [_make_corpus_entry(1), _make_corpus_entry(2)]}
    p = tmp_path / "v5.json"
    p.write_text(json.dumps(data))
    entries = _load_flat_corpus(p, tmp_path / "v4.json")
    assert len(entries) == 2


def test_load_flat_corpus_v4_flat_list(tmp_path: Path) -> None:
    """Loads entries from a flat list (v4 format) when v5 is missing."""
    data = [_make_corpus_entry(1), _make_corpus_entry(2), _make_corpus_entry(3)]
    p = tmp_path / "v4.json"
    p.write_text(json.dumps(data))
    entries = _load_flat_corpus(tmp_path / "v5.json", p)
    assert len(entries) == 3


def test_load_flat_corpus_both_missing(tmp_path: Path) -> None:
    """Returns empty list when both corpus files are missing."""
    entries = _load_flat_corpus(tmp_path / "a.json", tmp_path / "b.json")
    assert entries == []


def test_load_flat_corpus_v5_preferred_over_v4(tmp_path: Path) -> None:
    """v5 corpus is used when it exists, even if v4 also exists."""
    v5 = tmp_path / "v5.json"
    v4 = tmp_path / "v4.json"
    v5.write_text(json.dumps({"pairs": [_make_corpus_entry(1)]}))
    v4.write_text(json.dumps([_make_corpus_entry(2), _make_corpus_entry(3)]))
    entries = _load_flat_corpus(v5, v4)
    assert len(entries) == 1  # v5 has 1 pair


# ---------------------------------------------------------------------------
# _embed_entries
# ---------------------------------------------------------------------------


def test_embed_entries_shape_and_labels() -> None:
    """_embed_entries returns (N, embed_dim) embeddings and (N,) labels."""
    import jax.numpy as jnp

    embed_fn = exp631._make_embed_fn(embed_dim=16, seed=0)
    entries = [
        {"question": "Q1?", "response": "right", "is_correct": True},
        {"question": "Q2?", "response": "wrong", "is_correct": False},
    ]
    embs, labels = _embed_entries(entries, embed_fn)
    assert embs.shape == (2, 16)
    assert labels.shape == (2,)
    assert int(labels[0]) == 0  # is_correct=True -> label=0
    assert int(labels[1]) == 1  # is_correct=False -> label=1


# ---------------------------------------------------------------------------
# _split_entries
# ---------------------------------------------------------------------------


def test_split_entries_no_leakage() -> None:
    """Train and test sets share no question_index values."""
    entries = [_make_corpus_entry(i, is_correct=(i % 2 == 0)) for i in range(20)]
    train, test, ood = _split_entries(entries, train_frac=0.8, seed=0)
    train_indices = {e["question_index"] for e in train}
    test_indices = {e["question_index"] for e in test}
    assert train_indices.isdisjoint(test_indices)


def test_split_entries_ood_from_top_indices() -> None:
    """OOD entries use the top 20% of question_index values."""
    entries = [_make_corpus_entry(i, is_correct=(i % 2 == 0)) for i in range(20)]
    _, _, ood = _split_entries(entries, train_frac=0.8, seed=0)
    ood_indices = {e["question_index"] for e in ood}
    # All 20 unique indices; top 20% = indices 16-19
    assert all(idx >= 16 for idx in ood_indices)


# ---------------------------------------------------------------------------
# _compute_auc
# ---------------------------------------------------------------------------


def test_compute_auc_single_class_returns_half() -> None:
    """Returns 0.5 when all entries have the same label (AUC undefined)."""
    embed_fn = exp631._make_embed_fn(embed_dim=16, seed=0)
    entries = [_make_corpus_entry(i, is_correct=False) for i in range(5)]
    # init_params with small key
    import jax.random as jrandom
    params = exp631._init_params(jrandom.PRNGKey(0), embed_dim=16)
    auc = _compute_auc(params, entries, embed_fn)
    assert auc == pytest.approx(0.5, abs=1e-6)


def test_compute_auc_empty_entries_returns_half() -> None:
    """Returns 0.5 when entries list is empty."""
    embed_fn = exp631._make_embed_fn(embed_dim=16, seed=0)
    import jax.random as jrandom
    params = exp631._init_params(jrandom.PRNGKey(0), embed_dim=16)
    auc = _compute_auc(params, [], embed_fn)
    assert auc == pytest.approx(0.5, abs=1e-6)


# ---------------------------------------------------------------------------
# _compute_ece
# ---------------------------------------------------------------------------


def test_compute_ece_empty_returns_zero() -> None:
    """Returns 0.0 when entries list is empty."""
    embed_fn = exp631._make_embed_fn(embed_dim=16, seed=0)
    import jax.random as jrandom
    params = exp631._init_params(jrandom.PRNGKey(0), embed_dim=16)
    ece = _compute_ece(params, [], embed_fn)
    assert ece == pytest.approx(0.0, abs=1e-6)


def test_compute_ece_returns_float() -> None:
    """Returns a float for a non-empty entry list."""
    embed_fn = exp631._make_embed_fn(embed_dim=16, seed=0)
    entries = [
        _make_corpus_entry(1, is_correct=True),
        _make_corpus_entry(2, is_correct=False),
    ]
    import jax.random as jrandom
    params = exp631._init_params(jrandom.PRNGKey(0), embed_dim=16)
    ece = _compute_ece(params, entries, embed_fn)
    assert isinstance(ece, float)
    assert 0.0 <= ece <= 1.0


# ---------------------------------------------------------------------------
# _train_capo
# ---------------------------------------------------------------------------


def test_train_capo_returns_param_dict() -> None:
    """_train_capo returns a param dict with expected MLP keys."""
    embed_fn = exp631._make_embed_fn(embed_dim=16, seed=0)
    entries = [
        _make_corpus_entry(i, is_correct=(i % 2 == 0)) for i in range(10)
    ]
    params = _train_capo(
        entries, embed_fn, n_epochs=2, lambda_calib=0.10, seed=0, embed_dim=16
    )
    assert set(params.keys()) == {"w1", "b1", "w2", "b2"}
    assert params["w1"].shape == (16, 16)
    assert params["w2"].shape == (1, 16)


# ---------------------------------------------------------------------------
# _select_lambda
# ---------------------------------------------------------------------------


def test_select_lambda_picks_lowest_ece_above_auc_floor() -> None:
    """_select_lambda selects candidate with lowest ECE among those above AUC floor.

    We mock _train_capo and the evaluation functions so the test runs fast
    and verifies selection logic only.
    """
    embed_fn = exp631._make_embed_fn(embed_dim=16, seed=0)
    entries = [_make_corpus_entry(i, is_correct=(i % 2 == 0)) for i in range(10)]

    call_idx = [0]

    # Fake training returns distinguishable param dicts via different seeds.
    def fake_train(train_e, embed_f, n_epochs, lambda_calib, seed, **kw):
        import jax.random as jrandom
        return exp631._init_params(jrandom.PRNGKey(int(lambda_calib * 100)), embed_dim=16)

    # Fake AUC/ECE: lambda=0.05 -> auc=0.80, ece=0.15
    #               lambda=0.10 -> auc=0.75, ece=0.08  <- best ECE above floor
    #               lambda=0.20 -> auc=0.70, ece=0.12
    auc_map = {0.05: 0.80, 0.10: 0.75, 0.20: 0.70}
    ece_map = {0.05: 0.15, 0.10: 0.08, 0.20: 0.12}

    import jax.random as jrandom

    def fake_auc(params, entries, embed_f):
        # Identify lambda from PRNGKey seed embedded in w1 shape (can't introspect)
        # Instead use a call-count approach.
        call_idx[0] += 1
        return list(auc_map.values())[(call_idx[0] - 1) % 3]

    ece_idx = [0]

    def fake_ece(params, entries, embed_f):
        ece_idx[0] += 1
        return list(ece_map.values())[(ece_idx[0] - 1) % 3]

    with (
        patch.object(exp631, "_train_capo", side_effect=fake_train),
        patch.object(exp631, "_compute_auc", side_effect=fake_auc),
        patch.object(exp631, "_compute_ece", side_effect=fake_ece),
    ):
        best_lambda, best_params = _select_lambda(
            entries, entries, embed_fn,
            candidates=[0.05, 0.10, 0.20],
            n_epochs=2,
            auc_floor=0.70,
        )

    # 0.10 has ECE=0.08 (lowest) and AUC=0.75 (above floor).
    assert best_lambda == pytest.approx(0.10, abs=1e-6)


def test_select_lambda_all_below_auc_floor_picks_highest_auc() -> None:
    """When all lambdas fall below AUC floor, picks candidate with highest AUC."""
    embed_fn = exp631._make_embed_fn(embed_dim=16, seed=0)
    entries = [_make_corpus_entry(i, is_correct=(i % 2 == 0)) for i in range(10)]

    auc_values = [0.60, 0.65, 0.55]  # all below floor=0.70; 0.10 has highest AUC
    ece_values = [0.20, 0.25, 0.18]
    auc_idx = [0]
    ece_idx = [0]

    def fake_train(train_e, embed_f, n_epochs, lambda_calib, seed, **kw):
        import jax.random as jrandom
        return exp631._init_params(jrandom.PRNGKey(0), embed_dim=16)

    def fake_auc(params, entries, embed_f):
        v = auc_values[auc_idx[0] % 3]
        auc_idx[0] += 1
        return v

    def fake_ece(params, entries, embed_f):
        v = ece_values[ece_idx[0] % 3]
        ece_idx[0] += 1
        return v

    with (
        patch.object(exp631, "_train_capo", side_effect=fake_train),
        patch.object(exp631, "_compute_auc", side_effect=fake_auc),
        patch.object(exp631, "_compute_ece", side_effect=fake_ece),
    ):
        best_lambda, best_params = _select_lambda(
            entries, entries, embed_fn,
            candidates=[0.05, 0.10, 0.20],
            n_epochs=2,
            auc_floor=0.70,
        )

    # 0.10 has AUC=0.65 (highest of the three below-floor candidates)
    assert best_lambda == pytest.approx(0.10, abs=1e-6)


# ---------------------------------------------------------------------------
# _save_model_npz
# ---------------------------------------------------------------------------


def test_save_model_npz_writes_correct_arrays(tmp_path: Path) -> None:
    """Saved .npz contains the same arrays as the param dict."""
    import jax.numpy as jnp
    import jax.random as jrandom

    params = exp631._init_params(jrandom.PRNGKey(0), embed_dim=16)
    out_path = tmp_path / "model.npz"
    _save_model_npz(params, out_path)
    assert out_path.exists()
    loaded = np.load(str(out_path))
    for key in ("w1", "b1", "w2", "b2"):
        assert key in loaded
        np.testing.assert_allclose(np.array(params[key]), loaded[key], atol=1e-6)


# ---------------------------------------------------------------------------
# main() integration
# ---------------------------------------------------------------------------


def _make_flat_corpus_file(path: Path, n_correct: int, n_incorrect: int) -> None:
    """Write a minimal flat corpus JSON with correct/incorrect pairs."""
    entries = [
        {
            "question_index": i,
            "question": f"Q{i}?",
            "response": f"resp{i}",
            "is_correct": True,
        }
        for i in range(n_correct)
    ] + [
        {
            "question_index": n_correct + i,
            "question": f"Q{n_correct + i}?",
            "response": f"wrong{i}",
            "is_correct": False,
        }
        for i in range(n_incorrect)
    ]
    path.write_text(json.dumps(entries))


def test_main_no_corpus_writes_blocked_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """main() writes honest_verdict='blocked_no_corpus' when all corpus files are missing."""
    results_dir = tmp_path / "results"
    results_dir.mkdir()

    monkeypatch.setattr(exp631, "_REPO_ROOT", tmp_path)
    monkeypatch.setattr(exp631, "ORACLE_CORPUS_PATH", tmp_path / "results" / "oracle.json")
    monkeypatch.setattr(exp631, "CORPUS_V5_PATH", tmp_path / "results" / "v5.json")
    monkeypatch.setattr(exp631, "CORPUS_V4_PATH", tmp_path / "results" / "v4.json")
    monkeypatch.setattr(exp631, "MODEL_OUT_PATH", tmp_path / "results" / "model.npz")
    monkeypatch.setattr(exp631, "DELIVERABLE", "results/experiment_631_jepa_v14_oracle.json")
    monkeypatch.setattr(exp631, "apply_env_autofix", lambda: None)
    monkeypatch.setattr(
        exp631, "ExperimentTimeoutWatchdog",
        lambda exp_id, timeout_minutes=40: _NullCtx(),
    )

    exp631.main()

    out = json.loads((results_dir / "experiment_631_jepa_v14_oracle.json").read_text())
    assert out["honest_verdict"] == "blocked_no_corpus"
    assert out["status"] == "blocked_no_corpus"


def test_main_fallback_corpus_writes_success_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """main() uses flat fallback corpus and writes a success artifact with required fields."""
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    _make_flat_corpus_file(results_dir / "v5.json", n_correct=50, n_incorrect=50)

    monkeypatch.setattr(exp631, "_REPO_ROOT", tmp_path)
    monkeypatch.setattr(exp631, "ORACLE_CORPUS_PATH", tmp_path / "results" / "oracle.json")
    monkeypatch.setattr(exp631, "CORPUS_V5_PATH", results_dir / "v5.json")
    monkeypatch.setattr(exp631, "CORPUS_V4_PATH", tmp_path / "results" / "v4.json")
    monkeypatch.setattr(exp631, "MODEL_OUT_PATH", results_dir / "model.npz")
    monkeypatch.setattr(exp631, "DELIVERABLE", "results/experiment_631_jepa_v14_oracle.json")
    monkeypatch.setattr(exp631, "N_EPOCHS", 2)
    monkeypatch.setattr(exp631, "LAMBDA_CALIB_CANDIDATES", [0.10])
    monkeypatch.setattr(exp631, "apply_env_autofix", lambda: None)
    monkeypatch.setattr(
        exp631, "ExperimentTimeoutWatchdog",
        lambda exp_id, timeout_minutes=40: _NullCtx(),
    )

    exp631.main()

    out = json.loads((results_dir / "experiment_631_jepa_v14_oracle.json").read_text())
    assert out["status"] == "success"
    assert out["corpus_source"] == "fallback"
    for field in (
        "result_schema", "v13_ood_auc", "v13_ece",
        "v14_in_dist_auc", "v14_ood_auc", "v14_ece",
        "lambda_calib_selected", "n_training_pairs",
        "calibration_improved", "ood_maintained",
        "model_saved", "honest_verdict",
    ):
        assert field in out, f"Missing required field: {field}"
    assert out["result_schema"] == "carnot.jepa_v14_oracle.v1"
    assert out["v13_ood_auc"] == pytest.approx(0.868, abs=1e-3)
    assert out["v13_ece"] == pytest.approx(0.207, abs=1e-3)


def test_main_oracle_fallback_when_no_violated_steps(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """main() falls back to flat corpus when oracle exists but has no violated steps.

    SCENARIO-VERIFY-177: oracle corpus_ready=True but n_violated_steps=0 everywhere.
    """
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    # Write oracle with 100 chains, all steps labeled 'correct'
    oracle_chains = [_make_oracle_chain(str(i), False) for i in range(100)]
    (results_dir / "oracle.json").write_text(json.dumps(oracle_chains))
    _make_flat_corpus_file(results_dir / "v5.json", n_correct=30, n_incorrect=30)

    monkeypatch.setattr(exp631, "_REPO_ROOT", tmp_path)
    monkeypatch.setattr(exp631, "ORACLE_CORPUS_PATH", results_dir / "oracle.json")
    monkeypatch.setattr(exp631, "CORPUS_V5_PATH", results_dir / "v5.json")
    monkeypatch.setattr(exp631, "CORPUS_V4_PATH", tmp_path / "results" / "v4.json")
    monkeypatch.setattr(exp631, "MODEL_OUT_PATH", results_dir / "model.npz")
    monkeypatch.setattr(exp631, "DELIVERABLE", "results/experiment_631_jepa_v14_oracle.json")
    monkeypatch.setattr(exp631, "N_EPOCHS", 2)
    monkeypatch.setattr(exp631, "LAMBDA_CALIB_CANDIDATES", [0.10])
    monkeypatch.setattr(exp631, "apply_env_autofix", lambda: None)
    monkeypatch.setattr(
        exp631, "ExperimentTimeoutWatchdog",
        lambda exp_id, timeout_minutes=40: _NullCtx(),
    )

    exp631.main()

    out = json.loads((results_dir / "experiment_631_jepa_v14_oracle.json").read_text())
    assert out["status"] == "success"
    # corpus_source must be 'fallback' because oracle had 0 violated steps
    assert out["corpus_source"] == "fallback"


def test_main_honest_verdict_logic(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """honest_verdict field reflects calibration_improved and ood_maintained flags."""
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    _make_flat_corpus_file(results_dir / "v5.json", n_correct=30, n_incorrect=30)

    monkeypatch.setattr(exp631, "_REPO_ROOT", tmp_path)
    monkeypatch.setattr(exp631, "ORACLE_CORPUS_PATH", tmp_path / "results" / "oracle.json")
    monkeypatch.setattr(exp631, "CORPUS_V5_PATH", results_dir / "v5.json")
    monkeypatch.setattr(exp631, "CORPUS_V4_PATH", tmp_path / "results" / "v4.json")
    monkeypatch.setattr(exp631, "MODEL_OUT_PATH", results_dir / "model.npz")
    monkeypatch.setattr(exp631, "DELIVERABLE", "results/experiment_631_jepa_v14_oracle.json")
    monkeypatch.setattr(exp631, "N_EPOCHS", 2)
    monkeypatch.setattr(exp631, "LAMBDA_CALIB_CANDIDATES", [0.10])
    monkeypatch.setattr(exp631, "apply_env_autofix", lambda: None)
    monkeypatch.setattr(
        exp631, "ExperimentTimeoutWatchdog",
        lambda exp_id, timeout_minutes=40: _NullCtx(),
    )

    exp631.main()

    out = json.loads((results_dir / "experiment_631_jepa_v14_oracle.json").read_text())
    calib = out["calibration_improved"]
    ood = out["ood_maintained"]
    expected = (
        "v14_calibrated_ood_maintained" if (calib and ood)
        else "v14_calibrated_ood_dropped" if calib
        else "v14_uncalibrated"
    )
    assert out["honest_verdict"] == expected


# ---------------------------------------------------------------------------
# Null context manager for monkeypatching ExperimentTimeoutWatchdog
# ---------------------------------------------------------------------------


class _NullCtx:
    """A context manager that does nothing — replaces ExperimentTimeoutWatchdog in tests."""

    def __enter__(self) -> "_NullCtx":
        return self

    def __exit__(self, *args: object) -> None:
        pass
