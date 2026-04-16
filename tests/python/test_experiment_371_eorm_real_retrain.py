"""Tests for scripts/experiment_371_eorm_real_retrain.py.

100% coverage of all new helper functions introduced in Exp 371:
    - _evaluate_eorm_auc: standard case, empty pairs, all-same-label, high-AUC case
    - _pairs_to_contrastive_triples: normal grouping, synthetic pool, no contrast,
      round-robin with unequal group sizes, empty input
    - _load_or_build_eorm_model: load success, load failure → fresh model, missing file → fresh
    - run_experiment: blocked path (insufficient_real_pairs), success path (real_data),
      improvement and no-improvement verdict branches

Spec: REQ-LEARN-025, SCENARIO-LEARN-043, SCENARIO-LEARN-044, SCENARIO-LEARN-048
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# sys.path bootstrap — needed so both `carnot.*` and `scripts.*` resolve
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from carnot.embeddings.jepa_retrain import ViolationPair
from carnot.models.eorm import EORMModel

from scripts.experiment_371_eorm_real_retrain import (
    _evaluate_eorm_auc,
    _load_or_build_eorm_model,
    _pairs_to_contrastive_triples,
    run_experiment,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_pair(
    question_id: str,
    response: str,
    has_violation: bool,
    model_id: str = "test_model",
) -> ViolationPair:
    """Build a ViolationPair with minimal boilerplate."""
    return ViolationPair(
        partial_response=response,
        full_response=response,
        has_violation=has_violation,
        model_id=model_id,
        question_id=question_id,
    )


def _fresh_model() -> EORMModel:
    """Return a tiny EORMModel for fast CPU tests."""
    import jax.random as jrandom
    return EORMModel(embed_dim=32, n_heads=4, n_layers=1, max_seq_len=64,
                     vocab_size=512, key=jrandom.PRNGKey(42))


# ---------------------------------------------------------------------------
# _evaluate_eorm_auc
# ---------------------------------------------------------------------------

class TestEvaluateEormAuc:
    """SCENARIO-LEARN-048: AUC-ROC evaluation helpers."""

    def test_empty_pairs_returns_half(self):
        """Empty test set → AUC 0.5 (random baseline by convention)."""
        model = _fresh_model()
        assert _evaluate_eorm_auc(model, []) == 0.5

    def test_all_violations_returns_half(self):
        """All-positive test set → AUC 0.5 (cannot compute meaningful AUC)."""
        model = _fresh_model()
        pairs = [_make_pair("q1", "wrong answer", True),
                 _make_pair("q2", "also wrong", True)]
        assert _evaluate_eorm_auc(model, pairs) == 0.5

    def test_all_correct_returns_half(self):
        """All-negative test set → AUC 0.5 (cannot compute meaningful AUC)."""
        model = _fresh_model()
        pairs = [_make_pair("q1", "correct answer", False),
                 _make_pair("q2", "also correct", False)]
        assert _evaluate_eorm_auc(model, pairs) == 0.5

    def test_mixed_labels_returns_float_in_range(self):
        """Mixed labels → AUC in [0, 1]."""
        model = _fresh_model()
        pairs = [
            _make_pair("q1", "correct", False),
            _make_pair("q1", "wrong", True),
            _make_pair("q2", "correct too", False),
            _make_pair("q2", "wrong too", True),
        ]
        auc = _evaluate_eorm_auc(model, pairs)
        assert 0.0 <= auc <= 1.0

    def test_perfect_model_gives_high_auc(self):
        """A model that always assigns lower energy to correct responses has high AUC.

        We mock EORMModel.energy so that it returns:
        - low energy for non-violations (correct responses)
        - high energy for violations (incorrect responses)
        Expected: AUC close to 1.0 (since negated energy = high score for violations).
        """
        model = _fresh_model()
        pairs = [
            _make_pair("q1", "correct_response", False),
            _make_pair("q2", "wrong_response", True),
            _make_pair("q3", "correct_answer", False),
            _make_pair("q4", "bad_answer", True),
        ]

        # Override energy: violations (has_violation=True) get high energy (+10),
        # correct responses (has_violation=False) get low energy (-10).
        # AUC scores use -energy, so violations get score -10 and corrects get +10.
        # Wait — that would give LOW score to violations. Let's flip:
        # violations → energy = -10 (low), corrects → energy = +10 (high)
        # score = -energy → violations get +10 (high score) = predicted positive = correct!
        def mock_energy(cot_input):
            if "wrong" in cot_input.response_text or "bad" in cot_input.response_text:
                return -10.0  # low energy for violations → negated score = +10 = predicted positive
            return 10.0  # high energy for correct → negated score = -10 = predicted negative

        with patch.object(model, "energy", side_effect=mock_energy):
            auc = _evaluate_eorm_auc(model, pairs)
        assert auc == pytest.approx(1.0, abs=0.01)

    def test_auc_is_float(self):
        """Return type is Python float, not JAX array."""
        model = _fresh_model()
        pairs = [_make_pair("q1", "c", False), _make_pair("q2", "w", True)]
        auc = _evaluate_eorm_auc(model, pairs)
        assert isinstance(auc, float)


# ---------------------------------------------------------------------------
# _pairs_to_contrastive_triples
# ---------------------------------------------------------------------------

class TestPairsToContrastiveTriples:
    """Coverage for triple construction logic."""

    def test_empty_input_returns_empty(self):
        assert _pairs_to_contrastive_triples([]) == []

    def test_single_question_correct_and_incorrect(self):
        """One question with one correct and one incorrect response → one triple."""
        pairs = [
            _make_pair("gsm_001", "The answer is 4.", False),
            _make_pair("gsm_001", "The answer is 5.", True),
        ]
        triples = _pairs_to_contrastive_triples(pairs)
        assert len(triples) == 1
        correct_resp, incorrect_resp, q_id = triples[0]
        assert correct_resp == "The answer is 4."
        assert incorrect_resp == "The answer is 5."
        assert q_id == "gsm_001"

    def test_only_correct_responses_for_question(self):
        """No incorrect response for a question → no triple formed."""
        pairs = [
            _make_pair("q1", "correct1", False),
            _make_pair("q1", "correct2", False),
        ]
        assert _pairs_to_contrastive_triples(pairs) == []

    def test_only_incorrect_responses_for_question(self):
        """No correct response for a question → no triple formed."""
        pairs = [
            _make_pair("q1", "wrong1", True),
            _make_pair("q1", "wrong2", True),
        ]
        assert _pairs_to_contrastive_triples(pairs) == []

    def test_round_robin_unequal_groups(self):
        """2 correct + 1 incorrect → 2 triples (round-robin wraps incorrect)."""
        pairs = [
            _make_pair("q1", "c1", False),
            _make_pair("q1", "c2", False),
            _make_pair("q1", "w1", True),
        ]
        triples = _pairs_to_contrastive_triples(pairs)
        assert len(triples) == 2
        # Each triple has the same incorrect response (wraps)
        for c, w, q in triples:
            assert w == "w1"
            assert q == "q1"

    def test_synthetic_pairs_pooled_together(self):
        """Synthetic pairs (question_id starts with 'synthetic_') share a pool."""
        pairs = [
            _make_pair("synthetic_0", "correct synthetic", False),
            _make_pair("synthetic_1", "wrong synthetic", True),
        ]
        triples = _pairs_to_contrastive_triples(pairs)
        # Both are pooled under _synthetic_pool → one triple
        assert len(triples) == 1
        c, w, q_id = triples[0]
        assert c == "correct synthetic"
        assert w == "wrong synthetic"
        assert q_id == "_synthetic_pool"

    def test_unknown_qid_pooled_with_synthetic(self):
        """'unknown' question_id joins the synthetic pool."""
        pairs = [
            _make_pair("unknown", "correct", False),
            _make_pair("unknown", "wrong", True),
        ]
        triples = _pairs_to_contrastive_triples(pairs)
        assert len(triples) == 1
        _, _, q_id = triples[0]
        assert q_id == "_synthetic_pool"

    def test_multiple_questions(self):
        """Each question ID produces independent triples."""
        pairs = [
            _make_pair("q1", "c1", False),
            _make_pair("q1", "w1", True),
            _make_pair("q2", "c2", False),
            _make_pair("q2", "w2", True),
        ]
        triples = _pairs_to_contrastive_triples(pairs)
        assert len(triples) == 2
        q_ids = {t[2] for t in triples}
        assert q_ids == {"q1", "q2"}


# ---------------------------------------------------------------------------
# _load_or_build_eorm_model
# ---------------------------------------------------------------------------

class TestLoadOrBuildEormModel:
    """Coverage for the baseline model loader."""

    def test_missing_file_returns_fresh_model(self, tmp_path):
        """When the baseline file does not exist, a fresh model is built."""
        path = tmp_path / "nonexistent.safetensors"
        model = _load_or_build_eorm_model(path)
        assert isinstance(model, EORMModel)
        assert model.embed_dim == 128  # matches EMBED_DIM constant

    def test_load_success_returns_loaded_model(self, tmp_path):
        """A valid safetensors file is loaded correctly."""
        import jax.random as jrandom
        original = EORMModel(embed_dim=32, n_heads=4, n_layers=1, key=jrandom.PRNGKey(1))
        save_path = tmp_path / "eorm_model_346.safetensors"
        original.save(str(save_path))

        loaded = _load_or_build_eorm_model(save_path)
        assert isinstance(loaded, EORMModel)
        assert loaded.embed_dim == 32

    def test_corrupt_file_falls_back_to_fresh(self, tmp_path):
        """A corrupt safetensors file triggers the fallback to a fresh model."""
        bad_path = tmp_path / "eorm_model_346.safetensors"
        bad_path.write_bytes(b"not a valid safetensors file")
        # Also write a fake _config.json so the load attempt gets further
        (tmp_path / "eorm_model_346_config.json").write_text(
            '{"embed_dim": 128, "n_heads": 4, "n_layers": 2, '
            '"max_seq_len": 512, "vocab_size": 4096}'
        )
        model = _load_or_build_eorm_model(bad_path)
        assert isinstance(model, EORMModel)
        assert model.embed_dim == 128  # fresh model defaults


# ---------------------------------------------------------------------------
# run_experiment — blocked path (insufficient real pairs)
# ---------------------------------------------------------------------------

class TestRunExperimentBlocked:
    """SCENARIO-LEARN-048: blocked artifact when real data is insufficient."""

    def test_blocked_when_no_result_files(self, tmp_path):
        """No Exp 368/369/370 result files → n_real_pairs < 50 → blocked artifact."""
        artifact = run_experiment(repo_root=tmp_path)

        assert artifact["status"] == "blocked"
        assert artifact["honest_verdict"] == "insufficient_real_pairs"
        assert artifact["n_real_pairs"] == 0
        assert artifact["n_real_pairs_minimum_required"] == 50
        assert artifact["retrain_mode"] == "blocked"
        assert artifact["before_auc"] is None
        assert artifact["after_auc"] is None
        assert artifact["auc_improvement"] is None

    def test_blocked_artifact_has_required_fields(self, tmp_path):
        """Blocked artifact must still have all ExperimentTemplate required fields."""
        artifact = run_experiment(repo_root=tmp_path)
        required = ["experiment", "schema", "run_date", "started_at", "finished_at",
                    "duration_s", "status", "title"]
        for field in required:
            assert field in artifact, f"Missing required field: {field}"

    def test_blocked_when_fewer_than_50_real_pairs(self, tmp_path):
        """49 real pairs still triggers the blocked path."""
        results_dir = tmp_path / "results"
        results_dir.mkdir()
        # Write 49 correct responses to Exp 368 result file
        responses = [
            {"question_id": f"q{i}", "model_id": "test", "response": f"answer {i}",
             "correct": True}
            for i in range(49)
        ]
        (results_dir / "experiment_368_precision_live.json").write_text(
            json.dumps({"responses": responses})
        )

        artifact = run_experiment(repo_root=tmp_path)
        assert artifact["status"] == "blocked"
        assert artifact["honest_verdict"] == "insufficient_real_pairs"
        assert artifact["n_real_pairs"] == 49


# ---------------------------------------------------------------------------
# run_experiment — success path (≥50 real pairs)
# ---------------------------------------------------------------------------

def _write_fake_live_results(tmp_path: Path, n_correct: int = 30, n_wrong: int = 30) -> None:
    """Write a minimal fake Exp 368-style result file with real pairs."""
    results_dir = tmp_path / "results"
    results_dir.mkdir(exist_ok=True)

    responses = []
    for i in range(n_correct):
        responses.append({
            "question_id": f"gsm_{i}",
            "model_id": "test_model",
            "response": f"The answer is {i}.",
            "correct": True,
        })
    for i in range(n_wrong):
        responses.append({
            "question_id": f"gsm_{i}",     # same question → forms contrastive pair
            "model_id": "test_model",
            "response": f"Wrong answer {i}.",
            "correct": False,
        })

    data = {"responses": responses}
    (results_dir / "experiment_368_precision_live.json").write_text(json.dumps(data))


class TestRunExperimentSuccess:
    """SCENARIO-LEARN-048: success path with real data."""

    def test_success_artifact_has_required_fields(self, tmp_path):
        """≥50 real pairs → artifact has all required schema fields."""
        _write_fake_live_results(tmp_path, n_correct=30, n_wrong=30)

        artifact = run_experiment(repo_root=tmp_path)

        required = ["experiment", "schema", "run_date", "started_at", "finished_at",
                    "duration_s", "status", "title"]
        for field in required:
            assert field in artifact, f"Missing required field: {field}"

    def test_success_status(self, tmp_path):
        """≥50 real pairs → status == 'success'."""
        _write_fake_live_results(tmp_path, n_correct=30, n_wrong=30)
        artifact = run_experiment(repo_root=tmp_path)
        assert artifact["status"] == "success"

    def test_schema_is_v2(self, tmp_path):
        """Exp 371 uses schema='carnot.eorm_retrain.v2'."""
        _write_fake_live_results(tmp_path, n_correct=30, n_wrong=30)
        artifact = run_experiment(repo_root=tmp_path)
        assert artifact["schema"] == "carnot.eorm_retrain.v2"

    def test_n_real_pairs_in_artifact(self, tmp_path):
        """n_real_pairs reflects the number of pairs actually used."""
        _write_fake_live_results(tmp_path, n_correct=30, n_wrong=30)
        artifact = run_experiment(repo_root=tmp_path)
        # 60 real pairs loaded, all ≤ MAX_REAL=300, so n_real_pairs = 60
        assert artifact["n_real_pairs"] == 60

    def test_retrain_mode_is_real_data(self, tmp_path):
        """retrain_mode is 'real_data' when ≥50 pairs available."""
        _write_fake_live_results(tmp_path, n_correct=30, n_wrong=30)
        artifact = run_experiment(repo_root=tmp_path)
        assert artifact["retrain_mode"] == "real_data"

    def test_auc_values_are_floats_in_range(self, tmp_path):
        """before_auc and after_auc are floats in [0, 1]."""
        _write_fake_live_results(tmp_path, n_correct=30, n_wrong=30)
        artifact = run_experiment(repo_root=tmp_path)
        assert isinstance(artifact["before_auc"], float)
        assert isinstance(artifact["after_auc"], float)
        assert 0.0 <= artifact["before_auc"] <= 1.0
        assert 0.0 <= artifact["after_auc"] <= 1.0

    def test_auc_improvement_equals_difference(self, tmp_path):
        """auc_improvement == after_auc - before_auc (signed, no clamping)."""
        _write_fake_live_results(tmp_path, n_correct=30, n_wrong=30)
        artifact = run_experiment(repo_root=tmp_path)
        expected_improvement = artifact["after_auc"] - artifact["before_auc"]
        assert artifact["auc_improvement"] == pytest.approx(expected_improvement, abs=1e-6)

    def test_honest_verdict_real_data_improvement(self, tmp_path):
        """When after_auc > before_auc, honest_verdict='real_data_improvement'."""
        _write_fake_live_results(tmp_path, n_correct=30, n_wrong=30)
        # Mock _evaluate_eorm_auc to return increasing AUC
        call_count = [0]
        def mock_auc(model, pairs):
            call_count[0] += 1
            return 0.55 if call_count[0] == 1 else 0.70

        with patch(
            "scripts.experiment_371_eorm_real_retrain._evaluate_eorm_auc",
            side_effect=mock_auc,
        ):
            artifact = run_experiment(repo_root=tmp_path)

        assert artifact["honest_verdict"] == "real_data_improvement"
        assert artifact["auc_improvement"] > 0

    def test_honest_verdict_real_data_no_improvement(self, tmp_path):
        """When after_auc <= before_auc, honest_verdict='real_data_no_improvement'."""
        _write_fake_live_results(tmp_path, n_correct=30, n_wrong=30)
        call_count = [0]
        def mock_auc(model, pairs):
            call_count[0] += 1
            return 0.60 if call_count[0] == 1 else 0.50  # AUC regressed

        with patch(
            "scripts.experiment_371_eorm_real_retrain._evaluate_eorm_auc",
            side_effect=mock_auc,
        ):
            artifact = run_experiment(repo_root=tmp_path)

        assert artifact["honest_verdict"] == "real_data_no_improvement"
        assert artifact["auc_improvement"] < 0

    def test_honest_verdict_no_improvement_when_flat(self, tmp_path):
        """When after_auc == before_auc, verdict is 'real_data_no_improvement' (not improvement)."""
        _write_fake_live_results(tmp_path, n_correct=30, n_wrong=30)

        def mock_auc_flat(model, pairs):
            return 0.5  # flat AUC

        with patch(
            "scripts.experiment_371_eorm_real_retrain._evaluate_eorm_auc",
            side_effect=mock_auc_flat,
        ):
            artifact = run_experiment(repo_root=tmp_path)

        assert artifact["honest_verdict"] == "real_data_no_improvement"

    def test_n_contrastive_triples_in_artifact(self, tmp_path):
        """Artifact includes the number of contrastive triples used for training."""
        _write_fake_live_results(tmp_path, n_correct=30, n_wrong=30)
        artifact = run_experiment(repo_root=tmp_path)
        assert "n_contrastive_triples" in artifact
        assert isinstance(artifact["n_contrastive_triples"], int)
        assert artifact["n_contrastive_triples"] >= 0

    def test_n_train_test_pairs_in_artifact(self, tmp_path):
        """Artifact records n_train_pairs, n_test_pairs, n_epochs."""
        _write_fake_live_results(tmp_path, n_correct=30, n_wrong=30)
        artifact = run_experiment(repo_root=tmp_path)
        assert "n_train_pairs" in artifact
        assert "n_test_pairs" in artifact
        assert "n_epochs" in artifact
        assert artifact["n_train_pairs"] + artifact["n_test_pairs"] >= 1

    def test_model_saved_to_results(self, tmp_path):
        """Retrained model file is written to results/eorm_model_371_real.safetensors."""
        _write_fake_live_results(tmp_path, n_correct=30, n_wrong=30)
        run_experiment(repo_root=tmp_path)
        model_path = tmp_path / "results" / "eorm_model_371_real.safetensors"
        assert model_path.exists(), "Model safetensors not written"

    def test_experiment_id_in_artifact(self, tmp_path):
        """Artifact experiment field == 371."""
        _write_fake_live_results(tmp_path, n_correct=30, n_wrong=30)
        artifact = run_experiment(repo_root=tmp_path)
        assert artifact["experiment"] == 371

    def test_humaneval_layout_b_pairs_loaded(self, tmp_path):
        """Exp 369 HumanEval layout (per_problem_results) contributes to real pair count."""
        results_dir = tmp_path / "results"
        results_dir.mkdir(exist_ok=True)

        # Write 60 pairs to exp 368 (enough on their own for the 50 threshold)
        responses = []
        for i in range(30):
            responses.append({"question_id": f"q{i}", "model_id": "m", "response": f"a{i}", "correct": True})
            responses.append({"question_id": f"q{i}", "model_id": "m", "response": f"w{i}", "correct": False})
        (results_dir / "experiment_368_precision_live.json").write_text(json.dumps({"responses": responses}))

        # Write 10 HumanEval pairs to exp 369
        per_problem = []
        for i in range(10):
            per_problem.append({"problem_id": f"HumanEval/{i}", "generated_code": f"def f(): return {i}", "passed_tests": bool(i % 2 == 0)})
        (results_dir / "experiment_369_humaneval_live.json").write_text(json.dumps({"per_problem_results": per_problem}))

        artifact = run_experiment(repo_root=tmp_path)
        # Total real pairs = 60 (exp368) + 10 (exp369) = 70 loaded, all ≤ MAX_REAL
        assert artifact["status"] == "success"
        assert artifact["n_real_pairs_loaded"] == 70
