"""Tests for Exp 1176 k=6 AND-composition validation.

Spec: REQ-VERIFY-1176, SCENARIO-VERIFY-1176
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest


_PROJECT_ROOT = Path(__file__).parent.parent.parent
_PYTHON_DIR = _PROJECT_ROOT / "python"
if str(_PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(_PYTHON_DIR))

for _pkg in ["carnot", "carnot.eval", "carnot.verify", "carnot.models", "carnot.pipeline"]:
    if _pkg not in sys.modules:
        _m = types.ModuleType(_pkg)
        _m.__path__ = [str(_PYTHON_DIR / _pkg.replace(".", "/"))]  # type: ignore[attr-defined]
        _m.__package__ = _pkg
        sys.modules[_pkg] = _m


import carnot.eval.k6_and_compose_validation as exp1176  # noqa: E402


class FakeSCVerifier:
    def score(self, response: str, context: str = "") -> float:
        del context
        if "wrong" in response:
            return 0.9
        if "borderline" in response:
            return 0.55
        return 0.1


class FakeAdapter:
    def __init__(self, name: str, wrong_score: float, clean_score: float = 0.1) -> None:
        self._name = name
        self._wrong_score = wrong_score
        self._clean_score = clean_score

    @property
    def name(self) -> str:
        return self._name

    def score(self, text: str) -> float:
        if "wrong" in text:
            return self._wrong_score
        if "borderline" in text:
            return 0.45
        return self._clean_score


def _write_json(path: Path, payload: object) -> Path:
    path.write_text(json.dumps(payload))
    return path


def test_load_rows_supports_json_collections_jsonl_and_rejects_bad_shapes(
    tmp_path: Path,
) -> None:
    """REQ-VERIFY-1176-2: FoVer loaders accept the corpus shapes used by experiments."""
    list_path = _write_json(tmp_path / "rows.json", [{"label": "correct"}, "skip"])
    dict_path = _write_json(tmp_path / "pairs.json", {"pairs": [{"label": "incorrect"}]})
    jsonl_path = tmp_path / "rows.jsonl"
    jsonl_path.write_text(json.dumps({"label": "correct"}) + "\n\n" + json.dumps(["skip"]))
    bad_path = _write_json(tmp_path / "bad.json", {"not_rows": True})

    assert exp1176.load_rows(list_path) == [{"label": "correct"}]
    assert exp1176.load_rows(dict_path) == [{"label": "incorrect"}]
    assert exp1176.load_rows(jsonl_path) == [{"label": "correct"}]
    with pytest.raises(ValueError, match="unsupported row payload"):
        exp1176.load_rows(bad_path)


def test_row_helpers_cover_alternate_label_and_text_fields() -> None:
    """REQ-VERIFY-1176-2: row normalization handles held-out FoVer schema variants."""
    assert exp1176.is_incorrect({"is_correct": False}) is True
    assert exp1176.is_incorrect({"step_correct": False}) is True
    assert exp1176.is_incorrect({"label": False}) is True
    assert exp1176.is_incorrect({"label": "correct"}) is False
    assert exp1176.is_incorrect({}) is False
    assert exp1176.row_text({"step": "step text"}) == "step text"
    assert exp1176.row_context({"prompt": "prompt text"}) == "prompt text"


def test_select_heldout_eval_rows_uses_exact_200_and_keeps_incorrect_examples() -> None:
    """REQ-VERIFY-1176-2: evaluation selection is held-out and exactly 200 examples."""
    rows = [
        {"step_text": f"correct {idx}", "label": "correct", "question_id": f"c{idx}"}
        for idx in range(210)
    ]
    rows.extend(
        {"step_text": f"wrong {idx}", "label": "incorrect", "question_id": f"w{idx}"}
        for idx in range(8)
    )

    selected = exp1176.select_heldout_eval_rows(rows, n_examples=200, seed=1176)

    assert len(selected) == 200
    assert sum(1 for row in selected if exp1176.is_incorrect(row)) == 8
    assert {row["question_id"] for row in selected}.issubset({row["question_id"] for row in rows})


def test_select_heldout_eval_rows_adjusts_when_correct_rows_are_rare() -> None:
    """REQ-VERIFY-1176-2: selection preserves exact size when one class is scarce."""
    rows = [{"step_text": f"correct {idx}", "label": "correct"} for idx in range(10)]
    rows.extend({"step_text": f"wrong {idx}", "label": "incorrect"} for idx in range(210))

    selected = exp1176.select_heldout_eval_rows(rows, n_examples=200, seed=2)

    assert len(selected) == 200
    assert sum(1 for row in selected if not exp1176.is_incorrect(row)) == 10


def test_select_heldout_eval_rows_rejects_single_class_or_too_small_inputs() -> None:
    """REQ-VERIFY-1176-2: AUROC evaluation requires enough held-out class coverage."""
    with pytest.raises(ValueError, match="at least 200"):
        exp1176.select_heldout_eval_rows([{"label": "correct"}], n_examples=200)

    rows = [{"label": "correct", "step_text": str(idx)} for idx in range(200)]
    with pytest.raises(ValueError, match="both correct and incorrect"):
        exp1176.select_heldout_eval_rows(rows, n_examples=200)


def test_build_contrastive_pairs_groups_correct_context_with_incorrect_response() -> None:
    """REQ-VERIFY-1176-1: SC-Energy fallback can rebuild Exp1168 training pairs."""
    rows = [
        {"question_id": "q1", "step_text": "First fact.", "label": "correct"},
        {"question_id": "q1", "step_text": "Second fact.", "label": "correct"},
        {"question_id": "q1", "step_text": "wrong unrelated claim.", "label": "incorrect"},
        {"question_id": "q2", "step_text": "Only one correct.", "label": "correct"},
        {"question_id": "q2", "step_text": "wrong but no context.", "label": "incorrect"},
    ]

    pairs = exp1176.build_contrastive_pairs(rows)

    assert pairs == [
        {
            "qid": "q1",
            "coherent": ("Second fact.", "First fact."),
            "incoherent": ("wrong unrelated claim.", "First fact. Second fact."),
        }
    ]


def test_load_sc_energy_from_exp1168_uses_gate_metadata_and_training_fallback(
    tmp_path: Path,
) -> None:
    """REQ-VERIFY-1176-1: no-checkpoint Exp1168 artifacts retrain deterministically."""
    train_rows = [
        {"question_id": "q1", "step_text": "Alice starts with 2.", "label": "correct"},
        {"question_id": "q1", "step_text": "Alice gets 3 and has 5.", "label": "correct"},
        {"question_id": "q1", "step_text": "wrong train travels 90.", "label": "incorrect"},
        {"question_id": "q2", "step_text": "Bob starts with 4.", "label": "correct"},
        {"question_id": "q2", "step_text": "Bob gets 1 and has 5.", "label": "correct"},
        {"question_id": "q2", "step_text": "wrong soup needs water.", "label": "incorrect"},
    ]
    _write_json(tmp_path / "train.json", train_rows)
    artifact_path = _write_json(
        tmp_path / "experiment_1168_sc_energy_7th_verifier.json",
        {
            "sc_energy_auroc_above_threshold": True,
            "k6_viable": True,
            "model_name": "deterministic",
            "hidden_dim": 12,
            "fover_labeled_pairs_path": "train.json",
        },
    )

    verifier, source = exp1176.load_sc_energy_from_exp1168(
        artifact_path,
        project_root=tmp_path,
        n_epochs=1,
    )

    assert verifier.model_name == "deterministic"
    assert verifier.hidden_dim == 12
    assert source["checkpoint_loaded"] is False
    assert source["sc_energy_loading_mode"] == "deterministic_retrain_from_exp1168_artifact"
    assert source["n_contrastive_pairs"] == 2
    assert source["n_train_pairs"] == 1


def test_load_sc_energy_from_exp1168_rejects_failed_gate(tmp_path: Path) -> None:
    """REQ-VERIFY-1176-1: Exp1176 remains gated on a viable Exp1168 result."""
    artifact_path = _write_json(
        tmp_path / "experiment_1168_sc_energy_7th_verifier.json",
        {"sc_energy_auroc_above_threshold": False, "k6_viable": False},
    )

    with pytest.raises(RuntimeError, match="Exp 1168 gate is not satisfied"):
        exp1176.load_sc_energy_from_exp1168(artifact_path, project_root=tmp_path)


def test_build_fixed_k5_verifiers_preserves_production_member_names(tmp_path: Path) -> None:
    """REQ-VERIFY-1176-3: k=5 members match the repaired production ensemble."""
    corpus_rows = [
        {"step_text": "2 + 2 = 4.", "label": "correct"},
        {"step_text": "3 + 3 = 6.", "label": "correct"},
        {"step_text": "wrong 2 + 2 = 5.", "label": "incorrect"},
        {"step_text": "wrong 3 + 3 = 7.", "label": "incorrect"},
    ]
    corpus_path = _write_json(tmp_path / "fover_corpus_v4.json", corpus_rows)

    verifiers = exp1176.build_fixed_k5_verifiers(
        corpus_path,
        n_correct=2,
        n_wrong=2,
        n_epochs=0,
    )

    assert [verifier.name for verifier in verifiers] == exp1176.K5_VERIFIER_NAMES
    assert all(isinstance(verifier.score("2 + 2 = 4."), float) for verifier in verifiers)


def test_score_eval_rows_and_metrics_use_max_energy_and_pairwise_correlations() -> None:
    """REQ-VERIFY-1176-3/4: k=6 energy is max(E1..E6) and reports SC correlations."""
    rows = [
        {"question": "q", "response": "clean answer", "label": "correct"},
        {"question": "q", "response": "borderline answer", "label": "correct"},
        {"question": "q", "response": "wrong answer one", "label": "incorrect"},
        {"question": "q", "response": "wrong answer two", "label": "incorrect"},
    ]
    verifiers = [
        FakeAdapter("SOSKANEnergyV3", wrong_score=0.6),
        FakeAdapter("SemEnergyProbe", wrong_score=0.7),
        FakeAdapter("ASTStructureVerifier", wrong_score=0.2),
        FakeAdapter("SemanticConsistencyVerifier", wrong_score=0.3),
        FakeAdapter("Z3MathVerifier", wrong_score=0.8),
    ]

    scores = exp1176.score_eval_rows(rows, FakeSCVerifier(), verifiers)
    metrics = exp1176.compute_validation_metrics(scores)

    assert scores.labels == [0, 0, 1, 1]
    assert scores.k5_scores == [0.1, 0.45, 0.8, 0.8]
    assert scores.k6_scores == [0.1, 0.55, 0.9, 0.9]
    assert metrics.k6_auroc == 1.0
    assert metrics.k6_above_k5 is True
    assert metrics.sc_energy_marginal_gain == pytest.approx(0.0598)
    assert set(metrics.sc_energy_r_corr_on_eval) == set(exp1176.K5_VERIFIER_NAMES)


def test_tie_aware_auroc_and_pearson_zero_variance_paths() -> None:
    """REQ-VERIFY-1176-4: metric helpers handle ties and flat verifier scores."""
    assert exp1176.tie_aware_auroc([0, 1], [0.5, 0.5]) == 0.5
    assert exp1176.tie_aware_auroc([0, 0], [0.1, 0.2]) == 0.5
    assert exp1176.pearson_r([1.0, 1.0], [0.0, 1.0]) == 0.0
    assert exp1176.pearson_r([0.0, 1.0], [0.0]) == 0.0


def test_honest_verdict_covers_improvement_no_improvement_and_correlation() -> None:
    """REQ-VERIFY-1176-5: verdict enum is derived from gain and correlation."""
    assert exp1176.honest_verdict(0.95, {"a": 0.9}) == "k6_improves_over_k5"
    assert exp1176.honest_verdict(0.90, {"a": 0.1}) == "k6_no_improvement"
    assert exp1176.honest_verdict(0.90, {"a": 0.6}) == "k6_degrades_due_to_correlation"


def test_build_artifact_has_required_fields_and_follow_up_note() -> None:
    """SCENARIO-VERIFY-1176: artifact schema includes required fields and decision note."""
    scores = exp1176.ValidationScores(
        labels=[0, 0, 1, 1],
        sc_scores=[0.1, 0.2, 0.8, 0.9],
        existing_scores={
            "SOSKANEnergyV3": [0.1, 0.2, 0.7, 0.8],
            "SemEnergyProbe": [0.2, 0.1, 0.6, 0.7],
            "ASTStructureVerifier": [0.0, 0.1, 0.5, 0.6],
            "SemanticConsistencyVerifier": [0.2, 0.2, 0.4, 0.4],
            "Z3MathVerifier": [0.1, 0.3, 0.8, 0.9],
        },
        k5_scores=[0.2, 0.3, 0.8, 0.9],
        k6_scores=[0.2, 0.3, 0.8, 0.9],
    )
    metrics = exp1176.compute_validation_metrics(scores)

    artifact = exp1176.build_artifact(
        metrics,
        sc_source={"checkpoint_loaded": False},
        eval_corpus_path=Path("data/fover_test_v4.json"),
        started_at="2026-05-02T00:00:00Z",
        duration_s=1.25,
    )

    assert exp1176.REQUIRED_ARTIFACT_FIELDS <= set(artifact)
    assert artifact["k6_and_compose_auroc_measured"] is True
    assert artifact["honest_verdict"] == "k6_improves_over_k5"
    assert "follow-up" in artifact["decision_note"].lower()
    assert artifact["pipeline_modified"] is False


def test_decision_note_documents_correlation_and_non_correlation_failures() -> None:
    """REQ-VERIFY-1176-5: negative outcomes document the limiting condition."""
    degraded = exp1176.ValidationMetrics(
        k5_auroc_baseline=0.9402,
        k5_auroc_on_eval=0.9,
        k6_auroc=0.9,
        k6_above_k5=False,
        k6_and_compose_auroc_measured=True,
        sc_energy_r_corr_on_eval={"SOSKANEnergyV3": 0.6},
        sc_energy_marginal_gain=-0.0402,
        honest_verdict="k6_degrades_due_to_correlation",
        largest_sc_energy_overlap="SOSKANEnergyV3",
        max_abs_sc_energy_r_corr=0.6,
        n_eval_examples=200,
        n_correct=100,
        n_incorrect=100,
    )
    no_gain = exp1176.ValidationMetrics(
        k5_auroc_baseline=0.9402,
        k5_auroc_on_eval=0.9,
        k6_auroc=0.9,
        k6_above_k5=False,
        k6_and_compose_auroc_measured=True,
        sc_energy_r_corr_on_eval={"SOSKANEnergyV3": 0.1},
        sc_energy_marginal_gain=-0.0402,
        honest_verdict="k6_no_improvement",
        largest_sc_energy_overlap="SOSKANEnergyV3",
        max_abs_sc_energy_r_corr=0.1,
        n_eval_examples=200,
        n_correct=100,
        n_incorrect=100,
    )

    assert "largest SC-Energy overlap" in exp1176._decision_note(degraded)
    assert "below the 0.5" in exp1176._decision_note(no_gain)
    assert exp1176._follow_up_task(no_gain) == ""


def test_write_artifact_and_run_experiment_orchestrate_validation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """SCENARIO-VERIFY-1176: runner writes the requested JSON artifact."""
    output_path = tmp_path / "experiment_1176_k6_and_compose_validation.json"
    scores = exp1176.ValidationScores(
        labels=[0, 1],
        sc_scores=[0.1, 0.9],
        existing_scores={name: [0.1, 0.8] for name in exp1176.K5_VERIFIER_NAMES},
        k5_scores=[0.1, 0.8],
        k6_scores=[0.1, 0.9],
    )

    monkeypatch.setattr(
        exp1176,
        "load_sc_energy_from_exp1168",
        lambda *args, **kwargs: (FakeSCVerifier(), {"checkpoint_loaded": False}),
    )
    monkeypatch.setattr(exp1176, "load_rows", lambda path: [{"label": "correct"}] * 200)
    monkeypatch.setattr(
        exp1176,
        "select_heldout_eval_rows",
        lambda rows, n_examples=200, seed=1176: rows,
    )
    monkeypatch.setattr(
        exp1176,
        "build_fixed_k5_verifiers",
        lambda *args, **kwargs: [FakeAdapter(name, 0.8) for name in exp1176.K5_VERIFIER_NAMES],
    )
    monkeypatch.setattr(exp1176, "score_eval_rows", lambda *args, **kwargs: scores)

    artifact = exp1176.run_experiment(
        project_root=tmp_path,
        exp1168_path=tmp_path / "exp1168.json",
        eval_corpus_path=tmp_path / "eval.json",
        soskan_training_path=tmp_path / "train.json",
        output_path=output_path,
    )

    loaded = json.loads(output_path.read_text())
    assert loaded["schema"] == "k6_and_compose_validation"
    assert loaded["k6_auroc"] == artifact["k6_auroc"]
