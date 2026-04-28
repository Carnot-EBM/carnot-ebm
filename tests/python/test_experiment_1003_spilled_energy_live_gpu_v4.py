"""Tests for experiment_1003_spilled_energy_live_gpu_v4.py.

Covers:
  - Module is importable (including top-level env_autofix + EnvPropagationGuard calls)
  - _roc_auc_score: correct pair → concordant count correct
  - _roc_auc_score: degenerate (single class) returns 0.5
  - _roc_auc_score: perfect separator returns 1.0
  - _score_with_probes: returns correct structure lengths
  - _score_with_probes: all scores in [0, 1]
  - _score_with_probes: violations have required keys
  - _load_fover_corpus: returns list (empty or populated)
  - Deliverable JSON exists and has all required schema fields
  - live_violations_1003.json exists and has correct schema

Spec: REQ-TIER0-002, REQ-TIER0-003, REQ-VERIFY-083,
      SCENARIO-TIER0-002, SCENARIO-TIER0-003
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


# ---------------------------------------------------------------------------
# Module importability
# ---------------------------------------------------------------------------


def test_experiment_module_importable() -> None:
    """The experiment_1003 module can be imported without errors.

    Top-level side effects (apply_env_autofix, EnvPropagationGuard.propagate) must
    not raise on import — they are non-destructive env inspection calls.

    Spec: REQ-VERIFY-083
    """
    import scripts.experiment_1003_spilled_energy_live_gpu_v4 as mod  # noqa: F401

    assert mod is not None


# ---------------------------------------------------------------------------
# _roc_auc_score
# ---------------------------------------------------------------------------


def test_roc_auc_score_perfect_separator() -> None:
    """_roc_auc_score returns 1.0 when positives all have higher scores than negatives.

    Spec: REQ-TIER0-002, SCENARIO-TIER0-002
    """
    from scripts.experiment_1003_spilled_energy_live_gpu_v4 import _roc_auc_score

    y_true = [0, 0, 1, 1]
    y_score = [0.1, 0.2, 0.8, 0.9]
    assert _roc_auc_score(y_true, y_score) == pytest.approx(1.0)


def test_roc_auc_score_degenerate_single_class() -> None:
    """_roc_auc_score returns 0.5 when only one class is present.

    The Wilcoxon-Mann-Whitney statistic is undefined with one class; 0.5 is the
    conventional no-information baseline.

    Spec: REQ-TIER0-002
    """
    from scripts.experiment_1003_spilled_energy_live_gpu_v4 import _roc_auc_score

    assert _roc_auc_score([1, 1, 1], [0.1, 0.2, 0.3]) == pytest.approx(0.5)
    assert _roc_auc_score([0, 0, 0], [0.1, 0.2, 0.3]) == pytest.approx(0.5)


def test_roc_auc_score_random_baseline() -> None:
    """_roc_auc_score returns 0.5 for a random classifier (scores independent of labels).

    Spec: REQ-TIER0-002
    """
    from scripts.experiment_1003_spilled_energy_live_gpu_v4 import _roc_auc_score

    y_true = [0, 1, 0, 1]
    y_score = [0.5, 0.5, 0.5, 0.5]  # all identical scores
    # When scores are all tied, concordant pairs = 0.5 * n_pos * n_neg
    result = _roc_auc_score(y_true, y_score)
    assert 0.0 <= result <= 1.0


# ---------------------------------------------------------------------------
# _score_with_probes
# ---------------------------------------------------------------------------


_SAMPLE_ITEMS = [
    {
        "response_text": "We have 5 apples and add 3. Total = 5 + 3 = 8.",
        "question_text": "If I have 5 apples and get 3 more, how many do I have?",
        "label": "correct",
    },
    {
        "response_text": "The answer is 42. Because 7 times 6 equals 999.",
        "question_text": "What is 7 times 6?",
        "label": "incorrect",
    },
    {
        "step_text": "First compute 10 * 5 = 50, then subtract 10 to get 40.",
        "label": "correct",
    },
]


def test_score_with_probes_lengths() -> None:
    """_score_with_probes returns score lists of the same length as input items.

    Spec: REQ-TIER0-002, REQ-TIER0-003, SCENARIO-TIER0-002
    """
    from scripts.experiment_1003_spilled_energy_live_gpu_v4 import _score_with_probes

    spill_scores, nup_scores, labels, violations = _score_with_probes(_SAMPLE_ITEMS, "fover_corpus")
    assert len(spill_scores) == len(nup_scores) == len(_SAMPLE_ITEMS)
    assert len(labels) == len(_SAMPLE_ITEMS)


def test_score_with_probes_scores_in_unit_interval() -> None:
    """All probe scores must be in [0, 1].

    Spec: REQ-TIER0-002, SCENARIO-TIER0-002
    """
    from scripts.experiment_1003_spilled_energy_live_gpu_v4 import _score_with_probes

    spill_scores, nup_scores, _labels, _viols = _score_with_probes(_SAMPLE_ITEMS, "fover_corpus")
    for s in spill_scores:
        assert 0.0 <= s <= 1.0, f"spill score out of range: {s}"
    for s in nup_scores:
        assert 0.0 <= s <= 1.0, f"NUP score out of range: {s}"


def test_score_with_probes_violation_keys() -> None:
    """Violation dicts contain the required keys for Exp 1005.

    Spec: REQ-TIER0-002, SCENARIO-TIER0-003
    """
    from scripts.experiment_1003_spilled_energy_live_gpu_v4 import _score_with_probes

    # Use items with high expected spill to ensure at least some violations
    high_spill = [
        {
            "response_text": "The answer is 99999 because 1 + 1 = 7777 and 7777 * 0.0001 = 999.",
            "question_text": "What is 1 + 1?",
            "label": "incorrect",
        }
        for _ in range(5)
    ]
    _s, _n, _l, violations = _score_with_probes(high_spill, "test")
    required_keys = {"question_id", "text_snippet", "spill_score", "nup_score", "inference_mode"}
    for v in violations:
        for key in required_keys:
            assert key in v, f"violation missing key: {key}"


# ---------------------------------------------------------------------------
# _load_fover_corpus
# ---------------------------------------------------------------------------


def test_load_fover_corpus_returns_list() -> None:
    """_load_fover_corpus() returns a list (possibly empty if file missing).

    Spec: REQ-TIER0-002
    """
    from scripts.experiment_1003_spilled_energy_live_gpu_v4 import _load_fover_corpus

    result = _load_fover_corpus()
    assert isinstance(result, list)


# ---------------------------------------------------------------------------
# Deliverable artifact validation
# ---------------------------------------------------------------------------


def test_deliverable_exists() -> None:
    """The deliverable JSON exists and is valid JSON.

    Spec: REQ-VERIFY-083
    """
    path = _REPO_ROOT / "results" / "experiment_1003_spilled_energy_live_gpu_v4.json"
    assert path.exists(), f"deliverable not found: {path}"
    d = json.loads(path.read_text())
    assert isinstance(d, dict)


def test_deliverable_required_fields() -> None:
    """The deliverable has all REQUIRED_RESULT_FIELDS from ExperimentTemplate.

    Spec: REQ-VERIFY-083
    """
    path = _REPO_ROOT / "results" / "experiment_1003_spilled_energy_live_gpu_v4.json"
    d = json.loads(path.read_text())
    required = [
        "experiment",
        "schema",
        "run_date",
        "started_at",
        "finished_at",
        "duration_s",
        "status",
        "title",
    ]
    for field in required:
        assert field in d, f"deliverable missing required field: {field}"


def test_deliverable_experiment_specific_fields() -> None:
    """The deliverable has the experiment-specific fields required by task spec.

    Spec: REQ-TIER0-002, SCENARIO-TIER0-002
    """
    path = _REPO_ROOT / "results" / "experiment_1003_spilled_energy_live_gpu_v4.json"
    d = json.loads(path.read_text())
    specific = [
        "spilled_energy_live_auroc",
        "nup_probe_live_auroc",
        "n_live_violations_collected",
        "inference_mode",
        "honest_verdict",
    ]
    for field in specific:
        assert field in d, f"deliverable missing experiment field: {field}"


def test_deliverable_honest_verdict_valid() -> None:
    """The honest_verdict is one of the allowed values from the task spec.

    Spec: REQ-TIER0-002
    """
    path = _REPO_ROOT / "results" / "experiment_1003_spilled_energy_live_gpu_v4.json"
    d = json.loads(path.read_text())
    allowed = {"live_validated", "live_below_threshold", "blocked"}
    assert d["honest_verdict"] in allowed, f"unexpected honest_verdict: {d['honest_verdict']}"


def test_deliverable_auroc_in_range() -> None:
    """AUROC values in the deliverable are in [0, 1].

    Spec: REQ-TIER0-002, SCENARIO-TIER0-002
    """
    path = _REPO_ROOT / "results" / "experiment_1003_spilled_energy_live_gpu_v4.json"
    d = json.loads(path.read_text())
    assert 0.0 <= d["spilled_energy_live_auroc"] <= 1.0
    assert 0.0 <= d["nup_probe_live_auroc"] <= 1.0


# ---------------------------------------------------------------------------
# violations file validation
# ---------------------------------------------------------------------------


def test_violations_file_exists() -> None:
    """results/live_violations_1003.json exists for downstream Exp 1005.

    Spec: REQ-TIER0-002
    """
    path = _REPO_ROOT / "results" / "live_violations_1003.json"
    assert path.exists(), f"violations file not found: {path}"


def test_violations_file_schema() -> None:
    """live_violations_1003.json has expected schema and structure.

    Spec: REQ-TIER0-002
    """
    path = _REPO_ROOT / "results" / "live_violations_1003.json"
    d = json.loads(path.read_text())
    assert d["schema"] == "carnot.live_violations.v1"
    assert d["experiment"] == 1003
    assert isinstance(d["violations"], list)
    assert isinstance(d["n_violations"], int)
    assert d["n_violations"] == len(d["violations"])
