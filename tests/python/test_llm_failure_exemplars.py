"""Tests for Exp 1112 LLM failure exemplar corpus.

Spec: REQ-VERIFY-1112, SCENARIO-VERIFY-1112
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path


_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_PYTHON_DIR = _PROJECT_ROOT / "python"
if str(_PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(_PYTHON_DIR))

for _pkg in ["carnot", "carnot.eval", "carnot.verify", "carnot.models"]:
    if _pkg not in sys.modules:
        _m = types.ModuleType(_pkg)
        _m.__path__ = [str(_PYTHON_DIR / _pkg.replace(".", "/"))]  # type: ignore[attr-defined]
        _m.__package__ = _pkg
        sys.modules[_pkg] = _m

from carnot.eval.llm_failure_exemplars import (  # noqa: E402
    REQUIRED_CATEGORY_MINIMUMS,
    build_exemplars,
    run_experiment,
    write_jsonl,
)
from carnot.verify.z3_math_verifier import Z3MathVerifier  # noqa: E402


def test_build_exemplars_covers_required_categories():
    """REQ-VERIFY-1112: corpus has >=30 rows and >=3 per required category."""
    exemplars = build_exemplars()
    assert len(exemplars) >= 30

    category_counts: dict[str, int] = {}
    for exemplar in exemplars:
        category_counts[exemplar["category"]] = category_counts.get(exemplar["category"], 0) + 1
        assert {
            "id",
            "category",
            "source",
            "prompt",
            "buggy_response",
            "correct_response",
            "mechanistic_root_cause",
            "carnot_energy_score",
            "carnot_verdict",
            "carnot_tier_detected",
        } <= set(exemplar)

    assert len(category_counts) >= 10
    for category, minimum in REQUIRED_CATEGORY_MINIMUMS.items():
        assert category_counts.get(category, 0) >= minimum


def test_z3_math_verifier_catches_decimal_comparison_error():
    """REQ-VERIFY-1107 / REQ-VERIFY-1112: 9.11 > 9.9 is a detected violation."""
    verifier = Z3MathVerifier()
    assert verifier.score("9.11 is larger than 9.9 because 11 > 9.") > 0.0
    assert verifier.score("9.9 is larger than 9.11 because 0.9 > 0.11.") == 0.0


def test_write_jsonl_writes_one_object_per_line(tmp_path: Path):
    """REQ-VERIFY-1112: exemplar corpus is JSONL, not a JSON array."""
    path = tmp_path / "exemplars.jsonl"
    rows = build_exemplars()[:3]
    write_jsonl(path, rows)

    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 3
    assert [json.loads(line)["id"] for line in lines] == [
        "exemplar_001",
        "exemplar_002",
        "exemplar_003",
    ]


def test_run_experiment_writes_required_artifact_fields(tmp_path: Path):
    """SCENARIO-VERIFY-1112: experiment writes corpus and result schema."""
    exemplar_path = tmp_path / "llm_failure_exemplars.jsonl"
    result_path = tmp_path / "experiment_1112.json"

    artifact = run_experiment(
        exemplar_path=exemplar_path,
        result_path=result_path,
        sos_epochs=5,
    )

    assert exemplar_path.exists()
    assert result_path.exists()
    assert artifact["n_exemplars"] >= 30
    assert artifact["n_categories"] >= 10
    assert artifact["exemplar_path"] == "data/llm_failure_exemplars.jsonl"
    assert artifact["llm_failure_exemplar_corpus_30_exemplars"] is True
    assert artifact["goodfire_cascade_tp_rate_measured"] is True
    assert artifact["positioning_note_written"] is True
    assert artifact["mathematical_objective_tier_tp_rate"] >= 0.9
    assert "SOSKANEnergyV3" in artifact["tier_tp_rates"]
    assert artifact["learned_tier_tp_rate"] == artifact["tier_tp_rates"]["SOSKANEnergyV3"]
    assert artifact["honest_verdict"] in {
        "corpus_complete_high_tp",
        "corpus_complete_low_tp",
        "corpus_partial",
        "failed",
    }

    scored_rows = [
        json.loads(line) for line in exemplar_path.read_text(encoding="utf-8").splitlines()
    ]
    assert all(row["carnot_energy_score"] is not None for row in scored_rows)
    assert any(row["carnot_tier_detected"] == "Z3MathVerifier" for row in scored_rows)
