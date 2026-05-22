"""Tests for Exp 2866 tiny exact BEAVER frontier feasibility.

Spec: REQ-VERIFY-2866, SCENARIO-VERIFY-2866.
"""

from __future__ import annotations

import ast
import json
from fractions import Fraction
from pathlib import Path
from typing import Any

import pytest

from carnot.verify import beaver_exact_tiny_frontier as exp
from carnot.verify.beaver_epr_bounded_probe import LabeledExample


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _proxy_artifact() -> dict[str, Any]:
    return {
        "honest_verdict": "complete: clean bounded-prefix/EPR proxy evaluated on local FoVer labels",
        "exact_beaver_implemented": False,
        "bounded_prefix_proxy_auc": 0.74,
        "n_examples": 100,
        "sample_rows": [
            {
                "example_id": "proxy-sample",
                "label": 1,
                "bounded_prefix_proxy_score": 1.0,
            }
        ],
    }


def _fover_rows(count_per_class: int = 3) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index in range(count_per_class):
        rows.append(
            {
                "question_id": f"correct-{index}",
                "step_text": f"Trace: {index} + 2 = {index + 2}. Then 3 * 4 = 12.",
                "label": "correct",
            }
        )
        rows.append(
            {
                "question_id": f"incorrect-{index}",
                "step_text": f"Trace: {index} + 2 = {index + 3}. Then 3 * 4 = 12.",
                "label": "incorrect",
            }
        )
    return rows


def test_req_verify_2866_solver_frontier_finds_first_false_completed_claim() -> None:
    """REQ-VERIFY-2866-3: Z3 proves the first completed false equality frontier."""

    text = "Trace: -2 + +5 = 3. Then 8 - 3 = 6. Finally 6 / 3 = 2."
    row = exp.score_example_exact_frontier(LabeledExample("bad", text, 1, "fixture"))

    assert row["exact_frontier_available"] is True
    assert row["solver_claim_count"] == 3
    assert row["exact_false_claim_count"] == 1
    assert row["exact_score"] == pytest.approx(1.0 / 3.0)
    assert row["first_exact_frontier_prefix_length"] == text.index("6.") + 1
    assert row["bounded_prefix_proxy_score"] == pytest.approx(1.0 / 3.0)
    assert row["exact_matches_proxy_decision"] is True


def test_scenario_verify_2866_writes_required_success_artifact(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-2866: a tiny exact subset is compared to Exp 2858 proxy."""

    _write_jsonl(tmp_path / "data" / "fover_corpus.jsonl", _fover_rows())
    _write_json(
        tmp_path / "results" / "experiment_2858_beaver_epr_clean_bounded_proxy_v2.json",
        _proxy_artifact(),
    )

    output_path = tmp_path / "custom_results" / exp.OUTPUT_FILENAME
    artifact = exp.run_experiment(
        exp.ExperimentConfig(
            repo_root=tmp_path,
            output_path=output_path,
            n_examples=4,
            started_at=10.0,
            clock=lambda: 12.5,
            tests_run=["focused-pytest"],
        )
    )

    saved = json.loads(output_path.read_text(encoding="utf-8"))
    assert saved == artifact
    assert set(exp.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["exact_beaver_implemented"] is False
    assert artifact["exact_frontier_available"] is True
    assert artifact["n_examples"] == 4
    assert str(artifact["solver_used"]).startswith("z3-solver ")
    assert artifact["blocked_reason"] is None
    assert artifact["random_seed"] == exp.RANDOM_SEED
    assert len(artifact["reproducibility_checksum"]) == 64
    assert artifact["duration_s"] == pytest.approx(2.5)
    assert artifact["tests_run"] == ["focused-pytest"]
    assert len(artifact["sample_rows"]) == 4
    assert artifact["exact_vs_proxy_comparison"]["exp2858_proxy_auc"] == pytest.approx(0.74)
    assert artifact["exact_vs_proxy_comparison"]["decision_agreement_rate"] == pytest.approx(1.0)
    assert artifact["exact_vs_proxy_comparison"]["exp2858_exact_beaver_implemented"] is False
    assert artifact["field_principles"]["exact_beaver_implemented"].startswith("false:")
    assert any(
        check["step"] == "test -f data/fover_corpus.jsonl" and check["passed"]
        for check in artifact["preconditions_checked"]
    )


def test_req_verify_2866_blocked_dependency_artifacts_are_honest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-VERIFY-2866-1/2: missing solver, proxy, or subset blocks honestly."""

    _write_jsonl(tmp_path / "data" / "fover_corpus.jsonl", _fover_rows())
    _write_json(
        tmp_path / "results" / "experiment_2858_beaver_epr_clean_bounded_proxy_v2.json",
        _proxy_artifact(),
    )

    monkeypatch.setattr(exp, "z3", None)
    solver_blocked = exp.run_experiment(
        exp.ExperimentConfig(repo_root=tmp_path, n_examples=4, started_at=1.0, clock=lambda: 1.25)
    )

    assert solver_blocked["honest_verdict"].startswith("blocked_dependency:")
    assert solver_blocked["blocked_reason"] == "blocked_dependency"
    assert solver_blocked["solver_used"] is None
    assert solver_blocked["exact_frontier_available"] is False
    assert solver_blocked["sample_rows"] == []

    monkeypatch.setattr(exp, "z3", exp._Z3_IMPORT)
    proxy_blocked = exp.run_experiment(
        exp.ExperimentConfig(
            repo_root=tmp_path / "without_proxy",
            n_examples=4,
            started_at=2.0,
            clock=lambda: 2.5,
        )
    )
    assert proxy_blocked["honest_verdict"].startswith("blocked_dependency:")
    assert proxy_blocked["blocked_reason"] == "blocked_dependency"
    assert proxy_blocked["n_examples"] == 0

    _write_json(tmp_path / "one_class" / "results" / exp.PROXY_ARTIFACT_PATH.name, _proxy_artifact())
    _write_jsonl(
        tmp_path / "one_class" / "data" / "fover_corpus.jsonl",
        [{"question_id": "only", "step_text": "1 + 1 = 2.", "label": "correct"}],
    )
    subset_blocked = exp.run_experiment(
        exp.ExperimentConfig(
            repo_root=tmp_path / "one_class",
            n_examples=2,
            started_at=3.0,
            clock=lambda: 4.0,
        )
    )
    assert subset_blocked["honest_verdict"].startswith("blocked_dependency:")
    assert subset_blocked["blocked_reason"] == "blocked_dependency"
    assert subset_blocked["exact_vs_proxy_comparison"]["comparison_status"] == "blocked"


def test_req_verify_2866_validation_and_solver_edges(tmp_path: Path) -> None:
    """REQ-VERIFY-2866-4/5: validation preserves exact/proxy claim boundaries."""

    _write_jsonl(tmp_path / "data" / "fover_corpus.jsonl", _fover_rows())
    _write_json(
        tmp_path / "results" / "experiment_2858_beaver_epr_clean_bounded_proxy_v2.json",
        _proxy_artifact(),
    )
    artifact = exp.run_experiment(
        exp.ExperimentConfig(repo_root=tmp_path, n_examples=2, started_at=5.0, clock=lambda: 5.5)
    )

    exp.validate_artifact(artifact)
    with pytest.raises(ValueError, match="missing required fields"):
        exp.validate_artifact({"honest_verdict": "complete: x"})
    with pytest.raises(ValueError, match="full exact BEAVER"):
        exp.validate_artifact(artifact | {"exact_beaver_implemented": True})
    with pytest.raises(ValueError, match="requires solver"):
        exp.validate_artifact(artifact | {"solver_used": None})
    with pytest.raises(ValueError, match="cannot have blocked_reason"):
        exp.validate_artifact(artifact | {"blocked_reason": "blocked_dependency"})
    with pytest.raises(ValueError, match="blocked artifacts require blocked_reason"):
        exp.validate_artifact(artifact | {"exact_frontier_available": False})
    with pytest.raises(ValueError, match="run_date"):
        exp.validate_artifact(artifact | {"run_date": "20260101"})

    unsupported = exp.score_example_exact_frontier(
        LabeledExample("unsupported", "Trace: 1 / 0 = 0.", 1, "fixture")
    )
    assert unsupported["exact_frontier_available"] is False
    assert unsupported["unsupported_claim_count"] == 1

    assert exp._z3_equality_holds("(6 - 2) / (1 + 1)", "2") is True
    assert exp._z3_equality_holds("0.5 + 0.5", "1") is True
    assert exp._fraction_from_ast(ast.parse("-(6 - 2) / +(1 + 1)", mode="eval").body) == -2
    assert exp._fraction_from_ast(ast.parse("2 * 3", mode="eval").body) == 6
    with pytest.raises(ValueError, match="division by zero"):
        exp._fraction_from_ast(ast.parse("1 / 0", mode="eval").body)
    with pytest.raises(ValueError, match="unsupported arithmetic expression"):
        exp._parse_arithmetic("name")
    with pytest.raises(ValueError, match="unsupported arithmetic expression"):
        exp._parse_arithmetic("010 - 5")
    with pytest.raises(ValueError, match="unsupported arithmetic AST"):
        exp._z3_from_ast(ast.parse("2 ** 3", mode="eval").body)
    with pytest.raises(ValueError, match="unsupported arithmetic AST"):
        exp._fraction_from_ast(ast.parse("2 ** 3", mode="eval").body)

    original_z3 = exp.z3
    try:
        exp.z3 = None
        with pytest.raises(ValueError, match="z3 unavailable"):
            exp._z3_equality_holds("1 + 1", "2")
        with pytest.raises(ValueError, match="z3 unavailable"):
            exp._z3_real(Fraction(1, 1))
    finally:
        exp.z3 = original_z3
