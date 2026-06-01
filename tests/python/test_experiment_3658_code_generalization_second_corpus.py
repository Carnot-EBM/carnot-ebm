"""Tests for Exp 3658 balanced second-corpus replication.

Spec: REQ-CODE-3658, SCENARIO-CODE-3658.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import pytest

from carnot.verify import code_generalization_second_corpus as exp


@dataclass(frozen=True)
class _HarnessResult:
    passed: bool
    error_type: str = "failure"
    error_message: str = "fixture"
    stdout: str = ""


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8"
    )


def _fixture_rows(n_rows: int = 60) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for idx in range(n_rows):
        label = idx % 2 == 0
        body = "return x + 1" if label else "return None"
        code = f"def f_{idx}(x):\n    {body}\n"
        rows.append(
            exp.corpus_row(
                candidate_code=code,
                label=label,
                test_outcome="candidate_passed" if label else "candidate_failed_tests",
                source="SCENARIO-CODE-3658 fixture",
                task_id=f"HumanEval/{idx // 2}",
                metadata={
                    "candidate_index": 0,
                    "corpus": "HumanEval",
                    "entry_point": f"f_{idx}",
                    "stable_id": f"HumanEval/{idx // 2}",
                },
            )
        )
    return rows


def _ranked_scores(rows: list[dict[str, Any]], *, transfers: bool) -> list[float]:
    scores = []
    for idx, row in enumerate(rows):
        is_error = not bool(row["label"])
        if transfers:
            scores.append(0.9 if is_error else 0.1)
        else:
            scores.append((0.1 if is_error else 0.9) + 0.0001 * idx)
    return scores


def _weak_confidence_scores(rows: list[dict[str, Any]]) -> list[float]:
    return [0.45 if not bool(row["label"]) else 0.55 for row in rows]


def _prior_transfer_artifact() -> dict[str, Any]:
    return {
        "hypothesis_supported": "transfer",
        "transfer_delta_vs_literature": {"meets_lower_anchor": True},
        "math_signal_code_auroc": {"point": 0.468468},
        "code_confidence_baseline_auroc": {"point": 0.361909},
    }


@pytest.mark.parametrize(
    (
        "case_name",
        "corpus_rows",
        "execution_scores",
        "math_scores",
        "expected_verdict",
        "expected_replicates",
        "expected_fire",
    ),
    [
        (
            "blocked",
            [],
            [],
            [],
            "complete: blocked_no_second_code_corpus",
            False,
            False,
        ),
        (
            "verifiers_inert",
            _fixture_rows(),
            [],
            "transfer",
            "complete: code_verifiers_inert_on_second_corpus_diagnosed",
            False,
            False,
        ),
        (
            "does_not_replicate",
            _fixture_rows(),
            "transfer",
            "no_transfer",
            "complete: code_generalization_does_not_replicate_single_corpus_was_artifact",
            False,
            True,
        ),
        (
            "replicates",
            _fixture_rows(),
            "transfer",
            "transfer",
            "complete: code_generalization_replicates_on_balanced_second_corpus_claim_hardened",
            True,
            True,
        ),
    ],
)
def test_scenario_code_3658_parametrized_honest_verdicts(
    tmp_path: Path,
    case_name: str,
    corpus_rows: list[dict[str, Any]],
    execution_scores: list[float] | str,
    math_scores: list[float] | str,
    expected_verdict: str,
    expected_replicates: bool,
    expected_fire: bool,
) -> None:
    """SCENARIO-CODE-3658: verdicts cover blocked, inert, null, and replication."""

    overrides: dict[str, Any] = {
        "confidence_scores": _weak_confidence_scores(corpus_rows),
    }
    if execution_scores == "transfer":
        overrides["execution_scores"] = _ranked_scores(corpus_rows, transfers=True)
    else:
        overrides["execution_scores"] = execution_scores
    if math_scores == "transfer":
        overrides["math_scores"] = _ranked_scores(corpus_rows, transfers=True)
    elif math_scores == "no_transfer":
        overrides["math_scores"] = _ranked_scores(corpus_rows, transfers=False)
    else:
        overrides["math_scores"] = math_scores

    artifact = exp.build_artifact(
        tmp_path,
        started_s=10.0,
        now_s=13.25,
        corpus_rows=corpus_rows,
        exp3641_artifact=_prior_transfer_artifact(),
        score_overrides=overrides,
        tests_run=[f"SCENARIO-CODE-3658 {case_name}"],
    )

    assert artifact["honest_verdict"] == expected_verdict
    assert artifact["code_generalization_replicates"] is expected_replicates
    assert type(artifact["code_generalization_replicates"]) is bool
    assert artifact["code_verifiers_fire"] is expected_fire
    assert set(exp.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert artifact["duration_s"] == pytest.approx(3.25)
    assert artifact["inference_substrate"].startswith("verifier_ensemble_against_cached_candidates")

    if corpus_rows:
        assert artifact["n_examples"] == len(corpus_rows)
        assert artifact["class_balance"]["min_class_fraction"] == pytest.approx(0.5)
        assert artifact["acceptance_gate"]["passed"] is expected_fire
        saved_rows = [
            json.loads(line)
            for line in (tmp_path / exp.CORPUS_REL_PATH).read_text(encoding="utf-8").splitlines()
        ]
        assert {"candidate_code", "label", "test_outcome"} <= set(saved_rows[0])
        assert len(artifact["math_signal_code_auroc"]["bootstrap_seeds"]) >= 3
    else:
        assert artifact["n_examples"] == 0
        assert artifact["acceptance_gate"]["passed"] is False


def test_req_code_3658_builds_balanced_humaneval_split_with_execution_labels(
    tmp_path: Path,
) -> None:
    """REQ-CODE-3658: a fresh HumanEval split is labeled by harness outcomes."""

    manifest_path = tmp_path / exp.HUMANEVAL_MANIFEST_REL_PATH
    manifest_rows = [
        {
            "canonical_solution": "    return x + 1\n",
            "dataset": "HumanEval",
            "entry_point": f"f_{idx}",
            "prompt": f"def f_{idx}(x):\n",
            "stable_id": f"HumanEval/{idx}",
            "tests": f"def check(candidate):\n    assert candidate({idx}) == {idx + 1}\n",
        }
        for idx in range(25)
    ]
    _write_jsonl(manifest_path, manifest_rows)

    def fake_executor(code: str, problem: dict[str, Any], timeout: float) -> _HarnessResult:
        del problem, timeout
        if "return None" in code:
            return _HarnessResult(passed=False, error_message="AssertionError")
        return _HarnessResult(passed=True, error_type="none", error_message="")

    rows, status = exp.build_second_labeled_code_corpus(
        tmp_path,
        target_per_class=25,
        executor=fake_executor,
    )

    counts = Counter(row["label"] for row in rows)
    assert counts == {True: 25, False: 25}
    assert status["selected_corpus_name"] == "HumanEval-split"
    assert status["n_manifest_rows_scanned"] == 25
    assert all({"candidate_code", "label", "test_outcome"} <= set(row) for row in rows)
    assert any("return None" in row["candidate_code"] for row in rows if not row["label"])
    assert exp.class_balance(rows)["balanced"] is True


def test_req_code_3658_write_artifact_and_validation_guards(tmp_path: Path) -> None:
    """REQ-CODE-3658: artifact writing and bare-bool validation stay explicit."""

    rows = _fixture_rows()
    output = exp.write_artifact(
        tmp_path,
        output_path=tmp_path / "result.json",
        corpus_rows=rows,
        exp3641_artifact=_prior_transfer_artifact(),
        score_overrides={
            "execution_scores": _ranked_scores(rows, transfers=True),
            "math_scores": _ranked_scores(rows, transfers=True),
            "confidence_scores": _weak_confidence_scores(rows),
        },
        tests_run=["REQ-CODE-3658 write_artifact"],
    )
    artifact = json.loads(output.read_text(encoding="utf-8"))

    assert output == tmp_path / "result.json"
    assert artifact["code_generalization_replicates"] is True
    assert artifact["second_code_corpus_name"] == "HumanEval-split"

    broken = dict(artifact)
    broken["code_generalization_replicates"] = {"value": True}
    with pytest.raises(ValueError, match="code_generalization_replicates"):
        exp.validate_artifact(broken)

    missing = dict(artifact)
    missing.pop("class_balance")
    with pytest.raises(ValueError, match="missing required"):
        exp.validate_artifact(missing)


def test_req_code_3658_helper_edges(tmp_path: Path) -> None:
    """REQ-CODE-3658: helpers keep blocked and malformed inputs honest."""

    unbalanced = _fixture_rows(10)[:8] + [row for row in _fixture_rows(10) if not row["label"]][:1]
    missing_manifest_artifact = exp.build_artifact(
        tmp_path,
        started_s=1.0,
        now_s=1.5,
        target_per_class=25,
        exp3641_artifact=_prior_transfer_artifact(),
    )
    blocked = exp.blocked_artifact(
        root=tmp_path,
        started_s=1.0,
        now_s=2.0,
        reason="blocked_no_second_code_corpus",
        corpus_rows=unbalanced,
    )

    assert missing_manifest_artifact["honest_verdict"] == "complete: blocked_no_second_code_corpus"
    assert blocked["honest_verdict"] == "complete: blocked_no_second_code_corpus"
    assert blocked["class_balance"]["balanced"] is False
    assert exp.class_balance([])["min_class_fraction"] == 0.0
    assert exp.acceptance_gate(49, True, exp.class_balance(_fixture_rows(60)))["passed"] is False
    assert (
        exp.terminal_verdict(
            n_examples=49,
            balance=exp.class_balance(_fixture_rows(60)),
            code_verifiers_fire=True,
            replicates=True,
        )
        == "complete: blocked_no_second_code_corpus"
    )
    assert (
        exp.code_generalization_replicates(
            code_verifiers_fire=True,
            balance=exp.class_balance(_fixture_rows(60)),
            math_metrics={"point": None},
            confidence_metrics={"point": 0.4},
            exp3641_artifact=_prior_transfer_artifact(),
        )
        is False
    )
    assert exp._repo_path(tmp_path, Path("/tmp/absolute")) == Path("/tmp/absolute")
    assert exp._read_exp3641_artifact(tmp_path) == {}

    with pytest.raises(ValueError, match="entry point"):
        exp.make_return_none_mutant("def other(x):\n    return x\n", "missing")

    passing = exp._default_humaneval_executor(
        "def f():\n    return 1\n",
        {"test": "def check(candidate):\n    assert candidate() == 1\n", "entry_point": "f"},
        1.0,
    )
    assert passing.passed is True

    valid = exp.build_artifact(
        tmp_path,
        corpus_rows=_fixture_rows(),
        exp3641_artifact=_prior_transfer_artifact(),
        score_overrides={
            "execution_scores": _ranked_scores(_fixture_rows(), transfers=True),
            "math_scores": _ranked_scores(_fixture_rows(), transfers=True),
            "confidence_scores": _weak_confidence_scores(_fixture_rows()),
        },
    )
    for field, value, message in [
        ("code_verifiers_fire", {"value": True}, "code_verifiers_fire"),
        ("honest_verdict", "blocked", "honest_verdict"),
        ("n_examples", "60", "n_examples"),
        ("duration_s", -1.0, "duration_s"),
    ]:
        broken = dict(valid)
        broken[field] = value
        with pytest.raises(ValueError, match=message):
            exp.validate_artifact(broken)
