"""Tests for Exp 3655 facts-row remeasurement with real NLI.

Spec: REQ-VERIFY-3655, SCENARIO-VERIFY-3655.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pytest

from carnot.verify import facts_row_remeasurement_real_nli_v5 as mod
from carnot.verify.facts_row_remeasurement_real_nli_v5 import (
    REQUIRED_ARTIFACT_FIELDS,
    build_artifact,
    facts_second_pair_of_eyes,
    score_facts_rows,
    validate_artifact,
)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _facts_rows() -> list[dict[str, Any]]:
    rows = []
    for idx, (label, confidence) in enumerate(
        [
            (0, 0.90),
            (0, 0.80),
            (0, 0.60),
            (1, 0.70),
            (1, 0.30),
            (1, 0.20),
        ]
    ):
        rows.append(
            {
                "question": f"Question {idx}",
                "answer": f"Candidate answer {idx}",
                "evidence_passage": f"Evidence passage {idx}",
                "is_hallucination": label,
                "model_confidence": confidence,
            }
        )
    return rows


def _seed_fixture(
    root: Path,
    *,
    nli_built: bool = True,
    leak_free: bool = True,
    model_based: bool = True,
) -> None:
    _write_jsonl(root / "data/realistic_factual_corpus_v3.jsonl", _facts_rows())
    _write_json(
        root / "results/experiment_3640_build_factual_corpus_v3.json",
        {
            "corpus_path_used": "data/realistic_factual_corpus_v3.jsonl",
            "facts_corpus_validated": True,
            "confidence_baseline_auroc_on_corpus": 0.666667,
        },
    )
    _write_json(
        root / "results/experiment_3642_corrected_cross_domain_remeasurement_v4.json",
        {
            "generalization_table": {
                "facts": {
                    "ensemble_auroc": {"point": 0.6495},
                    "domain_verdict": "domain_bound",
                }
            }
        },
    )
    substrate = (
        "model_based_transformers_checkpoint: fake-nli on cpu"
        if model_based
        else "disclosed_text_statistical_proxy_no_cached_nli_checkpoint"
    )
    _write_json(
        root / "results/experiment_3654_real_nli_atomic_claim_grounding_verifier.json",
        {
            "nli_grounding_built": nli_built,
            "grounding_leak_free": leak_free,
            "nli_substrate": substrate,
        },
    )


@pytest.mark.parametrize(
    (
        "honest_outcome",
        "score_overrides",
        "nli_built",
        "leak_free",
        "model_based",
        "expected_generalizes",
        "expected_positive_control",
    ),
    [
        (
            "generalizes",
            {
                "grounding_scores": [0.10, 0.30, 0.60, 0.50, 0.80, 0.90],
                "confidence_scores": [0.20, 0.40, 0.60, 0.30, 0.50, 0.70],
            },
            True,
            True,
            True,
            True,
            True,
        ),
        (
            "domain_bound",
            {
                "grounding_scores": [0.20, 0.50, 0.70, 0.30, 0.40, 0.60],
                "confidence_scores": [0.10, 0.20, 0.40, 0.30, 0.70, 0.80],
            },
            True,
            True,
            True,
            False,
            True,
        ),
        (
            "blocked",
            {
                "grounding_scores": [0.10, 0.30, 0.60, 0.50, 0.80, 0.90],
                "confidence_scores": [0.20, 0.40, 0.60, 0.30, 0.50, 0.70],
            },
            False,
            True,
            True,
            False,
            False,
        ),
        (
            "blocked",
            {
                "grounding_scores": [0.10, 0.30, 0.60, 0.50, 0.80, 0.90],
                "confidence_scores": [0.20, 0.40, 0.60, 0.30, 0.50, 0.70],
            },
            True,
            False,
            True,
            False,
            False,
        ),
        (
            "blocked",
            {
                "grounding_scores": [0.10, 0.30, 0.60, 0.50, 0.80, 0.90],
                "confidence_scores": [0.20, 0.40, 0.60, 0.30, 0.50, 0.70],
            },
            True,
            True,
            False,
            False,
            False,
        ),
    ],
)
def test_exp3655_parametrizes_honest_outcomes(
    tmp_path: Path,
    honest_outcome: str,
    score_overrides: dict[str, list[float]],
    nli_built: bool,
    leak_free: bool,
    model_based: bool,
    expected_generalizes: bool,
    expected_positive_control: bool,
) -> None:
    """SCENARIO-VERIFY-3655: honest synthetic outcomes drive the verdict."""

    _seed_fixture(
        tmp_path,
        nli_built=nli_built,
        leak_free=leak_free,
        model_based=model_based,
    )
    artifact = build_artifact(
        tmp_path,
        score_overrides=score_overrides,
        started_s=10.0,
        now_s=12.0,
        n_bootstrap=12,
    )

    validate_artifact(artifact)
    assert set(REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert set(REQUIRED_ARTIFACT_FIELDS) <= set(artifact["field_principles"])
    assert artifact["honest_outcome"] == honest_outcome
    assert type(artifact["facts_generalize_real_nli"]) is bool
    assert type(artifact["positive_control_valid"]) is bool
    assert artifact["facts_generalize_real_nli"] is expected_generalizes
    assert artifact["positive_control_valid"] is expected_positive_control

    if honest_outcome == "blocked":
        assert artifact["honest_verdict"] == (
            "complete: blocked_nli_grounding_verifier_unavailable_or_leaky"
        )
        assert artifact["grounding_auroc_real_nli"] is None
        assert artifact["mcnemar_p_facts"] is None
        return

    assert artifact["grounding_auroc_real_nli"]["n"] == 6
    assert len(artifact["grounding_auroc_real_nli"]["bootstrap_seeds"]) >= 3
    assert artifact["confidence_baseline_auroc"]["ci95"] is not None
    assert artifact["grounding_minus_confidence_delta"]["ci95"] is not None
    assert artifact["facts_conditional_catch_rate"]["denominator_confidence_missed_errors"] >= 0
    assert artifact["acceptance_gate"]["passed"] is True


def test_exp3655_second_pair_of_eyes_mcnemar_counts() -> None:
    """REQ-VERIFY-3655: fixed-FPR catch rate and McNemar are paired."""

    labels = [0, 0, 0, 1, 1, 1]
    confidence_scores = [0.10, 0.20, 0.30, 0.90, 0.40, 0.20]
    grounding_scores = [0.10, 0.20, 0.30, 0.80, 0.70, 0.60]

    stats = facts_second_pair_of_eyes(
        labels,
        grounding_scores,
        confidence_scores,
        fixed_confidence_fpr=0.0,
        n_bootstrap=12,
    )

    assert stats["numerator_grounding_catches_confidence_misses"] == 1
    assert stats["denominator_confidence_missed_errors"] == 1
    assert stats["point"] == 1.0
    assert stats["mcnemar"]["grounding_only_error_catches"] == 1
    assert stats["mcnemar"]["confidence_only_error_catches"] == 0
    assert stats["mcnemar"]["p_value"] == 1.0


def test_exp3655_score_path_uses_only_answer_and_evidence() -> None:
    """REQ-VERIFY-3655: verifier score path cannot read labels or gold answers."""

    class GuardedRow(dict[str, Any]):
        def get(self, key: str, default: Any = None) -> Any:
            if key in {"is_hallucination", "gold_answer", "right_answer"}:
                raise AssertionError(f"forbidden score-path key: {key}")
            return super().get(key, default)

    class SpyVerifier:
        model_based = True
        nli_substrate = "model_based_transformers_checkpoint: fake-nli on cpu"

        def __init__(self) -> None:
            self.calls: list[tuple[str, str]] = []

        def verify(self, answer: str, evidence: str) -> float:
            self.calls.append((answer, evidence))
            return 0.42

    verifier = SpyVerifier()
    scores = score_facts_rows(
        [
            GuardedRow(
                {
                    "answer": "Paris is the capital of France.",
                    "evidence_passage": "Paris is the capital city.",
                    "is_hallucination": 0,
                    "gold_answer": "Paris",
                }
            )
        ],
        verifier=verifier,
    )

    assert scores == [0.42]
    assert verifier.calls == [("Paris is the capital of France.", "Paris is the capital city.")]


def test_exp3655_validation_rejects_wrapped_bare_booleans(tmp_path: Path) -> None:
    """REQ-VERIFY-3655: core gate booleans remain bare top-level bools."""

    _seed_fixture(tmp_path)
    artifact = build_artifact(
        tmp_path,
        score_overrides={
            "grounding_scores": [0.10, 0.30, 0.60, 0.50, 0.80, 0.90],
            "confidence_scores": [0.20, 0.40, 0.60, 0.30, 0.50, 0.70],
        },
        started_s=1.0,
        now_s=2.0,
        n_bootstrap=8,
    )

    bad = dict(artifact, facts_generalize_real_nli={"value": True})
    with pytest.raises(ValueError, match="facts_generalize_real_nli"):
        validate_artifact(bad)

    missing = dict(artifact)
    missing.pop("n_examples")
    with pytest.raises(ValueError, match="missing required artifact fields"):
        validate_artifact(missing)

    bad_principles = dict(artifact, field_principles={})
    with pytest.raises(ValueError, match="field_principles"):
        validate_artifact(bad_principles)

    bad_duration = dict(artifact, duration_s=-1.0)
    with pytest.raises(ValueError, match="duration_s"):
        validate_artifact(bad_duration)


def test_exp3655_write_artifact(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-3655: artifact writing persists the validated schema."""

    _seed_fixture(tmp_path)
    output = mod.write_artifact(
        tmp_path,
        output_path="results/exp3655.json",
        score_overrides={
            "grounding_scores": [0.10, 0.30, 0.60, 0.50, 0.80, 0.90],
            "confidence_scores": [0.20, 0.40, 0.60, 0.30, 0.50, 0.70],
        },
        tests_run=["pytest tests/python/test_experiment_3655_facts_row_remeasurement_real_nli_v5.py"],
    )

    written = json.loads(output.read_text(encoding="utf-8"))
    assert written["tests_run"] == [
        "pytest tests/python/test_experiment_3655_facts_row_remeasurement_real_nli_v5.py"
    ]
    assert written["facts_generalize_real_nli"] is True


def test_exp3655_edge_branches_and_validation_errors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-VERIFY-3655: blocked and malformed edge paths stay explicit."""

    _seed_fixture(tmp_path)

    class SequencedVerifier:
        model_based = True
        nli_substrate = "model_based_transformers_checkpoint: fake-nli on cpu"

        def __init__(self) -> None:
            self.scores = iter([0.10, 0.30, 0.60, 0.50, 0.80, 0.90])

        def verify(self, answer: str, evidence: str) -> float:
            assert answer.startswith("Candidate answer")
            assert evidence.startswith("Evidence passage")
            return next(self.scores)

    artifact = build_artifact(
        tmp_path,
        verifier=SequencedVerifier(),
        started_s=1.0,
        now_s=2.0,
        n_bootstrap=8,
    )
    assert artifact["grounding_auroc_real_nli"]["point"] > 0.5

    no_finite = build_artifact(
        tmp_path,
        score_overrides={
            "grounding_scores": [float("nan")] * 6,
            "confidence_scores": [float("nan")] * 6,
        },
        started_s=1.0,
        now_s=2.0,
    )
    assert no_finite["blocked_reason"] == "blocked_no_finite_score_triplets"

    def fail_checkpoint_load(*, allow_proxy: bool) -> Any:
        assert allow_proxy is False
        raise RuntimeError("no cached checkpoint")

    monkeypatch.setattr(
        mod.NLIAtomicClaimGroundingVerifier,
        "from_cached_or_proxy",
        staticmethod(fail_checkpoint_load),
    )
    unavailable = build_artifact(tmp_path, started_s=1.0, now_s=2.0)
    assert unavailable["blocked_reason"] == "blocked_real_nli_checkpoint_unavailable: RuntimeError"

    assert mod.decisions_at_fpr([0, 1], [0.2, 0.8], -0.1)["decisions"] == [False, False]
    assert mod.bootstrap_conditional_catch_ci(
        [],
        [],
        [],
        fixed_confidence_fpr=0.1,
        n_bootstrap=1,
        seeds=(1,),
    ) == (None, [])
    assert (
        mod.terminal_verdict(facts_generalize_real_nli=False, positive_control_valid=False)
        == "complete: blocked_nli_grounding_verifier_unavailable_or_leaky"
    )
    assert mod.proxy_facts_auroc({}) == pytest.approx(0.6495)
    assert mod.proxy_facts_verdict({"generalization_table": {"facts": "bad"}}) == "domain_bound"

    bad_verdict = dict(artifact, honest_verdict="failed")
    with pytest.raises(ValueError, match="honest_verdict"):
        validate_artifact(bad_verdict)

    bad_principles = dict(artifact, field_principles=None)
    with pytest.raises(ValueError, match="field_principles"):
        validate_artifact(bad_principles)

    missing_metric = dict(artifact, grounding_auroc_real_nli=None)
    with pytest.raises(ValueError, match="grounding_auroc_real_nli"):
        validate_artifact(missing_metric)

    short_bootstrap = dict(artifact)
    short_bootstrap["confidence_baseline_auroc"] = dict(
        artifact["confidence_baseline_auroc"],
        bootstrap_seeds=[1, 2],
    )
    with pytest.raises(ValueError, match="bootstrap"):
        validate_artifact(short_bootstrap)

    missing_catch_rate = dict(artifact, facts_conditional_catch_rate=None)
    with pytest.raises(ValueError, match="facts_conditional_catch_rate"):
        validate_artifact(missing_catch_rate)

    bad_mcnemar = dict(artifact, mcnemar_p_facts="bad")
    with pytest.raises(ValueError, match="mcnemar_p_facts"):
        validate_artifact(bad_mcnemar)

    bad_examples = dict(artifact, n_examples=-1)
    with pytest.raises(ValueError, match="n_examples"):
        validate_artifact(bad_examples)

    assert mod._load_valid_v3_rows(tmp_path / "missing.jsonl") == (
        [],
        "blocked_missing_v3_facts_corpus",
    )
    empty = tmp_path / "empty.jsonl"
    empty.write_text("", encoding="utf-8")
    assert mod._load_valid_v3_rows(empty) == ([], "blocked_empty_v3_facts_corpus")
    malformed = tmp_path / "malformed.jsonl"
    malformed.write_text(json.dumps({"answer": "x"}) + "\n", encoding="utf-8")
    rows, reason = mod._load_valid_v3_rows(malformed)
    assert rows == []
    assert reason.startswith("blocked_v3_facts_corpus_schema_row_0")
    assert mod._resolve_corpus_path(tmp_path, {}) == tmp_path / mod.DEFAULT_CORPUS_REL_PATH

    outside = tmp_path.parent / f"{tmp_path.name}_outside.jsonl"
    assert mod._display_path(tmp_path, outside) == str(outside)
    assert mod._read_json_object(tmp_path / "does-not-exist.json") == {}
    invalid_json = tmp_path / "invalid.json"
    invalid_json.write_text("{", encoding="utf-8")
    assert mod._read_json_object(invalid_json) == {}
    odd_jsonl = tmp_path / "odd.jsonl"
    odd_jsonl.write_text("\n{\n[]\n", encoding="utf-8")
    assert mod._read_jsonl(odd_jsonl) == []
    assert mod._read_jsonl(tmp_path / "missing-lines.jsonl") == []
    assert mod._repo_path(tmp_path, outside) == outside
    assert math.isnan(mod._coerce_float("not-a-number", float("nan")))
    assert mod._coerce_float(float("nan"), 0.5) == 0.5
