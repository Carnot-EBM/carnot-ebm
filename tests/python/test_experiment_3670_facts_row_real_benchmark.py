"""Tests for Exp 3670 facts-row real-benchmark remeasurement.

Spec: REQ-VERIFY-3670, SCENARIO-VERIFY-3670.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from carnot.verify import facts_row_real_benchmark_3670 as mod
from carnot.verify.facts_row_real_benchmark_3670 import (
    REQUIRED_ARTIFACT_FIELDS,
    build_artifact,
    score_real_rows,
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


def _real_rows(n_negative: int = 100, n_positive: int = 100) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for idx in range(n_negative + n_positive):
        label = 1 if idx >= n_negative else 0
        rows.append(
            {
                "question": f"Real benchmark question {idx}",
                "answer": f"Real benchmark model answer {idx} with cited factual details.",
                "evidence_passage": (
                    f"Retrieved real benchmark evidence passage {idx} with article context."
                ),
                "is_hallucination": label,
                "model_confidence": 0.64 if label == 0 else 0.55,
            }
        )
    return rows


def _seed_fixture(
    root: Path,
    *,
    real_built: bool = True,
    nli_built: bool = True,
    leak_free: bool = True,
    model_based: bool = True,
) -> None:
    rows = _real_rows()
    _write_jsonl(root / "data/real_factual_corpus_ragtruth.jsonl", rows)
    _write_json(
        root / "results/experiment_3669_build_real_factual_corpus.json",
        {
            "honest_verdict": "complete: real_factual_corpus_built_ragtruth_non_degenerate",
            "real_factual_corpus_built": real_built,
            "corpus_non_degenerate": real_built,
            "corpus_path": "data/real_factual_corpus_ragtruth.jsonl",
            "confidence_baseline_auroc": 0.70,
            "n_examples": len(rows) if real_built else 0,
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
    _write_json(
        root / "results/experiment_3655_facts_row_remeasurement_real_nli_v5.json",
        {
            "grounding_auroc_real_nli": {"point": 0.743656},
            "facts_conditional_catch_rate": {"mcnemar": {"p_value": 0.00031}},
        },
    )


def _scores_generalizes() -> dict[str, list[float]]:
    grounding_neg = [0.10 + (idx % 70) / 1000 for idx in range(100)]
    grounding_pos = [0.72 + (idx % 70) / 1000 for idx in range(88)] + [
        0.05 + idx / 1000 for idx in range(12)
    ]
    confidence_neg = [0.24 + (idx % 50) / 1000 for idx in range(100)]
    confidence_pos = [0.28 + (idx % 50) / 1000 for idx in range(55)] + [
        0.18 + (idx % 30) / 1000 for idx in range(45)
    ]
    return {
        "grounding_scores": grounding_neg + grounding_pos,
        "confidence_scores": confidence_neg + confidence_pos,
    }


def _scores_catch_value_at_parity() -> dict[str, list[float]]:
    confidence_neg = [0.90 + idx / 1000 for idx in range(10)] + [
        0.30 + idx / 300 for idx in range(90)
    ]
    confidence_pos = [0.96 + idx / 1000 for idx in range(8)] + [
        0.45 + (idx % 20) / 1000 for idx in range(92)
    ]
    grounding_neg = [0.90 + idx / 1000 for idx in range(10)] + [
        0.42 + (idx % 50) / 1000 for idx in range(90)
    ]
    grounding_pos = [0.96 + idx / 1000 for idx in range(38)] + [
        0.10 + (idx % 50) / 1000 for idx in range(62)
    ]
    return {
        "grounding_scores": grounding_neg + grounding_pos,
        "confidence_scores": confidence_neg + confidence_pos,
    }


def _scores_domain_bound() -> dict[str, list[float]]:
    confidence_neg = [0.12 + (idx % 80) / 1000 for idx in range(100)]
    confidence_pos = [0.62 + (idx % 80) / 1000 for idx in range(86)] + [
        0.08 + idx / 1000 for idx in range(14)
    ]
    grounding_neg = [0.20 + (idx % 80) / 1000 for idx in range(100)]
    grounding_pos = [0.24 + (idx % 80) / 1000 for idx in range(48)] + [
        0.12 + (idx % 50) / 1000 for idx in range(52)
    ]
    return {
        "grounding_scores": grounding_neg + grounding_pos,
        "confidence_scores": confidence_neg + confidence_pos,
    }


@pytest.mark.parametrize(
    (
        "honest_outcome",
        "score_overrides",
        "real_built",
        "nli_built",
        "leak_free",
        "model_based",
        "expected_verdict",
        "expected_core_bool",
    ),
    [
        pytest.param(
            "generalizes_real",
            _scores_generalizes(),
            True,
            True,
            True,
            True,
            mod.GENERALIZES_VERDICT,
            True,
            id="generalizes_real",
        ),
        pytest.param(
            "catch_value_at_parity",
            _scores_catch_value_at_parity(),
            True,
            True,
            True,
            True,
            mod.CATCH_VALUE_VERDICT,
            True,
            id="catch_value_at_parity",
        ),
        pytest.param(
            "domain_bound_real",
            _scores_domain_bound(),
            True,
            True,
            True,
            True,
            mod.DOMAIN_BOUND_VERDICT,
            False,
            id="domain_bound_real",
        ),
        pytest.param(
            "blocked",
            _scores_generalizes(),
            False,
            True,
            True,
            True,
            mod.BLOCKED_VERDICT,
            False,
            id="blocked_real_corpus",
        ),
        pytest.param(
            "blocked",
            _scores_generalizes(),
            True,
            False,
            True,
            True,
            mod.BLOCKED_VERDICT,
            False,
            id="blocked_verifier_unavailable",
        ),
        pytest.param(
            "blocked",
            _scores_generalizes(),
            True,
            True,
            True,
            False,
            mod.BLOCKED_VERDICT,
            False,
            id="blocked_proxy_verifier",
        ),
    ],
)
def test_exp3670_parametrizes_honest_outcomes(
    tmp_path: Path,
    honest_outcome: str,
    score_overrides: dict[str, list[float]],
    real_built: bool,
    nli_built: bool,
    leak_free: bool,
    model_based: bool,
    expected_verdict: str,
    expected_core_bool: bool,
) -> None:
    """SCENARIO-VERIFY-3670: realistic fixtures choose honest terminal outcomes."""

    _seed_fixture(
        tmp_path,
        real_built=real_built,
        nli_built=nli_built,
        leak_free=leak_free,
        model_based=model_based,
    )
    artifact = build_artifact(
        tmp_path,
        score_overrides=score_overrides,
        started_s=10.0,
        now_s=14.0,
        n_bootstrap=8,
    )

    validate_artifact(artifact)
    assert set(REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert set(REQUIRED_ARTIFACT_FIELDS) <= set(artifact["field_principles"])
    assert artifact["honest_outcome"] == honest_outcome
    assert artifact["honest_verdict"] == expected_verdict
    assert type(artifact["facts_generalize_or_adds_value_real"]) is bool
    assert type(artifact["grounding_leak_free"]) is bool
    assert type(artifact["positive_control_valid"]) is bool
    assert artifact["facts_generalize_or_adds_value_real"] is expected_core_bool

    if honest_outcome == "blocked":
        assert artifact["grounding_auroc_real_corpus"] is None
        assert artifact["facts_conditional_catch_rate"] is None
        assert artifact["acceptance_gate"]["passed"] is False
        return

    assert artifact["grounding_auroc_real_corpus"]["n"] == 200
    assert len(artifact["grounding_auroc_real_corpus"]["bootstrap_seeds"]) >= 3
    assert artifact["confidence_baseline_auroc"]["ci95"] is not None
    assert artifact["grounding_minus_confidence_delta"]["ci95"] is not None
    assert artifact["facts_conditional_catch_rate"]["mcnemar"]["p_value"] == (
        artifact["mcnemar_p_facts"]
    )
    assert artifact["real_vs_synthetic_grounding_delta"]["synthetic_grounding_auroc"] == 0.743656
    assert artifact["acceptance_gate"]["passed"] is True


def test_exp3670_leak_guard_blocks_near_perfect_real_auroc(tmp_path: Path) -> None:
    """REQ-VERIFY-3670: AUROC >=0.99 on n>=200 is treated as a leak red flag."""

    _seed_fixture(tmp_path)
    scores = {
        "grounding_scores": [0.1] * 100 + [0.9] * 100,
        "confidence_scores": [0.2] * 100 + [0.6] * 60 + [0.1] * 40,
    }
    artifact = build_artifact(
        tmp_path,
        score_overrides=scores,
        started_s=1.0,
        now_s=2.0,
        n_bootstrap=8,
    )

    assert artifact["honest_verdict"] == mod.BLOCKED_VERDICT
    assert artifact["grounding_leak_free"] is False
    assert "grounding_auroc_at_or_above_0.99_on_n_ge_200" in artifact["leak_diagnostics"]
    assert artifact["positive_control_valid"] is False
    assert artifact["acceptance_gate"]["passed"] is False


def test_exp3670_score_path_uses_only_answer_and_evidence() -> None:
    """REQ-VERIFY-3670: verifier scoring cannot read labels or gold answers."""

    class GuardedRow(dict[str, Any]):
        def get(self, key: str, default: Any = None) -> Any:
            if key in {"is_hallucination", "gold_answer", "correct_answer"}:
                raise AssertionError(f"forbidden score-path key: {key}")
            return super().get(key, default)

    class SpyVerifier:
        model_based = True
        nli_substrate = "model_based_transformers_checkpoint: fake-nli on cpu"

        def __init__(self) -> None:
            self.calls: list[tuple[str, str]] = []

        def verify(self, answer: str, evidence: str) -> float:
            self.calls.append((answer, evidence))
            return 0.37

    verifier = SpyVerifier()
    scores = score_real_rows(
        [
            GuardedRow(
                {
                    "answer": "Anne Frank died before March 1945.",
                    "evidence_passage": "The article says the sisters likely died earlier.",
                    "is_hallucination": 0,
                    "gold_answer": "before March",
                }
            )
        ],
        verifier=verifier,
    )

    assert scores == [0.37]
    assert verifier.calls == [
        ("Anne Frank died before March 1945.", "The article says the sisters likely died earlier.")
    ]


def test_exp3670_validation_rejects_malformed_core_fields(tmp_path: Path) -> None:
    """REQ-VERIFY-3670: required fields and bare booleans are validated."""

    _seed_fixture(tmp_path)
    artifact = build_artifact(
        tmp_path,
        score_overrides=_scores_domain_bound(),
        started_s=1.0,
        now_s=2.0,
        n_bootstrap=8,
    )

    bad_bool = dict(artifact, facts_generalize_or_adds_value_real={"value": True})
    with pytest.raises(ValueError, match="facts_generalize_or_adds_value_real"):
        validate_artifact(bad_bool)

    missing = dict(artifact)
    missing.pop("n_examples")
    with pytest.raises(ValueError, match="missing required artifact fields"):
        validate_artifact(missing)

    bad_principles = dict(artifact, field_principles={})
    with pytest.raises(ValueError, match="field_principles"):
        validate_artifact(bad_principles)

    bad_verdict = dict(artifact, honest_verdict="failed")
    with pytest.raises(ValueError, match="honest_verdict"):
        validate_artifact(bad_verdict)

    bad_duration = dict(artifact, duration_s=-1.0)
    with pytest.raises(ValueError, match="duration_s"):
        validate_artifact(bad_duration)


def test_exp3670_write_artifact_persists_schema(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-3670: the script-facing writer persists validated JSON."""

    _seed_fixture(tmp_path)
    output = mod.write_artifact(
        tmp_path,
        output_path="results/exp3670.json",
        score_overrides=_scores_catch_value_at_parity(),
        tests_run=["pytest tests/python/test_experiment_3670_facts_row_real_benchmark.py"],
    )
    written = json.loads(output.read_text(encoding="utf-8"))

    assert written["honest_outcome"] == "catch_value_at_parity"
    assert written["facts_generalize_or_adds_value_real"] is True
    assert written["tests_run"] == [
        "pytest tests/python/test_experiment_3670_facts_row_real_benchmark.py"
    ]


def test_exp3670_file_helpers_and_edge_branches(tmp_path: Path) -> None:
    """REQ-VERIFY-3670: blocked helper branches stay explicit."""

    assert mod._read_json_object(tmp_path / "missing.json") == {}
    invalid = tmp_path / "invalid.json"
    invalid.write_text("{", encoding="utf-8")
    assert mod._read_json_object(invalid) == {}
    assert mod._read_jsonl(tmp_path / "missing.jsonl") == []
    odd_jsonl = tmp_path / "odd.jsonl"
    odd_jsonl.write_text("\n{\n[]\n", encoding="utf-8")
    assert mod._read_jsonl(odd_jsonl) == []
    assert mod._load_valid_real_rows(tmp_path / "missing.jsonl") == (
        [],
        "blocked_missing_real_factual_corpus",
    )
    empty = tmp_path / "empty.jsonl"
    empty.write_text("", encoding="utf-8")
    assert mod._load_valid_real_rows(empty) == ([], "blocked_empty_real_factual_corpus")
    malformed = tmp_path / "malformed.jsonl"
    malformed.write_text(json.dumps({"answer": "x"}) + "\n", encoding="utf-8")
    rows, reason = mod._load_valid_real_rows(malformed)
    assert rows == []
    assert reason.startswith("blocked_real_factual_corpus_schema_row_0")
    assert mod._resolve_real_corpus_path(tmp_path, {}) == (
        tmp_path / mod.DEFAULT_REAL_CORPUS_REL_PATH
    )
    outside = tmp_path.parent / f"{tmp_path.name}_outside.jsonl"
    assert mod._repo_path(tmp_path, outside) == outside
    assert mod._display_path(tmp_path, outside) == str(outside)
    assert mod._coerce_float("bad", 0.25) == 0.25
    assert mod._coerce_float(float("nan"), 0.25) == 0.25
    assert mod._round_or_none(None) is None
    assert mod._round_or_none("bad") is None
    assert mod._round_or_none(float("nan")) is None
    assert mod.synthetic_grounding_auroc({}) == pytest.approx(mod.SYNTHETIC_GROUNDING_AUROC)
    assert mod.real_corpus_precondition({})[0] is False
    assert mod.real_corpus_precondition(
        {"real_factual_corpus_built": True, "corpus_non_degenerate": False, "n_examples": 200}
    ) == (False, "blocked_exp3669_corpus_non_degenerate_not_true")
    assert mod.real_corpus_precondition(
        {"real_factual_corpus_built": True, "corpus_non_degenerate": True, "n_examples": 199}
    ) == (False, "blocked_exp3669_n_examples_lt_200")
    assert mod.real_nli_precondition({"nli_grounding_built": True})[0] is False
    assert (
        mod.terminal_verdict(
            honest_outcome="blocked",
            facts_generalize_or_adds_value_real=False,
        )
        == mod.BLOCKED_VERDICT
    )
    assert mod.auroc_metric_bundle([], [], n_bootstrap=1, seeds=(1,))["point"] is None
    assert mod.paired_delta_bundle([], [], [], n_bootstrap=1, seeds=(1,))["point"] is None
    assert mod.fast_tie_aware_auroc([1, 1], [0.2, 0.4]) == 0.5
    assert mod.decisions_at_fpr([], [], 0.1)["decisions"] == []
    assert mod.decisions_at_fpr([0, 1], [0.2, 0.8], -0.1)["decisions"] == [False, False]
    assert mod.exact_mcnemar_p(0, 0) is None
    assert mod.exact_mcnemar_p(5000, 5) == 0.0
    assert mod.bootstrap_conditional_catch_ci(
        [],
        [],
        [],
        fixed_confidence_fpr=0.1,
        n_bootstrap=1,
        seeds=(1,),
    ) == (None, [])
    assert mod.materially_beats_confidence({"point": None, "ci95": None}) is False
    assert mod.significant_positive_catch_value({"point": None, "mcnemar": {}}) is False
    assert mod.grounding_leak_diagnostics(
        evidence_excludes_gold=False,
        grounding_auroc=0.5,
        n_examples=200,
        score_path_answer_evidence_only=False,
    ) == [
        "separate_gold_answer_found_in_evidence",
        "score_path_read_label_or_gold_field",
    ]


def test_exp3670_verifier_branch_and_blocked_score_edges(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-VERIFY-3670: real verifier and no-finite-score branches are explicit."""

    _seed_fixture(tmp_path)

    class SequencedVerifier:
        model_based = True
        nli_substrate = "model_based_transformers_checkpoint: fake-nli on cpu"

        def __init__(self) -> None:
            self.scores = iter(_scores_domain_bound()["grounding_scores"])

        def verify(self, answer: str, evidence: str) -> float:
            assert answer.startswith("Real benchmark model answer")
            assert evidence.startswith("Retrieved real benchmark evidence")
            return next(self.scores)

    artifact = build_artifact(
        tmp_path,
        verifier=SequencedVerifier(),
        started_s=1.0,
        now_s=2.0,
        n_bootstrap=8,
    )
    assert artifact["nli_substrate"] == "model_based_transformers_checkpoint: fake-nli on cpu"
    assert artifact["grounding_auroc_real_corpus"]["n"] == 200

    no_finite = build_artifact(
        tmp_path,
        score_overrides={
            "grounding_scores": [float("nan")] * 200,
            "confidence_scores": [float("nan")] * 200,
        },
        started_s=1.0,
        now_s=2.0,
    )
    assert no_finite["blocked_reason"] == "blocked_no_finite_real_score_triplets"

    def fail_checkpoint_load(*, allow_proxy: bool) -> Any:
        assert allow_proxy is False
        raise RuntimeError("missing cached checkpoint")

    monkeypatch.setattr(
        mod.NLIAtomicClaimGroundingVerifier,
        "from_cached_or_proxy",
        staticmethod(fail_checkpoint_load),
    )
    unavailable = build_artifact(tmp_path, started_s=1.0, now_s=2.0)
    assert unavailable["blocked_reason"] == "blocked_real_nli_checkpoint_unavailable: RuntimeError"


def test_exp3670_validation_metric_edge_errors(tmp_path: Path) -> None:
    """REQ-VERIFY-3670: validation rejects malformed metric internals."""

    _seed_fixture(tmp_path)
    artifact = build_artifact(
        tmp_path,
        score_overrides=_scores_domain_bound(),
        started_s=1.0,
        now_s=2.0,
        n_bootstrap=8,
    )

    bad_principles_type = dict(artifact, field_principles=None)
    with pytest.raises(ValueError, match="field_principles"):
        validate_artifact(bad_principles_type)

    missing_metric = dict(artifact, grounding_auroc_real_corpus=None)
    with pytest.raises(ValueError, match="grounding_auroc_real_corpus"):
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
