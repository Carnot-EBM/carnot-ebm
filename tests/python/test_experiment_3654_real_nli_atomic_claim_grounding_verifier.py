"""Tests for Exp 3654 real NLI atomic-claim grounding verifier.

Spec: REQ-VERIFY-3654, SCENARIO-VERIFY-3654.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from carnot.verify import nli_atomic_claim_grounding_verifier as mod
from carnot.verify.nli_atomic_claim_grounding_verifier import (
    REQUIRED_ARTIFACT_FIELDS,
    TEXT_STATISTICAL_PROXY_SUBSTRATE,
    TextStatisticalEntailmentProxy,
    build_artifact,
    score_corpus_rows,
    split_atomic_claims,
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


def _factual_rows() -> list[dict[str, Any]]:
    return [
        {
            "question": "Which city is the capital of France?",
            "answer": "Paris is the capital of France.",
            "evidence_passage": "France's capital city is Paris, which sits on the Seine.",
            "is_hallucination": 0,
            "model_confidence": 0.78,
        },
        {
            "question": "Which river is associated with Egypt?",
            "answer": "The Nile flows through Egypt.",
            "evidence_passage": "The Nile is the major river commonly associated with Egypt.",
            "is_hallucination": 0,
            "model_confidence": 0.62,
        },
        {
            "question": "Which city is the capital of France?",
            "answer": "Berlin is the capital of France.",
            "evidence_passage": "France's capital city is Paris, while Berlin is in Germany.",
            "is_hallucination": 1,
            "model_confidence": 0.81,
        },
        {
            "question": "Which river is associated with Egypt?",
            "answer": "The Amazon is the longest river in Egypt.",
            "evidence_passage": "The Nile is the major river commonly associated with Egypt.",
            "is_hallucination": 1,
            "model_confidence": 0.51,
        },
    ]


def _seed_v3_fixture(root: Path, rows: list[dict[str, Any]] | None = None) -> None:
    corpus_path = root / "data/realistic_factual_corpus_v3.jsonl"
    _write_jsonl(corpus_path, rows if rows is not None else _factual_rows())
    _write_json(
        root / "results/experiment_3640_build_factual_corpus_v3.json",
        {
            "corpus_path_used": "data/realistic_factual_corpus_v3.jsonl",
            "facts_corpus_validated": True,
            "confidence_baseline_auroc_on_corpus": 0.744576,
        },
    )


class StaticVerifier:
    def __init__(self, scores: list[float], *, model_based: bool, substrate: str) -> None:
        self._scores = list(scores)
        self.model_based = model_based
        self.nli_substrate = substrate
        self.calls: list[tuple[str, str]] = []

    def verify(self, model_answer: str, evidence_passage: str) -> float:
        self.calls.append((model_answer, evidence_passage))
        return self._scores.pop(0)


@pytest.mark.parametrize(
    (
        "honest_outcome",
        "seed_corpus",
        "verifier",
        "expected_built",
        "expected_leak_free",
    ),
    [
        (
            "built_model_based",
            True,
            StaticVerifier(
                [0.10, 0.80, 0.70, 0.90],
                model_based=True,
                substrate="model_based_transformers_checkpoint: fake-nli on cpu",
            ),
            True,
            True,
        ),
        (
            "built_proxy_disclosed",
            True,
            StaticVerifier(
                [0.10, 0.80, 0.70, 0.90],
                model_based=False,
                substrate=TEXT_STATISTICAL_PROXY_SUBSTRATE,
            ),
            True,
            True,
        ),
        (
            "leak_detected",
            True,
            StaticVerifier(
                [0.05, 0.10, 0.90, 0.95],
                model_based=True,
                substrate="model_based_transformers_checkpoint: fake-nli on cpu",
            ),
            True,
            False,
        ),
        (
            "blocked",
            False,
            StaticVerifier(
                [],
                model_based=True,
                substrate="model_based_transformers_checkpoint: fake-nli on cpu",
            ),
            False,
            False,
        ),
    ],
)
def test_exp3654_parametrizes_honest_outcomes(
    tmp_path: Path,
    honest_outcome: str,
    seed_corpus: bool,
    verifier: StaticVerifier,
    expected_built: bool,
    expected_leak_free: bool,
) -> None:
    """SCENARIO-VERIFY-3654: model/proxy/leak/blocked outcomes stay honest."""

    if seed_corpus:
        _seed_v3_fixture(tmp_path)

    artifact = build_artifact(
        tmp_path,
        verifier=verifier,
        started_s=10.0,
        now_s=13.0,
        n_bootstrap=12,
    )

    validate_artifact(artifact)
    assert set(REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert set(REQUIRED_ARTIFACT_FIELDS) <= set(artifact["field_principles"])
    assert type(artifact["nli_grounding_built"]) is bool
    assert type(artifact["grounding_leak_free"]) is bool
    assert artifact["nli_grounding_built"] is expected_built
    assert artifact["grounding_leak_free"] is expected_leak_free
    assert artifact["honest_outcome"] == honest_outcome

    if not seed_corpus:
        assert artifact["honest_verdict"] == "complete: blocked_no_v3_facts_corpus"
        assert artifact["grounding_auroc"] is None
        assert artifact["n_examples"] == 0
        assert verifier.calls == []
        return

    rows = _factual_rows()
    assert verifier.calls == [(row["answer"], row["evidence_passage"]) for row in rows]
    assert artifact["grounding_auroc"]["n"] == len(rows)
    assert len(artifact["grounding_auroc"]["bootstrap_seeds"]) >= 3
    assert artifact["confidence_baseline_auroc"] == 0.744576
    assert artifact["n_examples"] == len(rows)

    if honest_outcome == "built_proxy_disclosed":
        assert artifact["honest_verdict"].endswith("proxy_disclosed_no_checkpoint")
        assert artifact["nli_substrate"] == TEXT_STATISTICAL_PROXY_SUBSTRATE
    if honest_outcome == "leak_detected":
        assert artifact["honest_verdict"].endswith("leak_detected_untrusted")
        assert "auroc_at_or_above_0.99" in artifact["leak_diagnostics"]


def test_score_corpus_rows_uses_only_model_answer_and_evidence() -> None:
    """REQ-VERIFY-3654: verifier scoring has no label or gold-answer input path."""

    class GuardedRow(dict[str, Any]):
        def get(self, key: str, default: Any = None) -> Any:
            if key in {"is_hallucination", "gold_answer", "correct_answer"}:
                raise AssertionError(f"score path read forbidden key: {key}")
            return super().get(key, default)

    verifier = StaticVerifier(
        [0.25],
        model_based=True,
        substrate="model_based_transformers_checkpoint: fake-nli on cpu",
    )
    scores = score_corpus_rows(
        [
            GuardedRow(
                {
                    "answer": "Paris is the capital of France.",
                    "evidence_passage": "Paris is identified as France's capital city.",
                    "is_hallucination": 0,
                    "gold_answer": "Paris",
                }
            )
        ],
        verifier=verifier,
    )

    assert scores == [0.25]
    assert verifier.calls == [
        ("Paris is the capital of France.", "Paris is identified as France's capital city.")
    ]


def test_atomic_claim_splitter_uses_factual_text_not_placeholders() -> None:
    """REQ-VERIFY-3654: answers are decomposed into factual-looking claims."""

    claims = split_atomic_claims(
        "Paris is France's capital. It sits on the Seine, and it hosts the Louvre."
    )

    assert claims[0] == "Paris is France's capital"
    assert "It sits on the Seine" in claims
    assert "it hosts the Louvre" in claims
    assert split_atomic_claims("") == []
    assert mod._hypothesis_for_claim("   ") == ""


def test_disclosed_proxy_orders_supported_and_unsupported_claims() -> None:
    """REQ-VERIFY-3654: proxy fallback is explicitly disclosed and leak-free."""

    verifier = TextStatisticalEntailmentProxy()
    supported = verifier.verify(
        "Paris is the capital of France.",
        "France's capital city is Paris.",
    )
    unsupported = verifier.verify(
        "Berlin is the capital of France.",
        "France's capital city is Paris.",
    )

    assert verifier.model_based is False
    assert verifier.nli_substrate == TEXT_STATISTICAL_PROXY_SUBSTRATE
    assert supported < unsupported
    assert verifier.verify("", "France's capital city is Paris.") == 0.0
    assert verifier.verify("and", "and") == 0.0


def test_exp3654_validate_rejects_wrapped_grounding_bool(tmp_path: Path) -> None:
    """REQ-VERIFY-3654: nli_grounding_built is a bare top-level boolean gate."""

    _seed_v3_fixture(tmp_path)
    artifact = build_artifact(
        tmp_path,
        verifier=StaticVerifier(
            [0.10, 0.80, 0.70, 0.90],
            model_based=True,
            substrate="model_based_transformers_checkpoint: fake-nli on cpu",
        ),
        started_s=1.0,
        now_s=2.0,
        n_bootstrap=8,
    )
    artifact["nli_grounding_built"] = {"value": True}

    with pytest.raises(ValueError, match="nli_grounding_built"):
        validate_artifact(artifact)


def test_exp3654_checkpoint_loader_success_and_proxy_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SCENARIO-VERIFY-3654: cached model and no-checkpoint paths are both honest."""

    class FakeConfig:
        id2label = {0: "contradiction", 1: "entailment", 2: "neutral"}

    class FakeModel:
        config = FakeConfig()

    backend = mod.TransformersNLIBackend(
        "fake-checkpoint",
        tokenizer=object(),
        model=FakeModel(),
        torch_module=object(),
        device="cpu",
    )

    def fake_load(cls: type, checkpoint: str, *, device: str | None = None) -> Any:
        assert checkpoint == "fake-checkpoint"
        assert device == "cpu"
        return backend

    monkeypatch.setattr(
        mod.TransformersNLIBackend,
        "from_cached_checkpoint",
        classmethod(fake_load),
    )
    verifier = mod.NLIAtomicClaimGroundingVerifier.from_cached_or_proxy(
        checkpoint_candidates=("fake-checkpoint",),
        device="cpu",
    )
    assert isinstance(verifier, mod.NLIAtomicClaimGroundingVerifier)
    assert verifier.nli_substrate == "model_based_transformers_checkpoint: fake-checkpoint on cpu"

    def fail_load(cls: type, checkpoint: str, *, device: str | None = None) -> Any:
        raise OSError(f"{checkpoint} is not cached")

    monkeypatch.setattr(
        mod.TransformersNLIBackend,
        "from_cached_checkpoint",
        classmethod(fail_load),
    )
    proxy = mod.NLIAtomicClaimGroundingVerifier.from_cached_or_proxy(
        checkpoint_candidates=("missing-checkpoint",),
        allow_proxy=True,
    )
    assert isinstance(proxy, TextStatisticalEntailmentProxy)
    assert "missing-checkpoint" in str(proxy.unavailable_reason)

    with pytest.raises(RuntimeError, match="no cached NLI checkpoint"):
        mod.NLIAtomicClaimGroundingVerifier.from_cached_or_proxy(
            checkpoint_candidates=("missing-checkpoint",),
            allow_proxy=False,
        )


def test_exp3654_validation_and_leak_guard_branches(tmp_path: Path) -> None:
    """REQ-VERIFY-3654: invalid artifacts and gold-evidence leaks are rejected."""

    _seed_v3_fixture(tmp_path)
    artifact = build_artifact(
        tmp_path,
        verifier=StaticVerifier(
            [0.10, 0.80, 0.70, 0.90],
            model_based=True,
            substrate="model_based_transformers_checkpoint: fake-nli on cpu",
        ),
        started_s=1.0,
        now_s=2.0,
        n_bootstrap=8,
    )

    missing = dict(artifact)
    missing.pop("n_examples")
    with pytest.raises(ValueError, match="missing required artifact fields"):
        validate_artifact(missing)

    bad_verdict = dict(artifact, honest_verdict="failed")
    with pytest.raises(ValueError, match="honest_verdict"):
        validate_artifact(bad_verdict)

    bad_principles = dict(artifact, field_principles=None)
    with pytest.raises(ValueError, match="field_principles"):
        validate_artifact(bad_principles)

    incomplete_principles = dict(artifact, field_principles={})
    with pytest.raises(ValueError, match="field_principles"):
        validate_artifact(incomplete_principles)

    missing_metric = dict(artifact, grounding_auroc=None)
    with pytest.raises(ValueError, match="grounding_auroc"):
        validate_artifact(missing_metric)

    short_bootstrap = dict(artifact)
    short_bootstrap["grounding_auroc"] = dict(artifact["grounding_auroc"], bootstrap_seeds=[1, 2])
    with pytest.raises(ValueError, match="bootstrap"):
        validate_artifact(short_bootstrap)

    bad_point = dict(artifact)
    bad_point["grounding_auroc"] = dict(artifact["grounding_auroc"], point=None)
    with pytest.raises(ValueError, match="point"):
        validate_artifact(bad_point)

    bad_duration = dict(artifact, duration_s=-1.0)
    with pytest.raises(ValueError, match="duration_s"):
        validate_artifact(bad_duration)

    assert (
        mod.evidence_excludes_gold_answer(
            [
                {
                    "answer": "Paris",
                    "gold_answer": "France",
                    "evidence_passage": "The separate gold answer is France.",
                }
            ]
        )
        is False
    )
    assert mod.leak_diagnostics_for_run(
        evidence_excludes_gold=False,
        grounding_auroc=0.5,
        score_path_answer_evidence_only=False,
    ) == [
        "separate_gold_answer_found_in_evidence",
        "score_path_read_label_or_gold_field",
    ]


def test_exp3654_blocked_schema_and_write_paths(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-3654: invalid v3 corpora stay blocked without metrics."""

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    empty_corpus = data_dir / "realistic_factual_corpus_v3.jsonl"
    empty_corpus.write_text("", encoding="utf-8")
    artifact = build_artifact(tmp_path, started_s=1.0, now_s=2.0)
    assert artifact["honest_outcome"] == "blocked"
    assert artifact["blocked_reason"] == "blocked_empty_v3_facts_corpus"

    _write_jsonl(empty_corpus, [{"answer": "Paris"}])
    artifact = build_artifact(tmp_path, started_s=1.0, now_s=2.0)
    assert artifact["blocked_reason"].startswith("blocked_v3_facts_corpus_schema")

    output = mod.write_artifact(tmp_path, output_path="results/exp3654_blocked.json")
    assert output.exists()
    written = json.loads(output.read_text(encoding="utf-8"))
    assert written["honest_verdict"] == "complete: blocked_no_v3_facts_corpus"


def test_exp3654_fallback_baseline_and_path_helpers(tmp_path: Path) -> None:
    """REQ-VERIFY-3654: missing metadata and odd confidence values are deterministic."""

    rows = _factual_rows()
    rows[0]["model_confidence"] = "not-a-number"
    rows[1]["model_confidence"] = float("nan")
    _write_jsonl(tmp_path / "data/realistic_factual_corpus_v3.jsonl", rows)
    _write_json(
        tmp_path / "results/experiment_3640_build_factual_corpus_v3.json",
        ["not", "a", "mapping"],
    )
    artifact = build_artifact(
        tmp_path,
        verifier=StaticVerifier(
            [0.10, 0.80, 0.70, 0.90],
            model_based=True,
            substrate="model_based_transformers_checkpoint: fake-nli on cpu",
        ),
        started_s=1.0,
        now_s=2.0,
        n_bootstrap=8,
    )
    assert (
        artifact["confidence_baseline_auroc"] == artifact["confidence_baseline_measured"]["point"]
    )

    absolute_corpus = tmp_path.parent / f"{tmp_path.name}_absolute_facts.jsonl"
    _write_jsonl(absolute_corpus, _factual_rows())
    _write_json(
        tmp_path / "results/experiment_3640_build_factual_corpus_v3.json",
        {
            "corpus_path_used": str(absolute_corpus),
            "facts_corpus_validated": True,
            "confidence_baseline_auroc_on_corpus": 0.6,
        },
    )
    artifact = build_artifact(
        tmp_path,
        verifier=StaticVerifier(
            [0.10, 0.80, 0.70, 0.90],
            model_based=True,
            substrate="model_based_transformers_checkpoint: fake-nli on cpu",
        ),
        started_s=1.0,
        now_s=2.0,
        n_bootstrap=8,
    )
    assert artifact["corpus_path_used"] == str(absolute_corpus)


def test_exp3654_terminal_verdict_edges() -> None:
    """SCENARIO-VERIFY-3654: model-backed non-leak outcomes compare honestly."""

    assert mod.terminal_verdict(
        model_based=True,
        grounding_leak_free=True,
        beats_proxy=True,
        beats_confidence=False,
    ).endswith("beats_proxy_not_confidence_facts_still_hard")
    assert mod.terminal_verdict(
        model_based=True,
        grounding_leak_free=True,
        beats_proxy=False,
        beats_confidence=False,
    ).endswith("does_not_beat_proxy_facts_still_hard")
