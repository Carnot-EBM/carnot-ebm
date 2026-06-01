"""Tests for Exp 3659 real-substrate trained EBM judge OOD retry.

Spec: REQ-VERIFY-3659, SCENARIO-VERIFY-3659.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

# Pre-warm torch at collection time so the repo RSS watchdog does not charge
# one-time CUDA/optimizer imports to whichever test first trains the ranker.
try:  # pragma: no cover - defensive for stripped-down environments.
    import torch

    _warm_layer = torch.nn.Linear(1, 1)
    _warm_optim = torch.optim.AdamW(_warm_layer.parameters(), lr=0.01)
    _warm_optim.zero_grad(set_to_none=True)
except Exception:
    pass

from carnot.verify.trained_ebm_judge_ood_counterpoint_v2 import JudgeExample
from carnot.verify import trained_ebm_judge_ood_real_substrate_v3 as mod
from carnot.verify.trained_ebm_judge_ood_real_substrate_v3 import (
    REQUIRED_ARTIFACT_FIELDS,
    TOY_HEAD_OOD_REFERENCE_AUROC,
    TorchEnergyRanker,
    build_artifact,
    real_substrate_delta,
    select_device,
    terminal_verdict,
    validate_artifact,
    write_artifact,
)


class FixtureEmbeddingProvider:
    """Small deterministic embedding fixture for REQ-VERIFY-3659 tests."""

    def __init__(self, *, available: bool = True) -> None:
        self.available = available

    def encode_examples(self, examples: list[JudgeExample]) -> np.ndarray:
        if not self.available:
            raise RuntimeError("fixture substrate unavailable")
        return np.asarray(
            [
                [
                    example.validity_signal,
                    1.0 - example.validity_signal,
                    example.confidence_error_signal,
                    float(len(example.text) % 5) / 5.0,
                ]
                for example in examples
            ],
            dtype=np.float32,
        )

    def substrate_report(self) -> dict[str, Any]:
        return {
            "kind": "fixture_real_embedding_substrate",
            "model_id": "fixture/real-transformer-embeddings",
            "device": "cpu",
            "available": self.available,
        }


def _examples(
    domain: str,
    pairs: list[tuple[int, float, float]],
    *,
    prefix: str,
) -> list[JudgeExample]:
    return [
        JudgeExample(
            domain=domain,
            text=f"{prefix} row {idx}",
            error_label=label,
            validity_signal=validity,
            confidence_error_signal=confidence,
        )
        for idx, (label, validity, confidence) in enumerate(pairs)
    ]


def _math_fixture() -> list[JudgeExample]:
    rows: list[tuple[int, float, float]] = []
    for i in range(24):
        rows.append((1, 0.88 + (i % 3) * 0.02, 0.5))
        rows.append((0, 0.08 + (i % 3) * 0.02, 0.5))
    return _examples("math", rows, prefix="math reasoning")


@pytest.mark.parametrize(
    (
        "ood_examples_by_domain",
        "feature_provider",
        "expected_verdict",
        "expected_transfer",
        "expect_ood_metric",
    ),
    [
        (
            {
                "code": _examples(
                    "code",
                    [(1, 0.95, 0.5), (1, 0.86, 0.5), (0, 0.12, 0.5), (0, 0.18, 0.5)],
                    prefix="code transfer",
                ),
                "facts": _examples(
                    "facts",
                    [(1, 0.92, 0.5), (1, 0.82, 0.5), (0, 0.10, 0.5), (0, 0.22, 0.5)],
                    prefix="facts transfer",
                ),
            },
            FixtureEmbeddingProvider(),
            "complete: real_substrate_trained_judge_transfers_ood_resourcing_was_the_bottleneck",
            True,
            True,
        ),
        (
            {
                "code": _examples(
                    "code",
                    [(1, 0.10, 0.95), (1, 0.18, 0.90), (0, 0.90, 0.05), (0, 0.82, 0.10)],
                    prefix="code reversed",
                )
            },
            FixtureEmbeddingProvider(),
            "complete: real_substrate_trained_judge_also_math_only_trained_judge_not_the_cross_domain_fix",
            False,
            True,
        ),
        (
            {},
            FixtureEmbeddingProvider(),
            "complete: blocked_no_ood_eval_corpus",
            False,
            False,
        ),
        (
            {"facts": _examples("facts", [(1, 0.9, 0.5), (0, 0.1, 0.5)], prefix="facts")},
            None,
            "complete: blocked_no_trainable_substrate",
            False,
            False,
        ),
    ],
)
def test_exp3659_parametrizes_transfer_null_and_blocked_verdicts(
    ood_examples_by_domain: dict[str, list[JudgeExample]],
    feature_provider: FixtureEmbeddingProvider | None,
    expected_verdict: str,
    expected_transfer: bool,
    expect_ood_metric: bool,
) -> None:
    """SCENARIO-VERIFY-3659: transfer, null, and blocked outcomes are all honest."""

    artifact = build_artifact(
        math_examples=_math_fixture(),
        ood_examples_by_domain=ood_examples_by_domain,
        feature_provider=feature_provider,
        started_s=0.0,
        now_s=12.0,
        seeds=(3659, 3660, 3661),
        epochs=120,
        device="cpu",
        force_no_trainable_substrate=feature_provider is None,
    )

    validate_artifact(artifact)
    assert artifact["honest_verdict"] == expected_verdict
    assert artifact["trained_judge_transfers_ood"] is expected_transfer
    assert type(artifact["trained_judge_transfers_ood"]) is bool
    assert (artifact["ood_judge_auroc"] is not None) is expect_ood_metric
    assert set(REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert set(REQUIRED_ARTIFACT_FIELDS) <= set(artifact["field_principles"])
    assert artifact["acceptance_gate"]["passed"] is expect_ood_metric
    if expected_transfer:
        assert artifact["ood_judge_auroc"] > artifact["shuffled_label_control_auroc"]
        assert artifact["ood_judge_auroc"] > artifact["confidence_only_baseline_auroc"]


@pytest.mark.parametrize(
    ("no_substrate", "ood_domains", "transfers", "expected"),
    [
        (True, ["code"], True, "complete: blocked_no_trainable_substrate"),
        (False, [], True, "complete: blocked_no_ood_eval_corpus"),
        (
            False,
            ["code"],
            True,
            "complete: real_substrate_trained_judge_transfers_ood_resourcing_was_the_bottleneck",
        ),
        (
            False,
            ["facts"],
            False,
            "complete: real_substrate_trained_judge_also_math_only_trained_judge_not_the_cross_domain_fix",
        ),
    ],
)
def test_exp3659_terminal_verdict_ladder(
    no_substrate: bool,
    ood_domains: list[str],
    transfers: bool,
    expected: str,
) -> None:
    """REQ-VERIFY-3659: terminal verdicts cover transfer, null, and blocked paths."""

    assert terminal_verdict(
        no_trainable_substrate=no_substrate,
        ood_domains_tested=ood_domains,
        trained_judge_transfers_ood=transfers,
    ) == expected


def test_exp3659_torch_ranker_learns_separable_energy() -> None:
    """REQ-VERIFY-3659: the trainable torch energy head learns a ranking signal."""

    X = np.asarray(
        [[0.9, 0.1, 0.5], [0.8, 0.2, 0.5], [0.1, 0.9, 0.5], [0.2, 0.8, 0.5]],
        dtype=np.float32,
    )
    y = [1, 1, 0, 0]
    ranker = TorchEnergyRanker(input_dim=3, epochs=120, lr=0.2, device="cpu").fit(X, y, seed=7)
    scores = ranker.predict_scores(X)
    assert scores[0] > scores[2]
    assert scores[1] > scores[3]
    assert ranker.n_params == 4


def test_exp3659_torch_ranker_rejects_bad_calls() -> None:
    """REQ-VERIFY-3659: malformed training and prediction calls fail explicitly."""

    ranker = TorchEnergyRanker(input_dim=3, device="cpu")
    with pytest.raises(RuntimeError):
        ranker.predict_scores(np.zeros((2, 3), dtype=np.float32))
    with pytest.raises(ValueError):
        ranker.fit(np.zeros((2, 2), dtype=np.float32), [0, 1], seed=1)
    with pytest.raises(ValueError):
        ranker.fit(np.zeros((2, 3), dtype=np.float32), [1], seed=1)
    fitted = TorchEnergyRanker(input_dim=3, epochs=5, device="cpu").fit(
        np.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32),
        [1, 0],
        seed=2,
    )
    with pytest.raises(ValueError):
        fitted.predict_scores(np.zeros((2, 2), dtype=np.float32))


def test_exp3659_default_load_and_provider_resolution_paths(tmp_path: Path, monkeypatch) -> None:
    """REQ-VERIFY-3659: default corpus loading and provider resolution are covered."""

    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "fover_corpus.jsonl").write_text(
        "\n".join(
            [
                json.dumps({"step_text": "1 + 1 = 3", "label": "incorrect"}),
                json.dumps({"step_text": "2 + 2 = 5", "label": "incorrect"}),
                json.dumps({"step_text": "1 + 1 = 2", "label": "correct"}),
                json.dumps({"step_text": "2 + 2 = 4", "label": "correct"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(mod, "default_feature_provider", lambda *, device=None: FixtureEmbeddingProvider())

    artifact = build_artifact(
        tmp_path,
        started_s=0.0,
        now_s=3.0,
        seeds=(1, 2, 3),
        epochs=10,
        device="cpu",
    )

    validate_artifact(artifact)
    assert artifact["honest_verdict"] == "complete: blocked_no_ood_eval_corpus"
    assert artifact["n_examples_per_domain"]["math"] == 4


def test_exp3659_feature_matrix_rejects_bad_embedding_shape() -> None:
    """REQ-VERIFY-3659: embedding providers must return one row per example."""

    class BadProvider(FixtureEmbeddingProvider):
        def encode_examples(self, examples: list[JudgeExample]) -> np.ndarray:
            return np.zeros((1, 2), dtype=np.float32)

    with pytest.raises(ValueError):
        mod.feature_matrix(_math_fixture()[:2], BadProvider())


def test_exp3659_write_artifact_with_injected_provider(tmp_path: Path) -> None:
    """REQ-VERIFY-3659: write_artifact persists a schema-valid artifact."""

    output = write_artifact(
        tmp_path,
        output_path=Path("results/out.json"),
        math_examples=_math_fixture(),
        ood_examples_by_domain={},
        feature_provider=FixtureEmbeddingProvider(),
        tests_run=["pytest synthetic"],
        started_s=0.0,
        now_s=2.0,
        device="cpu",
    )
    artifact = json.loads(output.read_text(encoding="utf-8"))

    validate_artifact(artifact)
    assert artifact["honest_verdict"] == "complete: blocked_no_ood_eval_corpus"
    assert artifact["tests_run"] == ["pytest synthetic"]


@pytest.mark.parametrize(
    "mutator",
    [
        lambda artifact: artifact.pop("judge_substrate"),
        lambda artifact: artifact.pop("field_principles"),
        lambda artifact: artifact["field_principles"].pop("judge_substrate"),
        lambda artifact: artifact.__setitem__("honest_verdict", "complete: bogus"),
        lambda artifact: artifact.__setitem__("trained_judge_transfers_ood", {"wrapped": False}),
        lambda artifact: artifact.__setitem__("judge_substrate", []),
        lambda artifact: artifact.pop("acceptance_gate"),
        lambda artifact: artifact.__setitem__("duration_s", -1),
    ],
)
def test_exp3659_validate_rejects_schema_poison(mutator) -> None:
    """REQ-VERIFY-3659: artifact validation rejects poisoned schema variants."""

    artifact = build_artifact(
        math_examples=_math_fixture(),
        ood_examples_by_domain={},
        feature_provider=FixtureEmbeddingProvider(),
        started_s=0.0,
        now_s=1.0,
        device="cpu",
    )
    mutator(artifact)
    with pytest.raises(ValueError):
        validate_artifact(artifact)


def test_exp3659_delta_vs_toy_reference() -> None:
    """REQ-VERIFY-3659: real-substrate delta is anchored to Exp 3646's toy head."""

    assert real_substrate_delta(None) is None
    assert real_substrate_delta(TOY_HEAD_OOD_REFERENCE_AUROC + 0.01) == pytest.approx(0.01)


def test_exp3659_device_helpers_cover_cuda_unavailable_branch(monkeypatch) -> None:
    """REQ-VERIFY-3659: device reporting is explicit even without torch."""

    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "torch":
            raise ImportError("blocked torch")
        return real_import(name, *args, **kwargs)

    assert select_device("cpu") == "cpu"
    monkeypatch.setattr(builtins, "__import__", fake_import)
    assert mod.cuda_available() is False
    assert select_device(None) == "cpu"
