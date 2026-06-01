"""Tests for Exp 3646 trained EBM judge OOD counterpoint.

Spec: REQ-VERIFY-3646, SCENARIO-VERIFY-3646.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from carnot.verify.trained_ebm_judge_ood_counterpoint_v2 import (
    REQUIRED_ARTIFACT_FIELDS,
    JudgeExample,
    SmallEnergyJudge,
    _apply_arithmetic,
    _coerce_float,
    _read_json_object,
    _read_jsonl,
    arithmetic_error_rate,
    build_artifact,
    code_confidence_error_signal,
    delta_vs_fixed,
    repeated_token_fraction,
    tie_aware_auroc,
    load_corpora,
    terminal_verdict,
    validate_artifact,
    write_artifact,
)


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
    for i in range(18):
        rows.append((1, 0.85 + (i % 3) * 0.03, 0.0))
        rows.append((0, 0.05 + (i % 3) * 0.03, 0.0))
    return _examples("math", rows, prefix="math reasoning")


@pytest.mark.parametrize(
    (
        "ood_examples_by_domain",
        "force_no_trainable_substrate",
        "expected_verdict",
        "expected_transfer",
        "expect_ood_metric",
    ),
    [
        (
            {
                "code": _examples(
                    "code",
                    [(1, 0.9, 0.5), (1, 0.8, 0.5), (0, 0.1, 0.5), (0, 0.2, 0.5)],
                    prefix="code transfer",
                ),
                "facts": _examples(
                    "facts",
                    [(1, 0.88, 0.5), (1, 0.78, 0.5), (0, 0.15, 0.5), (0, 0.25, 0.5)],
                    prefix="facts transfer",
                ),
            },
            False,
            "complete: trained_ebm_judge_transfers_ood_fixed_ensemble_was_the_bottleneck",
            True,
            True,
        ),
        (
            {
                "code": _examples(
                    "code",
                    [(1, 0.1, 0.9), (1, 0.2, 0.8), (0, 0.9, 0.1), (0, 0.8, 0.2)],
                    prefix="code reversed",
                )
            },
            False,
            "complete: trained_ebm_judge_also_math_only_transfer_not_a_training_artifact",
            False,
            True,
        ),
        ({}, False, "complete: blocked_no_ood_eval_corpus", False, False),
        (
            {"facts": _examples("facts", [(1, 0.9, 0.5), (0, 0.1, 0.5)], prefix="facts")},
            True,
            "complete: blocked_no_trainable_substrate",
            False,
            False,
        ),
    ],
)
def test_exp3646_parametrizes_transfer_null_and_blocked_verdicts(
    ood_examples_by_domain: dict[str, list[JudgeExample]],
    force_no_trainable_substrate: bool,
    expected_verdict: str,
    expected_transfer: bool,
    expect_ood_metric: bool,
) -> None:
    """SCENARIO-VERIFY-3646: verdicts are not hard-coded to the success case."""

    artifact = build_artifact(
        math_examples=_math_fixture(),
        ood_examples_by_domain=ood_examples_by_domain,
        started_s=0.0,
        now_s=9.0,
        seeds=(101, 102, 103),
        epochs=120,
        force_no_trainable_substrate=force_no_trainable_substrate,
    )

    validate_artifact(artifact)
    assert artifact["honest_verdict"] == expected_verdict
    assert artifact["trained_judge_transfers_ood"] is expected_transfer
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
            "complete: trained_ebm_judge_transfers_ood_fixed_ensemble_was_the_bottleneck",
        ),
        (
            False,
            ["facts"],
            False,
            "complete: trained_ebm_judge_also_math_only_transfer_not_a_training_artifact",
        ),
    ],
)
def test_exp3646_terminal_verdict_ladder(
    no_substrate: bool,
    ood_domains: list[str],
    transfers: bool,
    expected: str,
) -> None:
    """REQ-VERIFY-3646: terminal verdicts cover transfer, null, and blocked paths."""

    assert terminal_verdict(
        no_trainable_substrate=no_substrate,
        ood_domains_tested=ood_domains,
        trained_judge_transfers_ood=transfers,
    ) == expected


def test_exp3646_small_energy_judge_learns_separable_signal() -> None:
    """REQ-VERIFY-3646: the tiny trainable substrate learns a ranking head."""

    X = [[0.9, 0.0, 1.0, 1.0, 0.0], [0.8, 0.0, 1.0, 1.0, 0.0], [0.1, 0.0, 1.0, 1.0, 0.0], [0.2, 0.0, 1.0, 1.0, 0.0]]
    y = [1, 1, 0, 0]
    judge = SmallEnergyJudge(epochs=100, lr=0.5).fit(X, y, seed=7)
    scores = judge.predict_scores(X)
    assert scores[0] > scores[2]
    assert scores[1] > scores[3]
    assert judge.n_params == 6


def test_exp3646_small_energy_judge_guards_bad_calls() -> None:
    """REQ-VERIFY-3646: malformed feature matrices fail explicitly."""

    judge = SmallEnergyJudge()
    with pytest.raises(RuntimeError):
        judge.predict_scores([[0.0] * 5])
    with pytest.raises(ValueError):
        judge.fit([[0.0, 1.0]], [1], seed=1)


def test_exp3646_load_corpora_from_upstream_artifact_shapes(tmp_path: Path) -> None:
    """REQ-VERIFY-3646: cached math/code/facts corpora are normalized to examples."""

    (tmp_path / "data").mkdir()
    (tmp_path / "results").mkdir()
    (tmp_path / "data" / "fover_corpus.jsonl").write_text(
        "\n".join(
            [
                json.dumps({"step_text": "2 + 2 = 5", "label": "incorrect", "confidence": 1.0}),
                json.dumps({"step_text": "2 + 2 = 4", "label": "correct", "confidence": 1.0}),
                json.dumps({"step_text": "ignored", "label": "unknown"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (tmp_path / "data" / "code.jsonl").write_text(
        "\n".join(
            [
                json.dumps({"candidate_code": "def f(:\n pass", "label": False}),
                json.dumps({"candidate_code": "def f():\n    return 1\n", "label": True}),
                json.dumps({"candidate_code": "missing label"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (tmp_path / "data" / "facts.jsonl").write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "answer": "Berlin",
                        "evidence_passage": "Paris is the capital of France.",
                        "is_hallucination": 1,
                        "model_confidence": 0.4,
                        "question": "Capital?",
                    }
                ),
                json.dumps(
                    {
                        "answer": "Paris",
                        "evidence_passage": "Paris is the capital of France.",
                        "is_hallucination": 0,
                        "model_confidence": 0.8,
                        "question": "Capital?",
                    }
                ),
                json.dumps({"answer": "missing label"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (tmp_path / "results" / "experiment_3641_code_corpus_verifiers_fire_transfer_v3.json").write_text(
        json.dumps({"code_corpus_path": "data/code.jsonl", "code_verifiers_fire": True}),
        encoding="utf-8",
    )
    (tmp_path / "results" / "experiment_3640_build_factual_corpus_v3.json").write_text(
        json.dumps({"corpus_path_used": "data/facts.jsonl", "facts_corpus_validated": True}),
        encoding="utf-8",
    )

    math_examples, ood = load_corpora(tmp_path, max_math_examples=20)

    assert [example.domain for example in math_examples] == ["math", "math"]
    assert set(ood) == {"code", "facts"}
    assert [example.error_label for example in ood["code"]] == [1, 0]
    assert [example.error_label for example in ood["facts"]] == [1, 0]


def test_exp3646_build_and_write_artifact_loads_default_corpora(tmp_path: Path) -> None:
    """REQ-VERIFY-3646: write_artifact covers the script's default load path."""

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

    output = write_artifact(
        tmp_path,
        output_path=Path("results/out.json"),
        tests_run=["pytest synthetic"],
    )
    artifact = json.loads(output.read_text(encoding="utf-8"))

    validate_artifact(artifact)
    assert artifact["honest_verdict"] == "complete: blocked_no_ood_eval_corpus"
    assert artifact["tests_run"] == ["pytest synthetic"]


@pytest.mark.parametrize(
    ("results_payload", "expected_code", "expected_facts"),
    [
        ({}, [], []),
        (
            {
                "experiment_3641_code_corpus_verifiers_fire_transfer_v3.json": {
                    "code_verifiers_fire": False
                },
                "experiment_3640_build_factual_corpus_v3.json": {
                    "facts_corpus_validated": False
                },
            },
            [],
            [],
        ),
        (
            {
                "experiment_3641_code_corpus_verifiers_fire_transfer_v3.json": {
                    "code_verifiers_fire": True
                },
                "experiment_3640_build_factual_corpus_v3.json": {
                    "facts_corpus_validated": True
                },
            },
            [],
            [],
        ),
    ],
)
def test_exp3646_load_corpora_blocks_missing_or_unrunnable_ood(
    tmp_path: Path,
    results_payload: dict[str, dict],
    expected_code: list[JudgeExample],
    expected_facts: list[JudgeExample],
) -> None:
    """REQ-VERIFY-3646: blocked upstream OOD rows stay blocked."""

    (tmp_path / "data").mkdir()
    (tmp_path / "results").mkdir()
    (tmp_path / "data" / "fover_corpus.jsonl").write_text("", encoding="utf-8")
    for name, payload in results_payload.items():
        (tmp_path / "results" / name).write_text(json.dumps(payload), encoding="utf-8")

    _math, ood = load_corpora(tmp_path)

    assert ood.get("code", []) == expected_code
    assert ood.get("facts", []) == expected_facts


def test_exp3646_metric_and_signal_edge_cases(tmp_path: Path) -> None:
    """REQ-VERIFY-3646: helpers are total on degenerate control inputs."""

    assert code_confidence_error_signal({"metadata": {"syntax_success": False}}) == pytest.approx(0.75)
    assert code_confidence_error_signal(
        {"metadata": {"syntax_success": True, "runtime_success": True}}
    ) == pytest.approx(0.25)
    assert arithmetic_error_rate("no equation here") == pytest.approx(0.0)
    assert repeated_token_fraction("!!!") == pytest.approx(0.0)
    assert tie_aware_auroc([1, 1], [0.2, 0.3]) == pytest.approx(0.5)
    assert tie_aware_auroc([1, 0], [0.5, 0.5]) == pytest.approx(0.5)
    assert delta_vs_fixed(None) is None
    assert _coerce_float("bad", 0.25) == pytest.approx(0.25)
    assert _coerce_float(float("nan"), 0.25) == pytest.approx(0.25)
    assert _apply_arithmetic(3.0, "-", 1.0) == pytest.approx(2.0)
    assert _apply_arithmetic(3.0, "*", 2.0) == pytest.approx(6.0)
    assert _apply_arithmetic(8.0, "/", 2.0) == pytest.approx(4.0)
    assert _apply_arithmetic(8.0, "/", 0.0) is None
    assert _apply_arithmetic(8.0, "?", 2.0) is None

    missing = tmp_path / "missing.json"
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    list_path = tmp_path / "list.json"
    list_path.write_text("[]", encoding="utf-8")
    assert _read_json_object(missing) == {}
    assert _read_json_object(bad) == {}
    assert _read_json_object(list_path) == {}

    bad_jsonl = tmp_path / "bad.jsonl"
    bad_jsonl.write_text("\n{bad}\n[]\n{}\n", encoding="utf-8")
    assert _read_jsonl(tmp_path / "missing.jsonl") == []
    assert _read_jsonl(bad_jsonl) == [{}]


@pytest.mark.parametrize(
    "mutator",
    [
        lambda artifact: artifact.pop("judge_recipe"),
        lambda artifact: artifact.pop("field_principles"),
        lambda artifact: artifact["field_principles"].pop("judge_recipe"),
        lambda artifact: artifact.__setitem__("honest_verdict", "complete: bogus"),
        lambda artifact: artifact.__setitem__("trained_judge_transfers_ood", {"wrapped": False}),
        lambda artifact: artifact.pop("acceptance_gate"),
        lambda artifact: artifact.__setitem__("duration_s", -1),
    ],
)
def test_exp3646_validate_rejects_schema_violations(mutator) -> None:
    """REQ-VERIFY-3646: artifact contract rejects schema poison variants."""

    artifact = build_artifact(
        math_examples=_math_fixture(),
        ood_examples_by_domain={},
        started_s=0.0,
        now_s=1.0,
    )
    mutator(artifact)
    with pytest.raises(ValueError):
        validate_artifact(artifact)


def test_exp3646_validate_rejects_missing_required_field() -> None:
    """REQ-VERIFY-3646: artifact contract rejects incomplete value storage."""

    artifact = build_artifact(
        math_examples=_math_fixture(),
        ood_examples_by_domain={},
        started_s=0.0,
        now_s=1.0,
    )
    artifact.pop("judge_recipe")
    with pytest.raises(ValueError):
        validate_artifact(artifact)
