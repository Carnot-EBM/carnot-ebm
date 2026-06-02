"""Tests for Exp 3684 product value against self-certainty.

Spec: REQ-SPOE-3684, SCENARIO-SPOE-3684.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from carnot.pipeline import product_value_vs_self_certainty_3684 as exp


def _domain_examples(
    domain: str,
    outcome: str,
    *,
    n: int = 80,
) -> list[exp.RebaselineExample]:
    examples: list[exp.RebaselineExample] = []
    for idx in range(n):
        label = 1 if idx < n // 2 else 0
        if outcome == "ensemble_adds_value_over_self_certainty":
            ensemble = 0.94 - 0.002 * idx if label else 0.06 + 0.001 * (idx - n // 2)
            self_certainty = 0.5
            confidence = 0.5
        elif outcome == "value_collapses_vs_stronger_baseline":
            ensemble = 0.5
            self_certainty = 0.94 - 0.002 * idx if label else 0.06 + 0.001 * (idx - n // 2)
            confidence = self_certainty
        else:  # pragma: no cover - guarded by parametrization choices.
            raise ValueError(outcome)
        examples.append(
            exp.RebaselineExample(
                domain=domain,
                label=label,
                ensemble_energy=ensemble,
                confidence_error=confidence,
                self_certainty_error=self_certainty,
                example_id=f"{domain}-{outcome}-{idx}",
            )
        )
    return examples


@pytest.mark.parametrize(
    ("case_name", "examples", "expected_verdict", "expected_adds_value"),
    [
        (
            "ensemble_adds_value_over_self_certainty",
            _domain_examples("math", "ensemble_adds_value_over_self_certainty"),
            "complete: ensemble_adds_value_over_self_certainty_product_value_robust",
            True,
        ),
        (
            "value_collapses_vs_stronger_baseline",
            _domain_examples("math", "value_collapses_vs_stronger_baseline"),
            "complete: product_value_collapses_vs_self_certainty_claim_narrowed",
            False,
        ),
        (
            "blocked",
            [],
            "complete: blocked_no_labeled_corpus_for_rebaseline",
            False,
        ),
    ],
)
def test_scenario_spoe_3684_parametrized_honest_outcomes(
    case_name: str,
    examples: list[exp.RebaselineExample],
    expected_verdict: str,
    expected_adds_value: bool,
) -> None:
    """SCENARIO-SPOE-3684: robust, collapsed, and blocked outcomes are distinct."""

    artifact = exp.build_artifact_from_examples(
        examples,
        started_s=1.0,
        now_s=2.25,
        seeds=[11],
        n_bootstrap=16,
        tests_run=[f"SCENARIO-SPOE-3684 {case_name}"],
    )

    exp.validate_artifact(artifact)
    assert artifact["honest_verdict"] == expected_verdict
    assert artifact["ensemble_adds_value_over_self_certainty"] is expected_adds_value
    assert type(artifact["ensemble_adds_value_over_self_certainty"]) is bool
    assert artifact["duration_s"] == pytest.approx(1.25)
    assert artifact["tests_run"] == [f"SCENARIO-SPOE-3684 {case_name}"]
    assert set(exp.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)


def test_req_spoe_3684_self_certainty_proxy_and_metric_helpers() -> None:
    """REQ-SPOE-3684: self-certainty and paired-delta helpers are measured."""

    uniform = exp.token_distribution_self_certainty([[0.5, 0.5]])
    peaked = exp.token_distribution_self_certainty([[0.9, 0.1]])
    empty_distribution = exp.token_distribution_self_certainty([[0.0, float("nan")]])
    assert uniform == pytest.approx(0.0)
    assert empty_distribution == pytest.approx(0.0)
    assert peaked > uniform

    proxy = exp.self_certainty_error_proxy_from_confidence_errors([0.1, 0.5, 0.9])
    assert proxy[0] < proxy[1] < proxy[2]

    labels = [1, 1, 0, 0, 1, 0]
    fused = [0.95, 0.88, 0.05, 0.12, 0.84, 0.2]
    self_certainty = [0.5 for _ in labels]
    ensemble = fused
    confidence = [0.4 for _ in labels]

    auroc = exp.auroc_metric(labels, fused, seeds=[3], n_bootstrap=8)
    delta = exp.paired_auroc_delta_metric(
        labels,
        fused,
        self_certainty,
        seeds=[3],
        n_bootstrap=8,
    )
    recall = exp.recall_at_fixed_fpr_table(
        labels,
        fused_scores=fused,
        self_certainty_scores=self_certainty,
        ensemble_scores=ensemble,
        confidence_scores=confidence,
    )

    assert auroc["point"] == pytest.approx(1.0)
    assert delta["point"] > 0.0
    assert exp.delta_ci_excludes_zero_positive(delta) is True
    assert set(recall) == {"0.05", "0.10", "0.20"}
    assert recall["0.10"]["fused_recall"] >= recall["0.10"]["self_certainty_recall"]

    assert exp.auroc_metric([1, 1], [0.8, 0.7], seeds=[3]) == exp.empty_metric([3])
    no_bootstrap = exp.auroc_metric(labels, fused, seeds=[3], n_bootstrap=0)
    no_bootstrap_delta = exp.paired_auroc_delta_metric(
        labels,
        fused,
        self_certainty,
        seeds=[3],
        n_bootstrap=0,
    )
    blocked_delta = exp.paired_auroc_delta_metric([1, 1], [0.9, 0.8], [0.5, 0.5], seeds=[3])
    assert no_bootstrap["ci95"] == [1.0, 1.0]
    assert no_bootstrap_delta["ci95"][0] == no_bootstrap_delta["point"]
    assert blocked_delta["delta_ci_excludes_zero"] is False
    assert exp.delta_ci_excludes_zero_positive({"point": None, "ci95": None}) is False
    assert exp._round(math.inf) == math.inf


def test_req_spoe_3684_validation_and_write_artifact(tmp_path: Path) -> None:
    """REQ-SPOE-3684: artifact schema, bare bool, and write path are strict."""

    output = exp.write_artifact_from_examples(
        tmp_path,
        output_path="results/exp3684.json",
        examples=_domain_examples("math", "ensemble_adds_value_over_self_certainty"),
        started_s=0.0,
        now_s=1.0,
        seeds=[5],
        n_bootstrap=8,
    )
    artifact = json.loads(output.read_text(encoding="utf-8"))

    exp.validate_artifact(artifact)
    assert artifact["acceptance_gate"]["passed"] is True
    assert artifact["ensemble_adds_value_over_self_certainty"] is True
    assert artifact["self_certainty_implementation"]["proxy_disclosure_required"] is True

    broken_bool = dict(artifact, ensemble_adds_value_over_self_certainty={"value": True})
    with pytest.raises(ValueError, match="ensemble_adds_value_over_self_certainty"):
        exp.validate_artifact(broken_bool)

    missing = dict(artifact)
    missing.pop("self_certainty_auroc_per_domain")
    with pytest.raises(ValueError, match="missing required"):
        exp.validate_artifact(missing)

    bad_verdict = dict(artifact, honest_verdict="complete: unexpected")
    with pytest.raises(ValueError, match="terminal verdict"):
        exp.validate_artifact(bad_verdict)

    bad_duration = dict(artifact, duration_s=-1.0)
    with pytest.raises(ValueError, match="duration_s"):
        exp.validate_artifact(bad_duration)

    blocked_output = exp.write_artifact(
        tmp_path,
        output_path="results/blocked-exp3684.json",
        tests_run=["REQ-SPOE-3684 blocked write_artifact"],
    )
    blocked = json.loads(blocked_output.read_text(encoding="utf-8"))
    assert blocked["honest_verdict"] == "complete: blocked_no_labeled_corpus_for_rebaseline"
    assert blocked["tests_run"] == ["REQ-SPOE-3684 blocked write_artifact"]


def test_req_spoe_3684_cached_preconditions_block_missing_corpora(tmp_path: Path) -> None:
    """REQ-SPOE-3684: missing cached corpora block without fabricating metrics."""

    artifact = exp.build_artifact(tmp_path, started_s=0.0, now_s=1.0)

    assert artifact["honest_verdict"] == "complete: blocked_no_labeled_corpus_for_rebaseline"
    assert artifact["ensemble_adds_value_over_self_certainty"] is False
    assert artifact["self_certainty_auroc_per_domain"] == {}
    assert artifact["corpus_status"]["math"]["status"] == "missing"


def test_req_spoe_3684_one_class_domains_are_skipped() -> None:
    """REQ-SPOE-3684: one-class held-out domains do not create false wins."""

    examples = _domain_examples("math", "ensemble_adds_value_over_self_certainty")
    examples.extend(
        exp.RebaselineExample(
            domain="one_class",
            label=1,
            ensemble_energy=0.7,
            confidence_error=0.5,
            self_certainty_error=0.5,
            example_id=f"one-{idx}",
        )
        for idx in range(8)
    )

    artifact = exp.build_artifact_from_examples(
        examples,
        seeds=[7],
        n_bootstrap=8,
    )

    assert "one_class" not in artifact["self_certainty_auroc_per_domain"]
    assert artifact["n_examples_per_domain"]["one_class"] == 8


def test_req_spoe_3684_loader_uses_disclosed_proxy(tmp_path: Path) -> None:
    """REQ-SPOE-3684: cached rows become rebaseline rows with proxy disclosure."""

    data = tmp_path / "data"
    data.mkdir()
    (data / "fover_corpus_v4.json").write_text(
        json.dumps(
            [
                {"question_id": "m1", "step_text": "ok", "label": "correct", "confidence": 0.9},
                {"question_id": "m2", "step_text": "bad", "label": "incorrect", "confidence": 0.2},
            ]
        ),
        encoding="utf-8",
    )

    examples, status = exp.load_rebaseline_examples(
        tmp_path,
        score_overrides={
            "math": {"ensemble_scores": [0.1, 0.9], "confidence_scores": [0.1, 0.8]},
        },
    )

    assert status["math"]["status"] == "loaded"
    assert status["code"]["status"] == "missing"
    assert [example.self_certainty_error for example in examples] == pytest.approx(
        exp.self_certainty_error_proxy_from_confidence_errors([0.1, 0.8])
    )


def test_req_spoe_3684_detector_loader_guards(monkeypatch: pytest.MonkeyPatch) -> None:
    """REQ-SPOE-3684: sibling detector loading has cache and failure guards."""

    assert exp._load_second_pair_detector() is exp.spd
    saved_module = exp.sys.modules.pop(exp._SPD_MODULE_NAME, None)
    monkeypatch.setattr(exp.importlib.util, "spec_from_file_location", lambda *args: None)
    try:
        with pytest.raises(ImportError, match="second_pair_detector"):
            exp._load_second_pair_detector()
    finally:
        if saved_module is not None:
            exp.sys.modules[exp._SPD_MODULE_NAME] = saved_module
