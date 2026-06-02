"""Tests for the Phase-1 shipped second-pair detector surface.

Spec: REQ-SPOE-3671, REQ-SPOE-3671-ARTIFACT,
      SCENARIO-SPOE-3671, SCENARIO-SPOE-3672, SCENARIO-SPOE-3673.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from carnot.pipeline.second_pair_detector import (
    CandidateScoreInput,
    LabeledDetectorExample,
    build_ship_artifact,
    build_ship_artifact_from_examples,
    load_cached_labeled_examples,
    score_candidates,
    validate_ship_artifact,
    write_ship_artifact,
)


def _domain_examples(domain: str, outcome: str, *, n: int = 80) -> list[LabeledDetectorExample]:
    examples: list[LabeledDetectorExample] = []
    for idx in range(n):
        label = 1 if idx < n // 2 else 0
        if outcome == "fusion_wins":
            ensemble = 0.92 - 0.003 * idx if label else 0.08 + 0.001 * (idx - n // 2)
            confidence = 0.50
        elif outcome == "confidence_wins":
            confidence = 0.92 - 0.003 * idx if label else 0.08 + 0.001 * (idx - n // 2)
            ensemble = 1.0 - confidence
        elif outcome == "weak_headroom":
            ensemble = 0.42 + (0.01 if label else -0.01)
            confidence = 0.50
        else:  # pragma: no cover - guarded by parametrization choices.
            raise ValueError(outcome)
        examples.append(
            LabeledDetectorExample(
                domain=domain,
                label=label,
                ensemble_energy=ensemble,
                confidence_error=confidence,
                example_id=f"{domain}-{outcome}-{idx}",
            )
        )
    return examples


@pytest.mark.parametrize(
    ("case_name", "examples", "e2e_passed", "expected_verdict", "expected_shipped"),
    [
        (
            "ships_math_and_code",
            _domain_examples("math", "fusion_wins") + _domain_examples("code", "fusion_wins"),
            True,
            "complete: second_pair_of_eyes_detector_shipped_math_strong_code_honest_e2e_green",
            True,
        ),
        (
            "ships_math_only_code_weak",
            _domain_examples("math", "fusion_wins") + _domain_examples("code", "confidence_wins"),
            True,
            "complete: second_pair_of_eyes_detector_shipped_math_only_code_weak_documented_e2e_green",
            True,
        ),
        (
            "blocked",
            [],
            False,
            "complete: blocked_no_labeled_corpus_for_detector",
            False,
        ),
        (
            "blocked_no_headroom",
            _domain_examples("math", "confidence_wins")
            + _domain_examples("code", "confidence_wins"),
            True,
            "complete: blocked_no_labeled_corpus_for_detector",
            False,
        ),
    ],
)
def test_scenario_spoe_3671_parametrized_honest_ship_outcomes(
    case_name: str,
    examples: list[LabeledDetectorExample],
    e2e_passed: bool,
    expected_verdict: str,
    expected_shipped: bool,
) -> None:
    """SCENARIO-SPOE-3671: ship verdicts cover both ship states and blocked."""

    artifact = build_ship_artifact_from_examples(
        examples,
        started_s=1.0,
        now_s=2.5,
        e2e_override=e2e_passed,
        tests_run=[f"SCENARIO-SPOE-3671 {case_name}"],
    )

    validate_ship_artifact(artifact)
    assert artifact["honest_verdict"] == expected_verdict
    assert artifact["detector_shipped"] is expected_shipped
    assert type(artifact["detector_shipped"]) is bool
    assert artifact["duration_s"] == pytest.approx(1.5)
    assert artifact["wired_surface"] == "score_candidates MCP tool and carnot score-candidates CLI"

    if expected_shipped:
        assert artifact["e2e_test_passed"] is True
        assert "math" in artifact["fused_detector_auroc_per_domain"]
        assert "math" in artifact["calibration_brier_ece_per_domain"]
        assert "0.10" in artifact["recall_at_fixed_fpr_table"]["math"]
    else:
        if not examples:
            assert artifact["fused_detector_auroc_per_domain"] == {}


def test_scenario_spoe_3672_score_candidates_returns_calibrated_operating_point() -> None:
    """SCENARIO-SPOE-3672: score_candidates is the shipped calibrated API."""

    examples = _domain_examples("math", "fusion_wins") + _domain_examples("code", "confidence_wins")
    response = score_candidates(
        [
            CandidateScoreInput(
                candidate_id="heldout-math",
                domain="math",
                text="1 + 1 = 3",
                confidence=0.35,
                ensemble_energy=0.88,
            )
        ],
        examples=examples,
    )

    assert response["surface"] == "score_candidates"
    assert response["detector_module_path"] == "python/carnot/pipeline/second_pair_detector.py"
    assert response["calibration_source"]["n_examples_per_domain"] == {"code": 80, "math": 80}
    assert len(response["scores"]) == 1
    score = response["scores"][0]
    assert score["candidate_id"] == "heldout-math"
    assert 0.0 <= score["calibrated_error_score"] <= 1.0
    assert score["confidence_error"] == pytest.approx(0.65)
    assert score["ensemble_energy"] == pytest.approx(0.88)
    assert score["operating_point"]["fpr_budget"] == 0.10


def test_scenario_spoe_3672_score_candidates_mapping_and_computed_energy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SCENARIO-SPOE-3672/3696: mapping inputs and code-native fallbacks are scored."""

    from carnot.pipeline import second_pair_detector as spd

    class FakeVerifier:
        def score_rows(self, rows):
            return [type("Score", (), {"score": 0.77})() for _ in rows]

    monkeypatch.setattr(spd, "_score_math_rows", lambda rows: [0.33 for _ in rows])
    monkeypatch.setattr(spd.code_native_verifier_3695, "CodeNativeVerifier", FakeVerifier)

    response = score_candidates(
        [
            {
                "id": "mapping-math",
                "answer": "2 + 2 = 5",
                "confidence": 0.25,
            },
            CandidateScoreInput(
                candidate_id="default-domain",
                domain="",
                text="2 + 2 = 4",
                confidence_error=0.4,
                ensemble_energy=0.2,
            ),
            CandidateScoreInput(
                candidate_id="code-computed",
                domain="code",
                text="def broken(:",
            ),
        ],
        examples=_domain_examples("math", "fusion_wins") + _domain_examples("code", "fusion_wins"),
        default_domain="math",
    )

    scores = {row["candidate_id"]: row for row in response["scores"]}
    assert scores["mapping-math"]["ensemble_energy"] == pytest.approx(0.33)
    assert scores["mapping-math"]["confidence_error"] == pytest.approx(0.75)
    assert scores["default-domain"]["domain"] == "math"
    assert scores["code-computed"]["ensemble_energy"] == pytest.approx(0.77)
    assert scores["code-computed"]["confidence_error"] == pytest.approx(0.5)


def test_req_spoe_3671_surface_blocks_without_labeled_corpus(tmp_path: Path) -> None:
    """REQ-SPOE-3671: score surface and artifact block instead of fabricating rows."""

    artifact = build_ship_artifact(tmp_path, started_s=0.0, now_s=1.0)
    assert artifact["honest_verdict"] == "complete: blocked_no_labeled_corpus_for_detector"
    assert artifact["corpus_status"]["math"]["status"] == "missing"

    with pytest.raises(ValueError, match="labeled corpus"):
        score_candidates(
            [{"id": "x", "text": "no corpus", "ensemble_energy": 0.1}],
            root=tmp_path,
        )


def test_req_spoe_3671_automatic_e2e_smoke_and_skip_one_class_domain() -> None:
    """REQ-SPOE-3671: artifact calls the shipped surface when no override is set."""

    examples = _domain_examples("math", "fusion_wins") + _domain_examples("code", "fusion_wins")
    examples.extend(
        LabeledDetectorExample("one_class", 1, 0.8 + idx * 0.01, 0.5, f"one-{idx}")
        for idx in range(8)
    )

    artifact = build_ship_artifact_from_examples(examples, started_s=0.0, now_s=1.0)

    assert artifact["e2e_test_passed"] is True
    assert artifact["detector_shipped"] is True
    assert "one_class" not in artifact["fused_detector_auroc_per_domain"]


def test_req_spoe_3671_e2e_smoke_failure_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    """REQ-SPOE-3671: E2E smoke returns false on exceptions or no evaluable row."""

    from carnot.pipeline import second_pair_detector as spd

    examples = _domain_examples("math", "fusion_wins")
    _, holdout = spd.stratified_train_holdout(examples, seed=3671)

    def boom(*args: object, **kwargs: object) -> dict[str, object]:
        raise RuntimeError("fixture failure")

    monkeypatch.setattr(spd, "score_candidates", boom)
    assert spd._run_surface_e2e_smoke(examples, holdout, root=Path(".")) is False

    one_class = [LabeledDetectorExample("one", 1, 0.5, 0.5, "one")]
    assert spd._run_surface_e2e_smoke(one_class, one_class, root=Path(".")) is False


def test_scenario_spoe_3673_loader_prefers_balanced_exp3658_code_corpus(tmp_path: Path) -> None:
    """SCENARIO-SPOE-3673: Exp 3671 code rows come from the balanced corpus."""

    data = tmp_path / "data"
    results = tmp_path / "results"
    data.mkdir()
    results.mkdir()
    (data / "fover_corpus_v4.json").write_text(
        json.dumps(
            [
                {"question_id": "m1", "step_text": "ok", "label": "correct", "confidence": 0.9},
                {"question_id": "m2", "step_text": "bad", "label": "incorrect", "confidence": 0.1},
            ]
        ),
        encoding="utf-8",
    )
    (data / "code_verification_corpus_v1.jsonl").write_text(
        json.dumps({"candidate_code": "def old():\n    return 1", "label": True}) + "\n",
        encoding="utf-8",
    )
    (data / "code_verification_corpus_v2.jsonl").write_text(
        "\n".join(
            [
                json.dumps({"candidate_code": "def ok():\n    return 1", "label": True}),
                json.dumps({"candidate_code": "def bad(:", "label": False}),
            ]
        ),
        encoding="utf-8",
    )
    (results / "experiment_3658_code_generalization_second_corpus.json").write_text(
        json.dumps({"second_code_corpus_path": "data/code_verification_corpus_v2.jsonl"}),
        encoding="utf-8",
    )

    examples, status = load_cached_labeled_examples(
        tmp_path,
        use_balanced_code_corpus=True,
        score_overrides={
            "math": {"ensemble_scores": [0.1, 0.9], "confidence_scores": [0.1, 0.9]},
            "code": {"ensemble_scores": [0.2, 0.8], "confidence_scores": [0.1, 0.7]},
        },
    )

    assert status["code"]["path"].endswith("data/code_verification_corpus_v2.jsonl")
    assert status["code"]["balanced_exp3658"] is True
    assert [example.example_id for example in examples if example.domain == "code"] == [
        "code-0",
        "code-1",
    ]


def test_req_spoe_3671_write_ship_artifact_and_validation(tmp_path: Path) -> None:
    """REQ-SPOE-3671-ARTIFACT: ship artifact is persisted with bare bool guard."""

    output = write_ship_artifact(
        tmp_path,
        output_path="results/exp3671.json",
        examples=_domain_examples("math", "fusion_wins") + _domain_examples("code", "confidence_wins"),
        started_s=0.0,
        now_s=1.0,
        e2e_override=True,
        tests_run=["REQ-SPOE-3671 write_ship_artifact"],
    )
    artifact = json.loads(output.read_text(encoding="utf-8"))

    validate_ship_artifact(artifact)
    assert artifact["detector_shipped"] is True
    assert artifact["ensemble_alone_auroc_per_domain"]["math"] >= 0.9
    assert artifact["acceptance_gate"]["passed"] is True

    broken = dict(artifact)
    broken["detector_shipped"] = {"value": True}
    with pytest.raises(ValueError, match="detector_shipped"):
        validate_ship_artifact(broken)

    missing = dict(artifact)
    missing.pop("wired_surface")
    with pytest.raises(ValueError, match="missing required"):
        validate_ship_artifact(missing)

    bad_verdict = dict(artifact, honest_verdict="complete: unexpected")
    with pytest.raises(ValueError, match="terminal verdict"):
        validate_ship_artifact(bad_verdict)

    bad_e2e = dict(artifact, e2e_test_passed={"value": True})
    with pytest.raises(ValueError, match="e2e_test_passed"):
        validate_ship_artifact(bad_e2e)

    bad_duration = dict(artifact, duration_s=-1.0)
    with pytest.raises(ValueError, match="duration_s"):
        validate_ship_artifact(bad_duration)
