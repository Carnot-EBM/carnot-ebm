"""Tests for Exp 3695 code-native second-pair verifier.

Spec: REQ-SPOE-3695, SCENARIO-SPOE-3695.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from carnot.pipeline import code_native_verifier_3695 as exp


def _metric(point: float, low: float, high: float) -> dict[str, object]:
    return {
        "point": point,
        "ci95": [low, high],
        "n": 20,
        "n_positive_errors": 10,
        "n_negative_correct": 10,
        "bootstrap_seeds": [3695],
        "seed_mean_aurocs": [point],
    }


def _baseline() -> dict[str, object]:
    return {
        "fused": _metric(0.5, 0.36, 0.64),
        "ensemble": _metric(0.5, 0.36, 0.64),
        "confidence": _metric(0.5, 0.5, 0.5),
        "calibration_brier_ece": {"brier": 0.3, "ece": 0.2},
        "n_holdout": 20,
    }


@pytest.mark.parametrize(
    (
        "case_name",
        "blocked",
        "native_metric",
        "calibration_after",
        "expected_verdict",
        "expected_recovered",
    ),
    [
        (
            "code_signal_recovered",
            False,
            _metric(0.74, 0.61, 0.86),
            {"brier": 0.18, "ece": 0.08},
            "complete: code_native_signal_recovered_beats_chance_floor",
            True,
        ),
        (
            "code_remains_math_only",
            False,
            _metric(0.56, 0.44, 0.68),
            {"brier": 0.18, "ece": 0.08},
            "complete: code_remains_math_only_code_native_signal_also_fails_earned",
            False,
        ),
        (
            "blocked",
            True,
            {},
            {},
            "complete: blocked_no_code_corpus_or_ast_tooling",
            False,
        ),
    ],
)
def test_scenario_spoe_3695_parametrized_honest_outcomes(
    case_name: str,
    blocked: bool,
    native_metric: dict[str, object],
    calibration_after: dict[str, float],
    expected_verdict: str,
    expected_recovered: bool,
) -> None:
    """SCENARIO-SPOE-3695: fixtures cover recovered, math-only, and blocked."""

    artifact = exp.build_artifact_from_metrics(
        blocked=blocked,
        code_auroc_baseline={} if blocked else _baseline(),
        code_native_metric=native_metric,
        code_native_calibration_brier_ece=calibration_after,
        code_native_recall_at_fixed_fpr={} if blocked else {"0.10": {"code_native_recall": 0.5}},
        n_examples_code=0 if blocked else 60,
        adversarial_verify_clean=not blocked,
        started_s=1.0,
        now_s=2.5,
        tests_run=[f"SCENARIO-SPOE-3695 {case_name}"],
    )

    exp.validate_artifact(artifact)
    assert artifact["honest_verdict"] == expected_verdict
    assert artifact["code_signal_recovered"] is expected_recovered
    assert type(artifact["code_signal_recovered"]) is bool
    assert artifact["code_native_auroc"] == (None if blocked else native_metric["point"])
    assert artifact["code_native_auroc_ci"] == (None if blocked else native_metric["ci95"])
    assert artifact["duration_s"] == pytest.approx(1.5)
    assert artifact["tests_run"] == [f"SCENARIO-SPOE-3695 {case_name}"]


def test_req_spoe_3695_ast_and_runtime_features_are_code_native() -> None:
    """REQ-SPOE-3695: verifier parses AST and executes probes, not a constant."""

    verifier = exp.CodeNativeVerifier()
    correct = {
        "candidate_code": "def add_one(x: int) -> int:\n    return x + 1\n",
        "label": True,
        "metadata": {"entry_point": "add_one"},
    }
    returns_none = {
        "candidate_code": "def add_one(x: int) -> int:\n    return None\n",
        "label": False,
        "metadata": {"entry_point": "add_one"},
    }
    syntax_error = {
        "candidate_code": "def add_one(x: int) -> int\n    return x + 1\n",
        "label": False,
        "metadata": {"entry_point": "add_one"},
    }

    scored = verifier.score_rows([correct, returns_none, syntax_error])

    assert scored[0].score < scored[1].score
    assert scored[0].score < scored[2].score
    assert scored[0].features["ast_parseable"] == 1.0
    assert scored[1].features["runtime_return_none_rate"] == 1.0
    assert scored[1].features["runtime_type_mismatch_rate"] == 1.0
    assert scored[2].features["parse_error"] == 1.0
    assert scored[0].features["execution_attempted"] == 1.0
    assert len({row.score for row in scored}) > 1


def test_req_spoe_3695_metric_helpers_and_recall_table() -> None:
    """REQ-SPOE-3695: metrics and recall table are computed from scores."""

    labels = [1, 1, 0, 0]
    scores = [0.9, 0.8, 0.2, 0.1]

    metric = exp.auroc_metric(labels, scores, seeds=[7], n_bootstrap=4)
    calibration = exp.calibration_bundle(labels, scores)
    table = exp.recall_at_fixed_fpr_table(labels, scores, budgets=(0.0, 0.5))
    calibrated = exp.measure_code_native_calibration(
        labels=[1, 1, 1, 0, 0, 0, 1, 0],
        scores=[0.95, 0.9, 0.85, 0.05, 0.08, 0.12, 0.8, 0.1],
        seeds=[7],
        n_bootstrap=4,
    )
    blocked_calibration = exp.measure_code_native_calibration(
        labels=[1, 1],
        scores=[0.8, 0.7],
        seeds=[7],
        n_bootstrap=1,
    )

    assert metric["point"] == pytest.approx(1.0)
    assert metric["n_positive_errors"] == 2
    assert calibration["brier"] < 0.1
    assert table["0.00"]["code_native_recall"] == 1.0
    assert table["0.50"]["code_native_actual_fpr"] <= 0.5
    assert calibrated["code_native_calibration_brier_ece"]["brier"] >= 0.0
    assert set(calibrated["code_native_recall_at_fixed_fpr"]) == {"0.05", "0.10", "0.20"}
    assert calibrated["code_native_calibration_protocol"]["method"] == "logistic"
    assert blocked_calibration["code_native_calibration_protocol"]["blocked_reason"]
    assert exp.auroc_signal_excludes_chance(_metric(0.7, 0.51, 0.9)) is True
    assert exp.auroc_signal_excludes_chance(_metric(0.7, 0.5, 0.9)) is False
    assert exp.calibration_improved({"brier": 0.3, "ece": 0.2}, {"brier": 0.2, "ece": 0.1})
    assert not exp.calibration_improved({"brier": 0.3, "ece": 0.2}, {"brier": 0.31, "ece": 0.1})


def test_req_spoe_3695_validation_and_write_artifact(tmp_path: Path) -> None:
    """REQ-SPOE-3695: artifact contract keeps required fields and bare bools."""

    output = exp.write_artifact_from_metrics(
        tmp_path,
        output_path="results/exp3695.json",
        blocked=False,
        code_auroc_baseline=_baseline(),
        code_native_metric=_metric(0.74, 0.61, 0.86),
        code_native_calibration_brier_ece={"brier": 0.18, "ece": 0.08},
        code_native_recall_at_fixed_fpr={"0.10": {"code_native_recall": 0.5}},
        n_examples_code=60,
        adversarial_verify_clean=True,
        started_s=0.0,
        now_s=1.0,
    )
    artifact = json.loads(output.read_text(encoding="utf-8"))

    exp.validate_artifact(artifact)
    assert artifact["acceptance_gate"]["passed"] is True
    assert artifact["code_signal_recovered"] is True
    assert set(exp.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)

    broken = dict(artifact, code_signal_recovered={"value": True})
    with pytest.raises(ValueError, match="code_signal_recovered"):
        exp.validate_artifact(broken)

    broken = dict(artifact, adversarial_verify_clean=1)
    with pytest.raises(ValueError, match="adversarial_verify_clean"):
        exp.validate_artifact(broken)

    missing = dict(artifact)
    missing.pop("code_native_auroc")
    with pytest.raises(ValueError, match="missing required"):
        exp.validate_artifact(missing)

    bad_verdict = dict(artifact, honest_verdict="complete: unexpected")
    with pytest.raises(ValueError, match="terminal verdict"):
        exp.validate_artifact(bad_verdict)


def test_req_spoe_3695_blocked_precondition_for_missing_corpus(tmp_path: Path) -> None:
    """REQ-SPOE-3695: missing corpus or tooling yields the blocked verdict."""

    artifact = exp.build_artifact(tmp_path, started_s=0.0, now_s=1.0)

    assert artifact["honest_verdict"] == "complete: blocked_no_code_corpus_or_ast_tooling"
    assert artifact["code_signal_recovered"] is False
    assert artifact["n_examples_code"] == 0
    assert artifact["acceptance_gate"]["passed"] is False


def test_req_spoe_3695_adversarial_report_cleanliness() -> None:
    """REQ-SPOE-3695: adversarial clean requires no critical flags."""

    assert exp.adversarial_report_is_clean({"flags": []}) is True
    assert exp.adversarial_report_is_clean(
        {"flags": [{"kind": "METHODOLOGY", "severity": "warn"}]}
    ) is True
    assert exp.adversarial_report_is_clean(
        {"flags": [{"kind": "TAUTOLOGY", "severity": "critical"}]}
    ) is False
    assert exp.adversarial_report_is_clean({"flags": 3}) is False
    assert exp.compact_adversarial_report({"flags": [{"severity": "warn"}, "skip"]}) == {
        "flag_count": 1,
        "flags": [{"severity": "warn"}],
    }


def test_req_spoe_3695_defensive_branches_and_success_build(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-SPOE-3695: helper edges stay deterministic and fail closed."""

    verifier = exp.CodeNativeVerifier()
    empty = verifier.score_row({"candidate_code": "", "candidate_sha256": "empty"})
    assert empty.features["parse_error"] == 1.0

    fallback = verifier.score_row(
        {
            "candidate_code": "def actual(x: int) -> int:\n    return x\n",
            "metadata": {"entry_point": "missing"},
        }
    )
    assert fallback.features["missing_entry_point"] == 1.0
    assert fallback.detail["entry_point"] == "actual"

    no_function = verifier.score_row({"candidate_code": "x = 1\n", "metadata": {"entry_point": "f"}})
    assert no_function.features["missing_value_return"] == 1.0
    no_entry_metadata = verifier.score_row({"candidate_code": "def inferred() -> int:\n    return 1\n"})
    assert no_entry_metadata.detail["entry_point"] == "inferred"
    pass_only = verifier.score_row(
        {"candidate_code": "def f():\n    pass\n", "metadata": {"entry_point": "f"}}
    )
    assert pass_only.features["missing_value_return"] == 1.0

    branchy = verifier.score_rows(
        [
            {
                "candidate_code": "def f():\n    return 1\n    x = 2\n",
                "metadata": {"entry_point": "f"},
            },
            {
                "candidate_code": "def f(x: int) -> int:\n    return missing_name\n",
                "metadata": {"entry_point": "f"},
            },
            {
                "candidate_code": "def f(x: int) -> int:\n    return eval('x')\n",
                "metadata": {"entry_point": "f"},
            },
            {
                "candidate_code": "def f(x: int) -> int:\n    return 1 // x\n",
                "metadata": {"entry_point": "f"},
            },
            {
                "candidate_code": "def f() -> int:\n    return 1\n",
                "metadata": {"entry_point": "f"},
            },
        ]
    )
    assert branchy[0].features["dead_code_after_return"] == 1.0
    assert branchy[1].features["undefined_name_count_clamped"] > 0.0
    assert branchy[2].features["forbidden_call"] == 1.0
    assert branchy[3].features["runtime_exception_rate"] > 0.0
    assert branchy[4].features["execution_attempted"] == 1.0

    assert exp.auroc_metric([1, 1], [0.9, 0.8], seeds=[9]) == exp.empty_metric([9])
    no_bootstrap = exp.auroc_metric([1, 0], [0.9, 0.1], seeds=[9], n_bootstrap=0)
    assert no_bootstrap["ci95"] == [1.0, 1.0]
    assert exp.feature_summary([])["n_scored"] == 0
    assert exp.calibration_improved({"brier": 0.3}, {"brier": 0.2}) is False
    assert exp.auroc_signal_excludes_chance({}) is False

    assert exp._annotation_kind(None) == "int"
    assert exp._annotation_kind(exp.ast.parse("x: list[int]").body[0].annotation) == "list"  # type: ignore[attr-defined]
    assert exp._annotation_kind(exp.ast.parse("x: typing.Dict").body[0].annotation) == "dict"  # type: ignore[attr-defined]
    assert exp._annotation_kind(exp.ast.parse("x: 'tuple'").body[0].annotation) == "tuple"  # type: ignore[attr-defined]
    assert exp._annotation_kind(exp.ast.parse("1").body[0].value) == "int"  # type: ignore[attr-defined]
    assert exp._kind_from_text("Sequence") == "list"
    assert exp._kind_from_text("bool") == "bool"
    assert exp._kind_from_text("float") == "float"
    assert exp._kind_from_text("str") == "str"
    assert exp._kind_from_text("None") == "none"
    assert exp._values_for_kind("unknown") == [1, 0, -1]
    assert exp._value_matches_kind(None, "none") is True
    assert exp._value_matches_kind(object(), "unknown") is True
    assert exp._round(float("inf")) == float("inf")
    assert exp._metric_ci({"ci95": [0.1]}) is None
    assert exp._precondition_n_examples([]) == 0

    rows = [
        {
            "candidate_code": "def f(x: int) -> int:\n    return x\n",
            "label": True,
            "metadata": {"entry_point": "f"},
        },
        {
            "candidate_code": "def f(x: int) -> int:\n    return None\n",
            "label": False,
            "metadata": {"entry_point": "f"},
        },
    ]
    real_check_preconditions = exp.check_preconditions
    real_reconfirm_code_baseline = exp.reconfirm_code_baseline
    real_run_adversarial_verify_report = exp.run_adversarial_verify_report
    monkeypatch.setattr(
        exp,
        "check_preconditions",
        lambda root: [
            {"resource": "balanced_exp3658_code_corpus", "available": True, "n_examples": 2},
            {"resource": "code_extractor_ast_runtime_tooling", "available": True},
        ],
    )
    monkeypatch.setattr(exp, "load_balanced_code_rows", lambda root: (rows, {"status": "loaded"}))
    monkeypatch.setattr(exp, "reconfirm_code_baseline", lambda root, seeds, n_bootstrap: _baseline())
    built = exp.build_artifact(tmp_path, started_s=0.0, now_s=1.0, seeds=[1], n_bootstrap=1)
    assert built["n_examples_code"] == 2
    assert built["code_native_feature_summary"]["n_scored"] == 2

    monkeypatch.setattr(exp, "_probe_args", lambda function: [])
    skipped_runtime = exp.CodeNativeVerifier().score_row(
        {"candidate_code": "def f() -> int:\n    return 1\n", "metadata": {"entry_point": "f"}}
    )
    assert skipped_runtime.features["execution_attempted"] == 0.0

    def boom(root: Path):
        raise RuntimeError("fixture")

    monkeypatch.setattr(exp, "load_balanced_code_rows", boom)
    monkeypatch.setattr(exp, "check_preconditions", real_check_preconditions)
    checks = exp.check_preconditions(tmp_path)
    assert checks[0]["available"] is False

    class RaisingExtractor:
        def extract(self, text: str, domain: str | None = None) -> list[object]:
            raise RuntimeError("fixture")

    monkeypatch.setattr(exp, "CodeExtractor", RaisingExtractor)
    assert exp._code_tooling_precondition()["available"] is False

    monkeypatch.setattr(
        exp.exp3683.spd,
        "load_cached_labeled_examples",
        lambda root, use_balanced_code_corpus: (["example"], {"status": "loaded"}),
    )
    monkeypatch.setattr(
        exp.exp3683,
        "measure_baseline_code_operating_point",
        lambda examples, seeds, n_bootstrap: {"baseline": examples, "seeds": list(seeds)},
    )
    monkeypatch.setattr(exp, "reconfirm_code_baseline", real_reconfirm_code_baseline)
    assert exp.reconfirm_code_baseline(tmp_path, seeds=[3], n_bootstrap=1) == {
        "baseline": ["example"],
        "seeds": [3],
    }

    artifact = exp.build_artifact_from_metrics(
        blocked=False,
        code_auroc_baseline=_baseline(),
        code_native_metric=_metric(0.74, 0.61, 0.86),
        code_native_calibration_brier_ece={"brier": 0.18, "ece": 0.08},
        code_native_recall_at_fixed_fpr={"0.10": {"code_native_recall": 0.5}},
        n_examples_code=60,
        adversarial_verify_clean=True,
        started_s=0.0,
        now_s=1.0,
    )
    monkeypatch.setattr(exp, "build_artifact", lambda root, tests_run=None: dict(artifact))
    monkeypatch.setattr(exp, "run_adversarial_verify_report", lambda path: {"flags": []})
    output = exp.write_artifact(tmp_path, output_path="results/write-exp3695.json")
    assert json.loads(output.read_text(encoding="utf-8"))["adversarial_verify_clean"] is True

    monkeypatch.setattr(exp, "run_adversarial_verify_report", real_run_adversarial_verify_report)
    raw_report = exp.run_adversarial_verify_report(output)
    assert "flags" in raw_report
    monkeypatch.setattr(exp.importlib.util, "spec_from_file_location", lambda *args: None)
    with pytest.raises(ImportError, match="adversarial_verify"):
        exp.run_adversarial_verify_report(output)

    bad_duration = dict(artifact, duration_s=-1.0)
    with pytest.raises(ValueError, match="duration_s"):
        exp.validate_artifact(bad_duration)
