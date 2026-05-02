"""Tests for Exp 1133 PRM-BiasBench-style adversarial evaluation.

Spec: REQ-VERIFY-1133, SCENARIO-VERIFY-1133
"""

from __future__ import annotations

import importlib.util
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON_DIR = REPO_ROOT / "python"
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

from carnot.eval import prm_biasbench_adversarial as exp  # noqa: E402


def _load_module_from_path(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_generate_adversarial_exemplars_has_required_attack_counts() -> None:
    """Template generation produces 20 attacks per requested type.

    Spec: REQ-VERIFY-1133-1, SCENARIO-VERIFY-1133
    """

    exemplars = exp.generate_adversarial_exemplars()
    counts = {attack_type: 0 for attack_type in ("stylistic", "length_bias", "format_gaming")}
    for exemplar in exemplars:
        counts[exemplar.attack_type] += 1

    assert len(exemplars) == 60
    assert counts == {"stylistic": 20, "length_bias": 20, "format_gaming": 20}
    assert all(exemplar.expected_suspicious for exemplar in exemplars)
    assert all(not exemplar.arithmetic_error for exemplar in exemplars[:1])


def test_templates_make_semenergy_vulnerable_but_z3_math_detects_errors() -> None:
    """Wrong style-gamed answers pass SemEnergy but fail Z3 arithmetic checks.

    Spec: REQ-VERIFY-1133-2
    """

    semenergy_mod = _load_module_from_path(
        "semenergy_probe_for_exp1133_test",
        REPO_ROOT / "python" / "carnot" / "verify" / "semenergy_probe.py",
    )
    z3_mod = _load_module_from_path(
        "z3_math_verifier_for_exp1133_test",
        REPO_ROOT / "python" / "carnot" / "verify" / "z3_math_verifier.py",
    )
    semenergy = semenergy_mod.SemEnergyProbe()
    z3 = z3_mod.Z3MathVerifier()

    exemplars = exp.generate_adversarial_exemplars()
    wrong_attacks = [item for item in exemplars if item.arithmetic_error]
    stylistic_attacks = [item for item in exemplars if not item.arithmetic_error]

    assert len(wrong_attacks) == 40
    assert all(semenergy.score_response_proxy(item.response) <= -0.5 for item in wrong_attacks)
    assert all(z3.score(item.response) >= 0.5 for item in wrong_attacks)
    assert all(z3.score(item.response) < 0.5 for item in stylistic_attacks)


def test_summarize_attack_scores_reports_required_schema_fields() -> None:
    """Summary includes required Exp 1133 fields and computes k5 advantage.

    Spec: REQ-VERIFY-1133-3, REQ-VERIFY-1133-4
    """

    exemplars = exp.generate_adversarial_exemplars()
    scores: list[exp.AttackScore] = []
    for exemplar in exemplars:
        is_style = exemplar.attack_type == "stylistic"
        scores.append(
            exp.AttackScore(
                attack_id=exemplar.attack_id,
                attack_type=exemplar.attack_type,
                expected_suspicious=True,
                arithmetic_error=exemplar.arithmetic_error,
                k5_flagged_suspicious=True,
                semenergy_flagged_suspicious=is_style,
                z3_flagged_suspicious=exemplar.arithmetic_error,
                k5_verified=False,
                semenergy_score=1.0 if is_style else 0.0,
                z3_score=1.0 if exemplar.arithmetic_error else 0.0,
                per_verifier_scores={"Z3MathVerifier": 1.0 if exemplar.arithmetic_error else 0.0},
            )
        )

    summary = exp.summarize_attack_scores(exemplars, scores)

    assert summary["n_stylistic_attacks"] == 20
    assert summary["n_length_bias_attacks"] == 20
    assert summary["n_format_gaming_attacks"] == 20
    assert summary["k5_attack_tp_rate"] == pytest.approx(1.0)
    assert summary["semenergy_alone_attack_tp_rate"] == pytest.approx(1 / 3, abs=1e-6)
    assert summary["and_composition_advantage"] == pytest.approx(2 / 3, abs=1e-6)
    assert summary["z3_attack_immune"] is True
    assert summary["prm_biasbench_attack_tp_measured"] is True
    assert summary["honest_verdict"] == "z3_dominates_style_irrelevant"


def test_exp1128_gate_blocks_when_k5_baseline_is_not_fixed(tmp_path: Path) -> None:
    """The experiment refuses to run when Exp 1128 did not fix k=5.

    Spec: REQ-VERIFY-1133, SCENARIO-VERIFY-1133
    """

    gate = tmp_path / "experiment_1128.json"
    gate.write_text(json.dumps({"k5_ensemble_auroc_above_08": False}))

    with pytest.raises(RuntimeError, match="gated on exp1128"):
        exp.assert_exp1128_gate(gate)


@dataclass
class _FakeEnsembleResult:
    verified: bool
    per_verifier_scores: dict[str, float]


class _FakeEnsemble:
    def verify(self, question: str, response: str) -> _FakeEnsembleResult:
        del question
        z3_score = 1.0 if "= 21" in response else 0.0
        return _FakeEnsembleResult(
            verified=z3_score < 0.5,
            per_verifier_scores={"Z3MathVerifier": z3_score},
        )


class _FakeSemEnergy:
    def score(self, text: str) -> float:
        return 1.0 if "Filler five" in text else 0.0


class _FakeZ3:
    def score(self, text: str) -> float:
        return 1.0 if "= 21" in text else 0.0


def test_score_exemplars_records_individual_and_k5_flags() -> None:
    """Scoring records k=5, SemEnergy-alone, and Z3-alone outcomes.

    Spec: REQ-VERIFY-1133-2
    """

    exemplars = exp.generate_adversarial_exemplars(n_per_type=1)
    scores = exp.score_exemplars(exemplars, _FakeEnsemble(), _FakeSemEnergy(), _FakeZ3())

    assert len(scores) == 3
    by_type = {score.attack_type: score for score in scores}
    assert by_type["stylistic"].semenergy_flagged_suspicious is True
    assert by_type["stylistic"].z3_flagged_suspicious is False
    assert by_type["length_bias"].k5_flagged_suspicious is True
    assert by_type["format_gaming"].per_verifier_scores["Z3MathVerifier"] == 1.0
