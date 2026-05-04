"""Tests for Exp 1272 PRIME verifier-selection audit.

Spec: REQ-VERIFY-1272, SCENARIO-VERIFY-1272
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest

from carnot.eval import prime_verifier_selection_audit as exp


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def _fover_payload() -> dict:
    return {
        "metadata": {"schema": "carnot.fover_corpus.test"},
        "pairs": [
            {
                "question": "What is 2+2?",
                "response": "2+2=5, so the answer is 5.",
                "is_correct": False,
                "fover_labels": ["incorrect"],
                "cot_steps": [{"z3_label": "incorrect"}],
            },
            {
                "question": "What is 3+3?",
                "response": "3+3=6, so the answer is 6.",
                "is_correct": True,
                "fover_labels": ["correct"],
                "cot_steps": [{"z3_label": "correct"}],
            },
            {
                "question": "What is 4+4?",
                "response": "4+4=8, but final answer: 9.",
                "is_correct": False,
                "fover_labels": ["correct"],
                "cot_steps": [{"z3_label": "correct"}],
            },
            {
                "question": "What is 5+5?",
                "response": "5+5=11, but final answer: 10.",
                "is_correct": True,
                "fover_labels": ["incorrect"],
                "cot_steps": [{"z3_label": "incorrect"}],
            },
        ],
    }


def _exp1256_payload() -> dict:
    return {
        "experiment": "1256_verifier_orthogonality_audit_v3",
        "status": "complete",
        "verifier_names_k5": [
            "SOSKANEnergyV3",
            "SemEnergyProbe",
            "Z3MathVerifier",
        ],
        "pairwise_r_matrix": {
            "Z3MathVerifier|CausalReasoningVerifier": 0.1,
            "CausalReasoningVerifier|Z3MathVerifier": 0.1,
            "Z3MathVerifier|SymCodeVerifier": 0.2,
            "SymCodeVerifier|Z3MathVerifier": 0.2,
            "SemEnergyProbe|Z3MathVerifier": 0.0,
            "Z3MathVerifier|SemEnergyProbe": 0.0,
        },
        "max_pairwise_r_k5": 0.25,
        "k_eff": 3.5,
        "orthogonality_matrix_computed": True,
    }


def test_load_fover_rows_preserves_req1272_labels(tmp_path: Path) -> None:
    """REQ-VERIFY-1272-2: FoVer outcome and process labels are loaded."""

    fover_path = tmp_path / "fover_corpus_v5.json"
    _write_json(fover_path, _fover_payload())

    rows = exp.load_fover_rows(fover_path)

    assert len(rows) == 4
    assert rows[0].question == "What is 2+2?"
    assert rows[0].outcome_error is True
    assert rows[0].process_error is True
    assert rows[1].outcome_error is False
    assert rows[1].process_error is False


def test_prime_metric_math_and_weight_normalization_for_req1272() -> None:
    """REQ-VERIFY-1272-3/4: metrics feed a normalized GRPO weight vector."""

    rows = exp.rows_from_payload(_fover_payload())
    signals = {
        "Z3MathVerifier": [1.0, 0.0, 1.0, 1.0],
        "CausalReasoningVerifier": [1.0, 0.0, 0.0, 1.0],
        "SymCodeVerifier": [0.0, 0.0, 1.0, 0.0],
        "SemEnergyProbe": [0.0, 0.0, 0.0, 0.0],
        "k5_ensemble_summary": [1.0, 0.0, 1.0, 1.0],
    }

    artifact = exp.build_audit_artifact(
        rows,
        verifier_signals=signals,
        exp1256_payload=_exp1256_payload(),
        exp1271_payload={"status": "blocked"},
        run_date="20260504",
    )
    weights = artifact["verifier_weight_vector"]
    metrics = artifact["per_verifier_metrics"]

    assert artifact["verifier_weight_vector_written"] is True
    assert artifact["status"] == "complete"
    assert sum(weights.values()) == pytest.approx(1.0)
    assert set(weights) == set(signals)
    assert weights["Z3MathVerifier"] > weights["CausalReasoningVerifier"]
    assert weights["CausalReasoningVerifier"] > weights["SemEnergyProbe"]
    assert metrics["Z3MathVerifier"]["process_error_detection_rate"] == pytest.approx(1.0)
    assert metrics["Z3MathVerifier"]["final_answer_agreement"] == pytest.approx(0.75)
    assert metrics["SymCodeVerifier"]["process_error_detection_rate"] == pytest.approx(0.0)
    assert "exp1271_certificate_outputs" in artifact["missing_optional_fields"]


def test_insufficient_data_path_records_missing_fields_for_req1272() -> None:
    """REQ-VERIFY-1272-5: no process labels means no weight vector is claimed."""

    rows = exp.rows_from_payload(
        {
            "pairs": [
                {
                    "question": "q",
                    "response": "answer",
                    "is_correct": True,
                    "fover_labels": ["correct"],
                }
            ]
        }
    )

    artifact = exp.build_audit_artifact(
        rows,
        verifier_signals={"Z3MathVerifier": [0.0]},
        exp1256_payload={},
        exp1271_payload=None,
        run_date="20260504",
    )

    assert artifact["verifier_weight_vector_written"] is False
    assert artifact["verifier_weight_vector"] == {}
    assert artifact["status"] == "blocked"
    assert "process_error_labels" in artifact["missing_fields"]
    assert "outcome_label_classes" in artifact["missing_fields"]
    assert "exp1256_pairwise_r_matrix" in artifact["missing_fields"]


def test_req1272_helper_edges_are_deterministic(tmp_path: Path) -> None:
    """REQ-VERIFY-1272: loaders, penalties, and empty-data branches are stable."""

    assert exp._read_json_if_exists(None) is None
    assert exp._read_json_if_exists(tmp_path / "missing.json") is None

    rows = exp.rows_from_payload(
        [
            {
                "question": "raw-list row",
                "response": "bad step",
                "is_correct": False,
                "cot_steps": [{"verdict": "unsat"}],
            }
        ]
    )
    assert rows[0].process_error is True

    assert exp._pearson_binary([True], [False]) == 0.0
    assert exp._pearson_binary([True, True], [True, False]) == 0.0
    assert exp._pearson_binary([True, False], [True, False]) == pytest.approx(1.0)
    assert exp.normalize_weight_vector({"Z3MathVerifier": {"raw_weight_score": 0.0}}) == {}

    fallback_metrics = exp.compute_prime_metrics(
        exp.rows_from_payload(_fover_payload()),
        {
            "VerifierA": [1.0, 0.0, 1.0, 0.0],
            "VerifierB": [0.0, 1.0, 0.0, 1.0],
        },
        exp1256_payload={"pairwise_r_matrix": {"Other|Name": 0.5}},
    )
    assert fallback_metrics["VerifierA"]["pairwise_correlation_penalty"] == pytest.approx(1.0)

    empty_artifact = exp.build_audit_artifact(
        [],
        verifier_signals={},
        exp1256_payload={},
        exp1271_payload={"certificates": [{"step_id": "s0"}]},
        run_date="20260504",
    )
    assert "fover_pairs" in empty_artifact["missing_fields"]
    assert "positive_weight_scores" in empty_artifact["missing_fields"]
    assert empty_artifact["missing_optional_fields"] == []

    complete_without_cert = exp.build_audit_artifact(
        exp.rows_from_payload(_fover_payload()),
        verifier_signals={"Z3MathVerifier": [1.0, 0.0, 1.0, 1.0]},
        exp1256_payload=_exp1256_payload(),
        exp1271_payload={"status": "complete"},
        run_date="20260504",
    )
    assert "exp1271_certificate_outputs" in complete_without_cert["missing_optional_fields"]


def _fake_module(name: str, **attrs: object) -> types.ModuleType:
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    return module


def test_default_verifier_signal_reconstruction_for_req1272(monkeypatch: pytest.MonkeyPatch) -> None:
    """REQ-VERIFY-1272-3: default reconstruction uses available verifier adapters."""

    class FakeZ3:
        def score(self, text: str) -> float:
            return 1.0 if "bad" in text else 0.0

    class FakeSymCode:
        def detection_score(self, response: str) -> float:
            return 1.0 if "sym" in response else 0.0

    class FakeCausal:
        def detection_score(self, response: str) -> float:
            return 1.0 if "causal" in response else 0.0

    class FakeSemEnergy:
        def score(self, text: str) -> float:
            return 1.0 if "sem" in text else 0.0

    class FakeSOSKAN:
        def score(self, text: str) -> float:
            return 1.0 if "sos" in text else 0.0

    monkeypatch.setitem(
        sys.modules,
        "carnot.verify.z3_math_verifier",
        _fake_module("carnot.verify.z3_math_verifier", Z3MathVerifier=FakeZ3),
    )
    monkeypatch.setitem(
        sys.modules,
        "carnot.pipeline.symcode_verifier",
        _fake_module("carnot.pipeline.symcode_verifier", SymCodeVerifier=FakeSymCode),
    )
    monkeypatch.setitem(
        sys.modules,
        "carnot.pipeline.causal_reasoning_verifier",
        _fake_module(
            "carnot.pipeline.causal_reasoning_verifier",
            CausalReasoningVerifier=FakeCausal,
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "carnot.verify.and_composition_verifier",
        _fake_module(
            "carnot.verify.and_composition_verifier",
            SemEnergyProbeAdapter=FakeSemEnergy,
            SOSKANEnergyV3Adapter=FakeSOSKAN,
        ),
    )
    rows = exp.rows_from_payload(
        {
            "pairs": [
                {"question": "bad sem", "response": "sym causal sos", "is_correct": False},
                {"question": "clean", "response": "clean", "is_correct": True},
            ]
        }
    )

    signals = exp.evaluate_default_verifier_signals(rows)

    assert signals["Z3MathVerifier"] == [1.0, 0.0]
    assert signals["SymCodeVerifier"] == [1.0, 0.0]
    assert signals["CausalReasoningVerifier"] == [1.0, 0.0]
    assert signals["SemEnergyProbe"] == [1.0, 0.0]
    assert signals["SOSKANEnergyV3"] == [1.0, 0.0]
    assert signals["k5_ensemble_summary"] == [1.0, 0.0]


def test_default_verifier_signal_reconstruction_skips_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-VERIFY-1272-3: unavailable verifier adapters are treated as absent."""

    class RaisingVerifier:
        def __init__(self) -> None:
            raise RuntimeError("unavailable")

    monkeypatch.setitem(
        sys.modules,
        "carnot.verify.z3_math_verifier",
        _fake_module("carnot.verify.z3_math_verifier", Z3MathVerifier=RaisingVerifier),
    )
    monkeypatch.setitem(
        sys.modules,
        "carnot.pipeline.symcode_verifier",
        _fake_module("carnot.pipeline.symcode_verifier", SymCodeVerifier=RaisingVerifier),
    )
    monkeypatch.setitem(
        sys.modules,
        "carnot.pipeline.causal_reasoning_verifier",
        _fake_module(
            "carnot.pipeline.causal_reasoning_verifier",
            CausalReasoningVerifier=RaisingVerifier,
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "carnot.verify.and_composition_verifier",
        _fake_module(
            "carnot.verify.and_composition_verifier",
            SemEnergyProbeAdapter=RaisingVerifier,
            SOSKANEnergyV3Adapter=RaisingVerifier,
        ),
    )

    assert exp.evaluate_default_verifier_signals(exp.rows_from_payload(_fover_payload())) == {}


def test_run_experiment_writes_required_scenario1272_artifact(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-1272: runner writes the required Exp 1272 schema."""

    fover_path = tmp_path / "fover_corpus_v5.json"
    exp1256_path = tmp_path / "experiment_1256_verifier_orthogonality_audit_v3.json"
    exp1271_path = tmp_path / "experiment_1271_triggered_certificate_extraction_sota_gguf.json"
    output_path = tmp_path / "experiment_1272_prime_verifier_selection_audit.json"
    _write_json(fover_path, _fover_payload())
    _write_json(exp1256_path, _exp1256_payload())
    _write_json(exp1271_path, {"status": "blocked", "honest_verdict": "blocked_gate_check_failed"})

    artifact = exp.run_experiment(
        fover_path=fover_path,
        exp1256_path=exp1256_path,
        exp1271_path=exp1271_path,
        output_path=output_path,
        verifier_signals={
            "Z3MathVerifier": [1.0, 0.0, 1.0, 1.0],
            "CausalReasoningVerifier": [1.0, 0.0, 0.0, 1.0],
            "SymCodeVerifier": [0.0, 0.0, 1.0, 0.0],
            "SemEnergyProbe": [0.0, 0.0, 0.0, 0.0],
            "k5_ensemble_summary": [1.0, 0.0, 1.0, 1.0],
        },
    )
    persisted = json.loads(output_path.read_text(encoding="utf-8"))

    assert artifact == persisted
    assert exp.REQUIRED_ARTIFACT_FIELDS <= set(persisted)
    assert persisted["experiment"] == "1272_prime_verifier_selection_audit"
    assert persisted["schema"] == "prime_verifier_selection_audit_v1"
    assert persisted["run_date"] == "20260504"
    assert persisted["verifier_weight_vector_written"] is True
    assert persisted["honest_verdict"].startswith("prime_verifier_weights_selected_")
