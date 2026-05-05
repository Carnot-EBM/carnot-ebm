"""Tests for Exp 1312 triggered certificate extraction comparison.

Spec: REQ-VERIFY-1312,
      SCENARIO-VERIFY-1312
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from carnot.eval.certificate_grammar_backend_bakeoff import sample_certificate
from carnot.reporting import triggered_certificate_extraction_dccd_gbnf as mod


QWEN_ID = "unsloth/Qwen3.6-35B-A3B-GGUF"
GEMMA_ID = "unsloth/gemma-4-31B-it-GGUF"


def _response(
    *,
    item_id: str,
    parsed_label: str,
    verifier_label: str,
    raw_output: str,
    compact_encoding: bool,
    hf_id: str = QWEN_ID,
) -> dict[str, Any]:
    return {
        "hf_id": hf_id,
        "model_name": "Qwen3.6-35B-A3B" if hf_id == QWEN_ID else "Gemma4-31B-it",
        "gpu": 0 if hf_id == QWEN_ID else 1,
        "item_id": item_id,
        "family": "satquest",
        "expected_label": verifier_label,
        "compact_encoding": compact_encoding,
        "perturbation_index": 0,
        "raw_output": raw_output,
        "parsed_label": parsed_label,
        "verifier_label": verifier_label,
        "verified": parsed_label == verifier_label,
        "generation_source": "live_sota_llamacpp",
    }


def _exp1311(
    rows: list[dict[str, Any]],
    *,
    score: float = 0.9,
    headline: bool = True,
) -> dict[str, Any]:
    return {
        "status": "complete",
        "run_date": "20260505",
        "answer_stability_score": score,
        "headline_result_allowed": headline,
        "models_used": [QWEN_ID, GEMMA_ID],
        "responses": rows,
    }


def test_exp1312_embedded_raw_certificate_uses_1283_schema() -> None:
    """REQ-VERIFY-1312-4: raw-trigger parsing reuses the bounded certificate schema."""
    certificate = sample_certificate() | {"final_answer": "SAT"}
    parsed = mod.parse_certificate_text(f"prefix <CARNOT_CERT>{json.dumps(certificate)}")
    invalid = mod.parse_certificate_text('{"claims": []}')
    malformed = mod.parse_certificate_text('{"claims": ')
    non_object = mod.parse_certificate_text("[1, 2, 3]")

    assert parsed.parseable is True
    assert parsed.certificate["final_answer"] == "SAT"
    assert parsed.errors == []
    assert invalid.parseable is False
    assert any(error.startswith("missing ") for error in invalid.errors)
    assert malformed.errors == ["invalid_json: Expecting value"]
    assert non_object.errors == ["certificate must be object"]


@pytest.mark.parametrize(
    ("artifact", "reason"),
    [
        (_exp1311([], score=0.59), "answer_stability_below_gate"),
        (_exp1311([], headline=False), "exp1311_not_headline"),
        (_exp1311([]), "no_headline_sota_outputs"),
    ],
)
def test_exp1312_blocks_on_exp1311_gate_failures(
    artifact: dict[str, Any],
    reason: str,
) -> None:
    """REQ-VERIFY-1312-2: failed Exp 1311 gates become terminal blockers."""
    result = mod.build_comparison_artifact(exp1311_artifact=artifact, run_date="20260505")

    assert result["status"] == "blocked"
    assert result["blocked_reason"] == reason
    assert result["headline_result_allowed"] is False
    assert result["honest_verdict"] == f"blocked_{reason}"
    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(result)


def test_exp1312_filters_non_headline_rows_before_comparison() -> None:
    """REQ-VERIFY-1312-2: only live mandated SOTA rows count as headline outputs."""
    rows: list[Any] = [
        "not a row",
        _response(
            item_id="sq_sat_xor",
            parsed_label="SAT",
            verifier_label="SAT",
            raw_output="SAT",
            compact_encoding=False,
        )
        | {"generation_source": "injected"},
        _response(
            item_id="sq_sat_xor",
            parsed_label="SAT",
            verifier_label="SAT",
            raw_output="SAT",
            compact_encoding=False,
            hf_id="legacy/small-model",
        ),
    ]

    artifact = mod.build_comparison_artifact(
        exp1311_artifact=_exp1311(rows),
        run_date="20260505",
    )

    assert artifact["status"] == "blocked"
    assert artifact["blocked_reason"] == "no_headline_sota_outputs"


def test_exp1312_compares_raw_gbnf_dccd_and_repair_paths() -> None:
    """REQ-VERIFY-1312-3/5: comparison metrics include DCCD delta and repair success."""
    raw_certificate = sample_certificate() | {"final_answer": "SAT"}
    rows = [
        _response(
            item_id="sq_sat_xor",
            parsed_label="SAT",
            verifier_label="SAT",
            raw_output=f"<CARNOT_CERT>{json.dumps(raw_certificate)}",
            compact_encoding=False,
        ),
        _response(
            item_id="sq_compact_unsat_dimacs",
            parsed_label="SAT",
            verifier_label="UNSAT",
            raw_output="UNSAT because the two units conflict, but no JSON certificate",
            compact_encoding=True,
            hf_id=GEMMA_ID,
        ),
    ]

    artifact = mod.build_comparison_artifact(
        exp1311_artifact=_exp1311(rows),
        run_date="20260505",
    )

    assert artifact["status"] == "complete"
    assert artifact["headline_result_allowed"] is True
    assert artifact["models_used"] == [QWEN_ID, GEMMA_ID]
    assert artifact["certificate_parse_rate"] == pytest.approx(6 / 7)
    assert artifact["certificate_truthfulness_rate"] == pytest.approx(5 / 6)
    assert artifact["dccd_delta_over_grammar_only"] == pytest.approx(0.5)
    assert artifact["repair_success_rate"] == pytest.approx(1.0)
    assert artifact["path_metrics"]["raw_trigger"]["attempts"] == 2
    assert artifact["path_metrics"]["gbnf_constrained"]["truthful_rate"] == pytest.approx(0.5)
    assert artifact["path_metrics"]["dccd_compact"]["truthful_rate"] == pytest.approx(1.0)
    assert artifact["grammar_projection_tax_proxy"]["proxy"] == "extra_prompt_chars"
    assert artifact["honest_verdict"] == "triggered_certificate_dccd_gbnf_comparison_complete"


def test_exp1312_run_experiment_writes_final_artifact(tmp_path: Path) -> None:
    """REQ-VERIFY-1312-1: run_experiment writes the requested deliverable JSON."""
    rows = [
        _response(
            item_id="cb_unknown_missing_bound",
            parsed_label="ABSTAIN",
            verifier_label="UNKNOWN",
            raw_output="",
            compact_encoding=False,
        )
    ]
    exp1311_path = tmp_path / "results" / "experiment_1311.json"
    output_path = tmp_path / "results" / "experiment_1312.json"
    exp1311_path.parent.mkdir(parents=True)
    exp1311_path.write_text(json.dumps(_exp1311(rows)), encoding="utf-8")

    artifact = mod.run_experiment(
        project_root=tmp_path,
        run_date="20260505",
        exp1311_path=exp1311_path,
        output_path=output_path,
    )

    assert json.loads(output_path.read_text(encoding="utf-8")) == artifact
    assert artifact["status"] == "complete"
    assert artifact["certificate_parse_rate"] == pytest.approx(2 / 3)
    assert mod._normalize_label("unknown") == "UNKNOWN"
    assert mod._normalize_label("unbounded") == "UNBOUNDED"
