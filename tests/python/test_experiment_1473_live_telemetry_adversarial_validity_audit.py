"""Tests for Exp 1473 live telemetry adversarial validity audit.

Spec: REQ-VERIFY-1473, SCENARIO-VERIFY-1473.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from carnot.reporting.live_telemetry_adversarial_validity_audit import (
    REQUIRED_ARTIFACT_FIELDS,
    audit_beaver_lite_artifact,
    audit_telemetry_confounds,
    run_audit,
    write_in_progress_artifact,
)


def _row(
    *,
    case_id: str,
    expected_answer: str,
    response_text: str,
    token_logprobs: list[float],
    family: str = "fover_style",
    source: str = "live_sota_llamacpp",
) -> dict[str, Any]:
    return {
        "case_id": case_id,
        "family": family,
        "prompt": "Return 1 if true, 0 if false.",
        "expected_answer": expected_answer,
        "response_text": response_text,
        "completion_tokens": len(token_logprobs),
        "token_logprobs": token_logprobs,
        "top_logprobs": [{"1": -0.1, "0": -1.0} for _ in token_logprobs],
        "generation_source": source,
        "response_text_available": True,
        "token_logprobs_available": True,
        "topk_alternatives_available": True,
        "hf_id": "unsloth/Qwen3.6-35B-A3B-GGUF",
    }


def _beaver_artifact(*, mode: str = "live_exp1468") -> dict[str, Any]:
    return {
        "status": "complete",
        "benchmark_family": "BEAVER-style deterministic bounds",
        "questions_evaluated": ["q1", "q2", "q3"],
        "prefix_closed_constraint": [
            {
                "constraint_id": f"toy_{idx}",
                "description": "terminal response text must end with an integer in the inclusive range [0, 9999]",
                "terminal_only": True,
                "prefix_closed": True,
            }
            for idx in range(3)
        ],
        "unsafe_mass_bounds": [0.0, 0.0, 0.0],
        "empirical_violation_rates": [0.0, 0.0, 0.0],
        "bound_is_sound": True,
        "mock_or_live_logprobs": mode,
        "external_fit_verdict": "adopted_minimal_beaver_smoke_fit",
        "broad_benchmark_deferred": True,
        "honest_verdict": "sound_bound_live_exp1468"
        if mode == "live_exp1468"
        else "sound_bound_mock_logprobs",
        "n_completions": [1, 1, 1],
    }


def test_req_verify_1473_writes_in_progress_artifact(tmp_path: Path) -> None:
    """REQ-VERIFY-1473: seed the audit deliverable before source artifacts load."""
    output = tmp_path / "results" / "experiment_1473.json"

    artifact = write_in_progress_artifact(output)

    assert json.loads(output.read_text(encoding="utf-8")) == artifact
    assert set(REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert artifact["status"] == "in_progress"
    assert artifact["claim_allowed"] is False
    assert artifact["honest_verdict"] == "in_progress"


def test_req_verify_1473_length_baseline_blocks_telemetry_claim() -> None:
    """REQ-VERIFY-1473: length/token baselines can falsify a telemetry signal."""
    rows = [
        _row(case_id="false-short-1", expected_answer="1", response_text="1", token_logprobs=[-0.2]),
        _row(case_id="false-short-2", expected_answer="1", response_text="1", token_logprobs=[-0.3]),
        _row(
            case_id="true-long-1",
            expected_answer="0",
            response_text="long reasoning before 0",
            token_logprobs=[-0.1, -0.1, -0.1],
        ),
        _row(
            case_id="true-long-2",
            expected_answer="0",
            response_text="another long reasoning before 0",
            token_logprobs=[-0.1, -0.1, -0.1],
        ),
    ]
    exp1469 = {
        "status": "complete",
        "diagnostic_lineage_retired": False,
        "auroc_or_rank_signal": {
            "rank_signal": {
                "best_signal": {
                    "name": "token_logprob_mean",
                    "oriented_auroc": 0.75,
                    "label_key": "known_verifier_label",
                }
            }
        },
    }

    audit = audit_telemetry_confounds(rows, exp1469, run_date="20260507")

    assert audit["best_superficial_baseline"]["name"] in {
        "completion_tokens",
        "response_char_length",
        "token_count",
    }
    assert audit["superficial_baseline_matches_or_exceeds"] is True
    assert audit["length_confound"]["verdict"] == "fail"
    assert audit["claim_blockers"] == ["superficial_baseline_matches_or_exceeds"]


def test_req_verify_1473_source_retirement_blocks_even_when_baseline_is_weaker() -> None:
    """REQ-VERIFY-1473: an already retired diagnostic cannot become a headline."""
    rows = [
        _row(case_id="false-short-1", expected_answer="1", response_text="1", token_logprobs=[-0.1]),
        _row(case_id="true-short-1", expected_answer="0", response_text="0", token_logprobs=[-0.1]),
    ]
    exp1469 = {
        "status": "complete",
        "diagnostic_lineage_retired": True,
        "auroc_or_rank_signal": {
            "rank_signal": {
                "best_signal": {
                    "name": "topk_gap_mean",
                    "oriented_auroc": 1.0,
                    "label_key": "known_verifier_label",
                }
            }
        },
    }

    audit = audit_telemetry_confounds(rows, exp1469, run_date="20260507")

    assert audit["source_diagnostic_lineage_retired"] is True
    assert audit["superficial_baseline_matches_or_exceeds"] is False
    assert audit["claim_blockers"] == ["source_diagnostic_lineage_retired"]


def test_req_verify_1473_missing_reported_signal_fails_closed() -> None:
    """REQ-VERIFY-1473: absent binary rank evidence does not invent a signal."""
    rows = [
        _row(case_id="same-label-1", expected_answer="1", response_text="1", token_logprobs=[-0.1]),
        _row(case_id="same-label-2", expected_answer="1", response_text="1", token_logprobs=[-0.2]),
    ]

    audit = audit_telemetry_confounds(rows, {}, run_date="20260507")

    assert audit["reported_signal"] is None
    assert audit["reported_oriented_auroc"] == 0.0
    assert audit["best_superficial_baseline"] is None
    assert audit["length_confound"]["best_oriented_auroc"] == 0.0


def test_req_verify_1473_beaver_audit_flags_surface_constraint_gate() -> None:
    """REQ-VERIFY-1473: BEAVER-lite live/mock labeling is checked separately from signal validity."""
    live_audit = audit_beaver_lite_artifact(_beaver_artifact(mode="live_exp1468"))
    mock_audit = audit_beaver_lite_artifact(_beaver_artifact(mode="mock_logprobs"))

    assert live_audit["mock_logprobs_used"] is False
    assert live_audit["mock_live_label_clear"] is True
    assert live_audit["surface_constraint_only"] is True
    assert live_audit["single_logged_completion_gate"] is True
    assert live_audit["can_pass_without_real_verifier_signal"] is True
    assert live_audit["verdict"] == "fail_surface_constraint_only"
    assert mock_audit["mock_logprobs_used"] is True
    assert mock_audit["mock_live_label_clear"] is True


def test_scenario_verify_1473_run_writes_final_artifact_and_note(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-1473: the runner writes the required JSON and markdown note."""
    results = tmp_path / "results"
    docs = tmp_path / "docs" / "research-notes"
    exp1468_path = results / "experiment_1468.json"
    exp1469_path = results / "experiment_1469.json"
    exp1470_path = results / "experiment_1470.json"
    manifest_path = results / "manifest.jsonl"
    output_path = results / "experiment_1473.json"
    note_path = docs / "live_telemetry_adversarial_validity_audit.md"
    rows = [
        _row(case_id="false-short-1", expected_answer="1", response_text="1", token_logprobs=[-0.2]),
        _row(case_id="false-short-2", expected_answer="1", response_text="1", token_logprobs=[-0.3]),
        _row(
            case_id="true-long-1",
            expected_answer="0",
            response_text="long reasoning before 0",
            token_logprobs=[-0.1, -0.1, -0.1],
        ),
        _row(
            case_id="true-long-2",
            expected_answer="0",
            response_text="another long reasoning before 0",
            token_logprobs=[-0.1, -0.1, -0.1],
        ),
    ]
    results.mkdir(parents=True)
    exp1468_path.write_text(
        json.dumps(
            {
                "status": "complete",
                "honest_verdict": "live_sota_topk_telemetry_ready",
                "live_sota_model_inference_used": True,
                "topk_logprobs_available": True,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    exp1469_path.write_text(
        json.dumps(
            {
                "status": "complete",
                "diagnostic_lineage_retired": True,
                "auroc_or_rank_signal": {
                    "rank_signal": {
                        "best_signal": {
                            "name": "token_logprob_mean",
                            "oriented_auroc": 0.75,
                            "label_key": "known_verifier_label",
                        }
                    }
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    exp1470_path.write_text(json.dumps(_beaver_artifact()) + "\n", encoding="utf-8")
    manifest_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )

    artifact = run_audit(
        project_root=tmp_path,
        run_date="20260507",
        exp1468_artifact_path=exp1468_path,
        exp1469_artifact_path=exp1469_path,
        exp1470_artifact_path=exp1470_path,
        exp1468_manifest_path=manifest_path,
        output_path=output_path,
        audit_note_path=note_path,
    )

    assert json.loads(output_path.read_text(encoding="utf-8")) == artifact
    assert set(REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert artifact["status"] == "complete"
    assert artifact["length_confound_checked"] is True
    assert artifact["format_confound_checked"] is True
    assert artifact["prompt_family_confound_checked"] is True
    assert artifact["mock_logprob_leakage_checked"] is True
    assert artifact["claim_allowed"] is False
    assert artifact["honest_verdict"] == "telemetry_claim_blocked_adversarial_audit"
    assert note_path.read_text(encoding="utf-8").startswith("# Live Telemetry Adversarial Validity Audit")
