"""Tests for Exp 2843 BEAVER/EPR bounded-prefix probe.

Spec: REQ-VERIFY-2843, SCENARIO-VERIFY-2843.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pytest

from carnot.verify import beaver_epr_bounded_probe as exp


def _write_json(path: Path, payload: dict[str, Any] | list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _exp2836_payload(*, ready: bool = True) -> dict[str, Any]:
    return {
        "sota_runtime_ready": ready,
        "selected_python": "/repo/.venv/bin/python",
        "loader_probe": {
            "llama_cpp_import_ok": True,
            "llama_cpp_origin": "/repo/.venv/lib/python3.14/site-packages/llama_cpp/__init__.py",
        },
        "model_specs": {
            "primary": list(exp.HEADLINE_REQUIRED_ANY_OF),
            "legacy_cpu_smoke_only": list(exp.LEGACY_CPU_SMOKE_ONLY),
        },
        "sota_models_cached": [
            {
                "hf_id": "unsloth/gemma-4-26B-A4B-it-GGUF",
                "path": "/cache/gemma.gguf",
                "sha256": "a" * 64,
            }
        ],
    }


def _fover_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index in range(50):
        rows.append(
            {
                "question_id": f"ok-{index}",
                "step_text": f"Step: {index} + 2 = {index + 2}. Therefore done.",
                "label": "correct",
                "confidence": 1.0,
            }
        )
        rows.append(
            {
                "question_id": f"bad-{index}",
                "step_text": f"Step: {index} + 2 = {index + 3}. Therefore done.",
                "label": "incorrect",
                "confidence": 1.0,
            }
        )
    return rows


def _telemetry_rows(*, with_topk: bool = True) -> list[dict[str, Any]]:
    top_logprobs = [
        {"A": math.log(0.9), "B": math.log(0.1)},
        {"A": math.log(0.5), "B": math.log(0.5)},
        {"A": math.log(0.8), "B": math.log(0.2)},
    ]
    rows: list[dict[str, Any]] = []
    for index, correct in enumerate([True, False, True, False], start=1):
        row: dict[str, Any] = {
            "case_id": f"telemetry-{index}",
            "known_verifier_label": 1 if correct else 0,
            "topk_alternatives_available": with_topk,
            "token_logprobs_available": with_topk,
        }
        if with_topk:
            row["top_logprobs"] = top_logprobs
        rows.append(row)
    return rows


def test_req_verify_2843_prefix_closed_false_claim_constraint() -> None:
    """REQ-VERIFY-2843-2: false arithmetic claims are terminal prefix violations."""

    constraint = exp.ArithmeticFalseClaimConstraint()

    bad = constraint.explore_prefixes("First 2 + 5 = 8.\nThen 2 + 5 = 7.", prefix_stride=10)
    good = constraint.explore_prefixes("First 2 + 5 = 7.\nThen 3 * 4 = 12.", prefix_stride=10)
    unfinished = constraint.explore_prefixes("First 2 + 5 =", prefix_stride=5)
    operator_mix = constraint.explore_prefixes("-2 + +5 = 3. 8 - 3 = 5. 6 / 3 = 2.")

    assert bad.final_state.violates_constraint is True
    assert bad.false_claim_count == 1
    assert bad.score > good.score
    assert bad.to_dict()["final_state"]["violates_constraint"] is True
    assert good.final_state.violates_constraint is False
    assert good.checked_claim_count == 2
    assert good.score == pytest.approx(0.0)
    assert unfinished.final_state.violates_constraint is False
    assert unfinished.checked_claim_count == 0
    assert operator_mix.checked_claim_count == 3
    assert operator_mix.false_claim_count == 0
    assert exp._extract_arithmetic_claims("2 + 5 = 8", terminal=False) == []


def test_req_verify_2843_auroc_and_entropy_production_features() -> None:
    """REQ-VERIFY-2843-3/5: AUROC and top-k entropy-production are deterministic."""

    assert exp.compute_auroc([0, 0, 1, 1], [0.10, 0.20, 0.80, 0.90]) == pytest.approx(1.0)
    assert exp.compute_auroc([0, 1], [0.50, 0.50]) == pytest.approx(0.5)
    with pytest.raises(ValueError, match="both positive and negative"):
        exp.compute_auroc([1, 1], [0.1, 0.2])
    with pytest.raises(ValueError, match="same length"):
        exp.compute_auroc([0], [0.1, 0.2])

    features = exp.entropy_production_from_topk(
        [
            {"A": math.log(0.9), "B": math.log(0.1)},
            {"A": math.log(0.5), "B": math.log(0.5)},
            {"A": math.log(0.8), "B": math.log(0.2)},
        ]
    )

    assert features.available is True
    assert features.position_count == 3
    assert features.mean_entropy > 0.0
    assert features.total_positive_entropy_delta > 0.0
    assert features.max_entropy >= features.mean_entropy
    assert features.to_dict()["position_count"] == 3

    empty = exp.entropy_production_from_topk([])
    assert empty.available is False
    assert empty.position_count == 0

    assert exp._safe_eval_arithmetic("letters") is None
    assert exp._safe_eval_arithmetic("1 / 0") is None
    assert exp._safe_eval_arithmetic("1 ** 2") is None
    assert exp._to_decimal("not-a-number") is None
    assert exp._entropy_from_logprob_dict({"token": "not-a-logprob"}) is None
    assert exp.ArithmeticFalseClaimConstraint().explore_prefixes("1 / 0 = 0").checked_claim_count == 0


def test_scenario_verify_2843_success_artifact_schema(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-2843: 100 FoVer-style rows produce a proxy AUROC artifact."""

    _write_json(tmp_path / "results" / exp.EXP2836_FILENAME, _exp2836_payload())
    _write_jsonl(tmp_path / "data" / "fover_corpus.jsonl", _fover_rows())
    _write_jsonl(
        tmp_path / "results" / "live_sota_balanced_telemetry_manifest_1480.jsonl",
        _telemetry_rows(with_topk=True),
    )
    output_path = tmp_path / "results" / exp.OUTPUT_FILENAME

    artifact = exp.run_experiment(
        exp.ExperimentConfig(
            repo_root=tmp_path,
            output_path=output_path,
            run_date="20260522",
            started_at=10.0,
            clock=lambda: 14.5,
        )
    )

    saved = json.loads(output_path.read_text(encoding="utf-8"))
    assert saved == artifact
    assert set(exp.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["beaver_exact"] is False
    assert artifact["bounded_prefix_probe_auc"] == pytest.approx(1.0)
    assert artifact["n_examples"] == 100
    assert artifact["entropy_production_features_available"] is True
    assert artifact["topk_logprob_source"].endswith("live_sota_balanced_telemetry_manifest_1480.jsonl")
    assert artifact["entropy_production_summary"]["n_examples"] == 4
    assert artifact["model_specs"]["headline_required_any_of"] == list(exp.HEADLINE_REQUIRED_ANY_OF)
    assert artifact["duration_s"] == pytest.approx(4.5)
    assert any(check["resource"] == "exp2836_sota_runtime_ready" for check in artifact["preconditions_checked"])
    assert artifact["failure_modes"]["proxy_not_exact_beaver"] is True


def test_req_verify_2843_missing_topk_does_not_fabricate_entropy(tmp_path: Path) -> None:
    """REQ-VERIFY-2843-5: missing top-k logprobs disables EPR features honestly."""

    _write_json(tmp_path / "results" / exp.EXP2836_FILENAME, _exp2836_payload())
    _write_jsonl(tmp_path / "data" / "fover_corpus.jsonl", _fover_rows())
    _write_jsonl(
        tmp_path / "results" / "live_sota_balanced_telemetry_manifest_1480.jsonl",
        _telemetry_rows(with_topk=False),
    )

    artifact = exp.run_experiment(
        exp.ExperimentConfig(
            repo_root=tmp_path,
            output_path=tmp_path / "results" / exp.OUTPUT_FILENAME,
            run_date="20260522",
            started_at=1.0,
            clock=lambda: 2.0,
        )
    )

    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["entropy_production_features_available"] is False
    assert artifact["topk_logprob_source"] == "unavailable"
    assert artifact["entropy_production_summary"]["n_examples"] == 0


def test_req_verify_2843_insufficient_data_and_loader_details_are_honest(
    tmp_path: Path,
) -> None:
    """REQ-VERIFY-2843-1/3/5: missing resources block or degrade without fabrication."""

    payload = _exp2836_payload()
    payload["loader_probe"] = {"llama_cpp_import_ok": False}
    _write_json(tmp_path / "results" / exp.EXP2836_FILENAME, payload)
    _write_json(
        tmp_path / "data" / "tiny_fover.json",
        [
            {"question_id": "missing-label", "step_text": "1 + 1 = 2"},
            {"question_id": "ok", "step_text": "1 + 1 = 2", "correct": True},
            {"question_id": "bad", "step_text": "1 + 1 = 3", "correct": False},
        ],
    )
    _write_jsonl(
        tmp_path / "results" / "live_sota_balanced_telemetry_manifest_1480.jsonl",
        [
            {
                "case_id": "one-class",
                "known_verifier_label": 1,
                "top_logprobs": [{"A": math.log(0.7), "B": math.log(0.3)}],
            },
            {"case_id": "missing-label", "top_logprobs": [{"A": math.log(0.7), "B": math.log(0.3)}]},
        ],
    )

    artifact = exp.run_experiment(
        exp.ExperimentConfig(
            repo_root=tmp_path,
            output_path=tmp_path / "results" / exp.OUTPUT_FILENAME,
            fover_paths=("data/tiny_fover.json",),
            n_examples=100,
            started_at=5.0,
            clock=lambda: 6.0,
        )
    )

    assert artifact["honest_verdict"] == "blocked_insufficient_labeled_examples"
    assert artifact["n_examples"] == 0
    assert artifact["duration_s"] == pytest.approx(1.0)
    assert artifact["preconditions_checked"][2]["available"] is False
    assert "loader unavailable" in artifact["preconditions_checked"][2]["detail"]
    assert exp._load_entropy_telemetry(
        tmp_path,
        ("results/live_sota_balanced_telemetry_manifest_1480.jsonl",),
    ).entropy_production_auc is None
    assert exp._balanced_prefix(
        [
            exp.LabeledExample("a", "1 + 1 = 2", 0, "fixture"),
            exp.LabeledExample("b", "1 + 1 = 2", 0, "fixture"),
        ],
        2,
    )[0].example_id == "a"


def test_req_verify_2843_exp2836_not_ready_blocks_before_measurement(tmp_path: Path) -> None:
    """REQ-VERIFY-2843-1: failed SOTA preflight blocks without scoring examples."""

    _write_json(tmp_path / "results" / exp.EXP2836_FILENAME, _exp2836_payload(ready=False))
    _write_jsonl(tmp_path / "data" / "fover_corpus.jsonl", _fover_rows())
    output_path = tmp_path / "results" / exp.OUTPUT_FILENAME

    artifact = exp.run_experiment(
        exp.ExperimentConfig(
            repo_root=tmp_path,
            output_path=output_path,
            run_date="20260522",
            started_at=3.0,
            clock=lambda: 3.25,
        )
    )

    assert json.loads(output_path.read_text(encoding="utf-8")) == artifact
    assert artifact["honest_verdict"] == "blocked_exp2836_sota_runtime_not_ready"
    assert artifact["beaver_exact"] is False
    assert artifact["bounded_prefix_probe_auc"] is None
    assert artifact["n_examples"] == 0
    assert artifact["duration_s"] == pytest.approx(0.25)


def test_req_verify_2843_missing_exp2836_and_artifact_validation(tmp_path: Path) -> None:
    """REQ-VERIFY-2843-6: terminal artifacts enforce required fields and verdict prefixes."""

    artifact = exp.run_experiment(
        exp.ExperimentConfig(
            repo_root=tmp_path,
            output_path=tmp_path / "results" / exp.OUTPUT_FILENAME,
            started_at=7.0,
            clock=lambda: 7.5,
        )
    )

    assert artifact["honest_verdict"] == "blocked_exp2836_sota_runtime_not_ready"
    assert artifact["preconditions_checked"][0]["available"] is False

    with pytest.raises(ValueError, match="missing required fields"):
        exp._validate_terminal_artifact({"honest_verdict": "complete: x", "beaver_exact": False})
    valid = {field: None for field in exp.REQUIRED_ARTIFACT_FIELDS}
    valid.update({"honest_verdict": "complete: x", "beaver_exact": False})
    exp._validate_terminal_artifact(valid)
    with pytest.raises(ValueError, match="disallowed prefix"):
        exp._validate_terminal_artifact(valid | {"honest_verdict": "partial"})
    with pytest.raises(ValueError, match="exact BEAVER"):
        exp._validate_terminal_artifact(valid | {"beaver_exact": True})
