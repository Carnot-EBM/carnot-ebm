"""Tests for Exp 2858 clean bounded-prefix/EPR proxy.

Spec: REQ-VERIFY-2858, SCENARIO-VERIFY-2858.
"""

from __future__ import annotations

import builtins
import json
import math
from pathlib import Path
from typing import Any

import pytest

from carnot.verify import clean_bounded_prefix_proxy_v2 as exp


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _fover_rows(count_per_class: int = 60) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index in range(count_per_class):
        rows.append(
            {
                "question_id": f"correct-{index}",
                "step_text": f"Trace: {index} + 2 = {index + 2}.",
                "label": "correct",
            }
        )
        rows.append(
            {
                "question_id": f"incorrect-{index}",
                "step_text": f"Trace: {index} + 2 = {index + 3}.",
                "label": "incorrect",
            }
        )
    return rows


def _telemetry_rows() -> list[dict[str, Any]]:
    return [
        {
            "case_id": "entropy-correct",
            "known_verifier_label": 1,
            "top_logprobs": [
                {"A": math.log(0.9), "B": math.log(0.1)},
                {"A": math.log(0.6), "B": math.log(0.4)},
            ],
        },
        {
            "case_id": "entropy-incorrect",
            "known_verifier_label": 0,
            "top_logprobs": [
                {"A": math.log(0.8), "B": math.log(0.2)},
                {"A": math.log(0.5), "B": math.log(0.5)},
            ],
        },
    ]


def _clean_adversarial_report(_path: Path) -> dict[str, Any]:
    return {"loaded": True, "flag_count": 0, "flags": [], "returncode": 0}


def test_scenario_verify_2858_writes_clean_proxy_artifact(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-2858: successful local scoring writes clean proxy fields."""

    _write_jsonl(tmp_path / "data" / "fover_corpus.jsonl", _fover_rows())
    _write_jsonl(
        tmp_path / "results" / "live_sota_balanced_telemetry_manifest_1480.jsonl",
        _telemetry_rows(),
    )

    artifact = exp.run_experiment(
        exp.ExperimentConfig(
            repo_root=tmp_path,
            started_at=10.0,
            clock=lambda: 12.25,
        ),
        adversarial_verify_runner=_clean_adversarial_report,
    )

    saved = json.loads((tmp_path / "results" / exp.OUTPUT_FILENAME).read_text(encoding="utf-8"))
    encoded = json.dumps(saved, sort_keys=True)
    assert saved == artifact
    assert set(exp.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["beaver_exact"] is False
    assert artifact["exact_beaver_implemented"] is False
    assert artifact["live_model_invoked"] is False
    assert artifact["bounded_prefix_proxy_auc"] == pytest.approx(1.0)
    assert isinstance(artifact["entropy_production_auc"], float)
    assert artifact["n_examples"] == 100
    assert artifact["random_seed"] == exp.RANDOM_SEED
    assert artifact["run_date"] == "20260522"
    assert artifact["duration_s"] == pytest.approx(2.25)
    assert artifact["adversarial_verify_passed"] is True
    assert artifact["adversarial_verify_flags"] == []
    assert artifact["claim_boundary"].startswith("Proxy only:")
    assert "model_specs" not in artifact
    assert "GGUF" not in encoded
    assert "CUDA" not in encoded
    assert any(
        check["step"] == "test -f data/fover_corpus.jsonl" and check["passed"]
        for check in artifact["preconditions_checked"]
    )


def test_req_verify_2858_blocks_when_fover_missing(tmp_path: Path) -> None:
    """REQ-VERIFY-2858-1: missing FoVer writes blocked_fover_dataset and exits."""

    artifact = exp.run_experiment(
        exp.ExperimentConfig(repo_root=tmp_path, started_at=1.0, clock=lambda: 1.5),
        adversarial_verify_runner=_clean_adversarial_report,
    )

    saved = json.loads((tmp_path / "results" / exp.OUTPUT_FILENAME).read_text(encoding="utf-8"))
    assert saved == artifact
    assert artifact["honest_verdict"] == "blocked_fover_dataset"
    assert artifact["bounded_prefix_proxy_auc"] == 0.0
    assert artifact["entropy_production_auc"] == 0.0
    assert artifact["n_examples"] == 0
    assert artifact["adversarial_verify_passed"] is False
    assert artifact["adversarial_verify_flags"] == []
    assert any(
        check["step"] == "test -f data/fover_corpus.jsonl" and not check["passed"]
        for check in artifact["preconditions_checked"]
    )


def test_req_verify_2858_blocks_when_metrics_dependency_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-VERIFY-2858-1: metrics precondition failure blocks before scoring."""

    _write_jsonl(tmp_path / "data" / "fover_corpus.jsonl", _fover_rows())
    output_path = tmp_path / "custom_results" / exp.OUTPUT_FILENAME
    monkeypatch.setattr(exp, "_metrics_probe", lambda: (False, "missing sklearn"))

    artifact = exp.run_experiment(
        exp.ExperimentConfig(
            repo_root=tmp_path,
            output_path=output_path,
            started_at=1.0,
            clock=lambda: 1.25,
        ),
        adversarial_verify_runner=_clean_adversarial_report,
    )

    assert json.loads(output_path.read_text(encoding="utf-8")) == artifact
    assert artifact["honest_verdict"] == "blocked_metrics_dependency"
    assert artifact["n_examples"] == 0
    assert artifact["preconditions_checked"][2]["observed"] == "missing sklearn"


def test_req_verify_2858_blocks_on_insufficient_or_one_class_rows(tmp_path: Path) -> None:
    """REQ-VERIFY-2858-1/2: insufficient labeled local rows are not scored."""

    _write_jsonl(
        tmp_path / "data" / "fover_corpus.jsonl",
        [
            {"question_id": "one", "step_text": "1 + 1 = 2", "label": "correct"},
            {"question_id": "two", "step_text": "2 + 2 = 4", "label": "correct"},
        ],
    )

    artifact = exp.run_experiment(
        exp.ExperimentConfig(repo_root=tmp_path, started_at=2.0, clock=lambda: 3.0),
        adversarial_verify_runner=_clean_adversarial_report,
    )

    assert artifact["honest_verdict"] == "blocked_fover_dataset"
    assert artifact["n_examples"] == 0
    assert artifact["duration_s"] == pytest.approx(1.0)


def test_req_verify_2858_attaches_adversarial_verify_flags(tmp_path: Path) -> None:
    """REQ-VERIFY-2858-5: adversarial verification flags are persisted."""

    _write_jsonl(tmp_path / "data" / "fover_corpus.jsonl", _fover_rows())

    def flagged_report(_path: Path) -> dict[str, Any]:
        return {
            "loaded": True,
            "flag_count": 1,
            "flags": [{"kind": "TEST_FLAG", "severity": "warn", "detail": "fixture"}],
            "returncode": 1,
        }

    artifact = exp.run_experiment(
        exp.ExperimentConfig(repo_root=tmp_path, started_at=5.0, clock=lambda: 8.0),
        adversarial_verify_runner=flagged_report,
    )

    assert artifact["adversarial_verify_passed"] is False
    assert artifact["adversarial_verify_flags"] == [
        {"kind": "TEST_FLAG", "severity": "warn", "detail": "fixture"}
    ]
    assert artifact["reproducibility_checksum"] == json.loads(
        (tmp_path / "results" / exp.OUTPUT_FILENAME).read_text(encoding="utf-8")
    )["reproducibility_checksum"]


def test_req_verify_2858_metrics_probe_and_validation_edges(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-VERIFY-2858-3/4: schema validation rejects overstated claims."""

    real_import = builtins.__import__

    def fake_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "sklearn":
            raise ModuleNotFoundError("No module named 'sklearn'", name="sklearn")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    assert exp._metrics_probe() == (False, "missing sklearn")

    valid = {field: None for field in exp.REQUIRED_ARTIFACT_FIELDS}
    valid.update(
        {
            "honest_verdict": "complete: x",
            "beaver_exact": False,
            "exact_beaver_implemented": False,
            "live_model_invoked": False,
            "run_date": "20260522",
        }
    )

    exp._validate_artifact(valid)
    with pytest.raises(ValueError, match="missing required fields"):
        exp._validate_artifact({"honest_verdict": "complete: x"})
    with pytest.raises(ValueError, match="exact BEAVER"):
        exp._validate_artifact(valid | {"beaver_exact": True})
    with pytest.raises(ValueError, match="frontier proof"):
        exp._validate_artifact(valid | {"exact_beaver_implemented": True})
    with pytest.raises(ValueError, match="live model"):
        exp._validate_artifact(valid | {"live_model_invoked": True})
    with pytest.raises(ValueError, match="model_specs"):
        exp._validate_artifact(valid | {"model_specs": []})
    with pytest.raises(ValueError, match="run_date"):
        exp._validate_artifact(valid | {"run_date": "20260101"})
