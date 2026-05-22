"""Tests for Exp 2864 HaluEval/FEVER local-manifest calibration.

Spec: REQ-BENCH-2864, SCENARIO-BENCH-2864.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from carnot.eval import halueval_fever_full_calibration as mod


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    return _sha256(path)


def _rows(dataset: str, *, one_label_only: bool = False) -> list[dict[str, Any]]:
    labels = [0, 0, 1, 1]
    if one_label_only:
        labels = [0, 0, 0, 0]
    key = "candidate" if dataset == "halueval" else "claim"
    return [
        {
            "dataset": "HaluEval" if dataset == "halueval" else "FEVER",
            "stable_id": f"{dataset}-{idx}",
            "prompt": f"Context {idx}",
            key: f"{'unsupported' if label else 'grounded'} answer {idx}",
            "reference": f"reference {idx}",
            "source_name": "fixture",
            "label": label,
        }
        for idx, label in enumerate(labels)
    ]


def _contract_artifact(
    tmp_path: Path,
    *,
    halueval_rows: list[dict[str, Any]] | None = None,
    fever_rows: list[dict[str, Any]] | None = None,
    halueval_ready: bool = True,
    fever_ready: bool = True,
    corrupt_fever_sha: bool = False,
) -> tuple[Path, dict[str, Path], dict[str, str]]:
    manifest_dir = tmp_path / "data" / "eval_manifests"
    paths = {
        "halueval": manifest_dir / "halueval_20260522.jsonl",
        "fever": manifest_dir / "fever_20260522.jsonl",
    }
    sha = {
        "halueval": _write_jsonl(paths["halueval"], halueval_rows or _rows("halueval")),
        "fever": _write_jsonl(paths["fever"], fever_rows or _rows("fever")),
    }
    if corrupt_fever_sha:
        sha["fever"] = "0" * 64
    artifact = {
        "honest_verdict": "complete: eval manifest contract ready",
        "halueval_ready": halueval_ready,
        "fever_ready": fever_ready,
        "manifest_contract_ready": halueval_ready and fever_ready,
        "resolved_manifest_paths": {name: str(path) for name, path in paths.items()},
        "resolved_manifest_sha256": sha,
        "resolved_manifest_counts": {
            "halueval": len(halueval_rows or _rows("halueval")),
            "fever": len(fever_rows or _rows("fever")),
        },
        "run_date": "20260522",
    }
    artifact_path = tmp_path / mod.EXP2863_REL_PATH
    _write_json(artifact_path, artifact)
    return artifact_path, paths, sha


def _label_score(example: mod.CalibrationExample) -> float:
    return 0.2 if example.label == 0 else 0.8


def test_scenario_bench_2864_full_calibration_uses_exp2863_paths(
    tmp_path: Path,
) -> None:
    """SCENARIO-BENCH-2864: dated Exp2863 manifests drive metrics and artifact fields."""

    _contract_artifact(tmp_path)

    artifact = mod.run_calibration(
        mod.CalibrationConfig(
            repo_root=tmp_path,
            results_dir=tmp_path / "results",
            bootstrap_reps=20,
            random_seed=2864,
            started_at=5.0,
            clock=lambda: 8.5,
        ),
        scorer=_label_score,
        adversarial_verifier=lambda _path: {"passed": True, "flags": []},
        write=True,
    )

    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["halueval_fever_ready"] is True
    assert artifact["full_benchmark_ready"] is True
    assert artifact["live_model_invoked"] is False
    assert artifact["halueval_auroc"] == pytest.approx(1.0)
    assert artifact["fever_auroc"] == pytest.approx(1.0)
    assert artifact["halueval_n_examples"] == 4
    assert artifact["fever_n_examples"] == 4
    assert artifact["label_counts_by_dataset"] == {
        "fever": {"0": 2, "1": 2},
        "halueval": {"0": 2, "1": 2},
    }
    assert artifact["auroc_ci95_by_dataset"]["halueval"] == [1.0, 1.0]
    assert artifact["auroc_ci95_by_dataset"]["fever"] == [1.0, 1.0]
    assert artifact["random_seed"] == 2864
    assert len(artifact["reproducibility_checksum"]) == 64
    assert artifact["run_date"] == "20260522"
    assert artifact["duration_s"] == pytest.approx(3.5)
    assert artifact["adversarial_verify_passed"] is True
    assert artifact["adversarial_verify_flags"] == []
    assert artifact["field_principles"]["live_model_invoked"].startswith("Always false")
    for name, path in artifact["manifest_paths_used"].items():
        assert path.endswith(f"{name}_20260522.jsonl")
        assert not path.endswith(f"{name}.jsonl")
        assert artifact["manifest_sha256_used"][name] == _sha256(Path(path))
    assert any(row["check"] == "exp2863_halueval_ready" for row in artifact["preconditions_checked"])
    saved = json.loads((tmp_path / "results" / mod.OUTPUT_FILENAME).read_text(encoding="utf-8"))
    assert saved == artifact


def test_req_bench_2864_blocks_on_missing_readiness_or_checksum(tmp_path: Path) -> None:
    """REQ-BENCH-2864: readiness and checksum failures block metrics honestly."""

    _contract_artifact(tmp_path, halueval_ready=True, fever_ready=True, corrupt_fever_sha=True)

    artifact = mod.run_calibration(
        mod.CalibrationConfig(repo_root=tmp_path, results_dir=tmp_path / "results"),
        scorer=_label_score,
        adversarial_verifier=lambda _path: {"passed": True, "flags": []},
        write=False,
    )

    assert artifact["honest_verdict"] == "blocked_eval_manifest_contract"
    assert artifact["halueval_fever_ready"] is False
    assert artifact["full_benchmark_ready"] is False
    assert artifact["halueval_auroc"] is None
    assert artifact["fever_auroc"] is None
    assert artifact["halueval_n_examples"] == 0
    assert artifact["fever_n_examples"] == 0
    assert artifact["auroc_ci95_by_dataset"] == {}
    assert artifact["label_counts_by_dataset"] == {}
    assert artifact["manifest_sha256_used"]["fever"] == "0" * 64
    assert any(
        row["check"] == "manifest_checksum_fever" and row["passed"] is False
        for row in artifact["preconditions_checked"]
    )


def test_req_bench_2864_writes_nulls_for_unavailable_auroc_and_persists_flags(
    tmp_path: Path,
) -> None:
    """REQ-BENCH-2864: one-class labels produce null metrics and verifier flags persist."""

    _contract_artifact(tmp_path, halueval_rows=_rows("halueval", one_label_only=True))

    artifact = mod.run_calibration(
        mod.CalibrationConfig(
            repo_root=tmp_path,
            results_dir=tmp_path / "results",
            bootstrap_reps=5,
            random_seed=11,
            started_at=1.0,
            clock=lambda: 3.0,
        ),
        scorer=_label_score,
        adversarial_verifier=lambda _path: {
            "passed": False,
            "flags": [{"kind": "UNIT_TEST", "severity": "warn", "detail": "fixture"}],
        },
        write=True,
    )

    assert artifact["honest_verdict"] == "blocked_unavailable_auroc"
    assert artifact["halueval_fever_ready"] is True
    assert artifact["full_benchmark_ready"] is False
    assert artifact["halueval_auroc"] is None
    assert artifact["fever_auroc"] == pytest.approx(1.0)
    assert artifact["auroc_ci95_by_dataset"]["halueval"] is None
    assert artifact["label_counts_by_dataset"]["halueval"] == {"0": 4}
    assert artifact["adversarial_verify_passed"] is False
    assert artifact["adversarial_verify_flags"] == [
        {"kind": "UNIT_TEST", "severity": "warn", "detail": "fixture"}
    ]
    saved = json.loads((tmp_path / "results" / mod.OUTPUT_FILENAME).read_text(encoding="utf-8"))
    assert saved["adversarial_verify_flags"] == artifact["adversarial_verify_flags"]


def test_req_bench_2864_edge_paths_for_coverage(tmp_path: Path) -> None:
    """REQ-BENCH-2864: malformed rows, scoring failures, and CLI edges stay explicit."""

    bad_rows = [
        {"stable_id": "skip-label", "prompt": "p", "candidate": "c", "label": "bad"},
        {"stable_id": "skip-candidate", "prompt": "p", "label": 1},
        {"stable_id": "ok", "prompt": "p", "candidate": "c", "label": "1"},
    ]
    path = tmp_path / "manifest.jsonl"
    _write_jsonl(path, bad_rows)

    examples = mod.load_manifest_examples(path, "halueval")

    assert [example.stable_id for example in examples] == ["ok"]
    assert examples[0].score_text == "p\nReference: \nCandidate: c"

    metrics = mod.evaluate_examples(
        examples,
        scorer=lambda _example: float("nan"),
        bootstrap_reps=1,
        seed=1,
    )
    assert metrics["n_examples"] == 1
    assert metrics["n_scored"] == 0
    assert metrics["auroc"] is None
    assert metrics["auroc_ci95"] is None

    assert mod._load_exp2863(tmp_path / "missing.json") == {}
    assert mod._metric_value({"auroc": None}) is None
    assert mod._ci_value({"auroc_ci95": [None, None]}) is None
    assert mod._real_adversarial_verify(tmp_path / "missing-artifact.json")["passed"] is False
    assert mod.main(["--repo-root", str(tmp_path), "--no-adversarial-verify"]) == 0
