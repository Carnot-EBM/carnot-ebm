"""Tests for the stable evaluation manifest resolver contract.

Spec: REQ-BENCH-2863, SCENARIO-BENCH-2863.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from carnot.eval import manifest_resolver as mod


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _write_manifest(path: Path, rows: int = 1) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps({"stable_id": f"row-{idx}"}, sort_keys=True) for idx in range(rows)]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return _digest(path)


def _materialization_artifact(tmp_path: Path) -> Path:
    manifest_dir = tmp_path / "data" / "eval_manifests"
    paths = {
        "halueval": manifest_dir / "halueval_20260522.jsonl",
        "fever": manifest_dir / "fever_20260522.jsonl",
        "mbpp": manifest_dir / "mbpp_20260522.jsonl",
        "humaneval": manifest_dir / "humaneval_20260522.jsonl",
        "truthfulqa": manifest_dir / "truthfulqa_20260522.jsonl",
    }
    sha = {name: _write_manifest(path) for name, path in paths.items()}
    artifact = {
        "honest_verdict": "complete: local benchmark manifests materialized",
        "halueval_ready": True,
        "fever_ready": True,
        "mbpp_ready": True,
        "humaneval_ready": True,
        "truthfulqa_ready": True,
        "manifest_counts": {
            "halueval": 500,
            "fever": 500,
            "mbpp": 100,
            "humaneval": 164,
            "truthfulqa": 200,
        },
        "manifest_paths": {name: str(path) for name, path in paths.items()},
        "manifest_sha256": sha,
        "dataset_status": {
            name: {"detail": f"ready: {name}", "manifest_path": str(paths[name])}
            for name in paths
        },
        "synthetic_rows_created": False,
        "run_date": "20260522",
    }
    artifact_path = tmp_path / mod.MATERIALIZATION_ARTIFACT_REL_PATH
    _write_json(artifact_path, artifact)
    return artifact_path


def test_scenario_bench_2863_halueval_and_fever_resolve_dated_paths(
    tmp_path: Path,
) -> None:
    """SCENARIO-BENCH-2863: HaluEval/FEVER use Exp 2849 dated paths and checksums."""

    _materialization_artifact(tmp_path)

    resolved = mod.resolve_manifest_contract(
        repo_root=tmp_path,
        corpora=("halueval", "fever"),
    )

    assert set(resolved) == {"halueval", "fever"}
    assert resolved["halueval"].contract_ready is True
    assert resolved["fever"].contract_ready is True
    assert resolved["halueval"].path.endswith("halueval_20260522.jsonl")
    assert resolved["fever"].path.endswith("fever_20260522.jsonl")
    assert not resolved["halueval"].path.endswith("halueval.jsonl")
    assert not resolved["fever"].path.endswith("fever.jsonl")
    assert resolved["halueval"].sha256 == _digest(Path(resolved["halueval"].path))
    assert resolved["fever"].sha256 == _digest(Path(resolved["fever"].path))


def test_req_bench_2863_writes_required_contract_artifact(tmp_path: Path) -> None:
    """REQ-BENCH-2863: contract artifact exposes all corpus readiness booleans."""

    _materialization_artifact(tmp_path)

    artifact = mod.write_contract_artifact(
        repo_root=tmp_path,
        tests_run=[".venv/bin/pytest tests/python/test_eval_manifest_resolver.py -q"],
        started_at=2.0,
        clock=lambda: 5.25,
    )

    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["manifest_contract_ready"] is True
    assert artifact["manifest_source_artifact"] == str(mod.MATERIALIZATION_ARTIFACT_REL_PATH)
    assert artifact["synthetic_rows_created"] is False
    assert artifact["run_date"] == "20260522"
    assert artifact["duration_s"] == 3.25
    assert artifact["tests_run"] == [
        ".venv/bin/pytest tests/python/test_eval_manifest_resolver.py -q"
    ]
    for name in mod.CANONICAL_CORPORA:
        assert artifact[f"{name}_ready"] is True
        assert artifact["resolved_manifest_paths"][name].endswith(f"{name}_20260522.jsonl")
        assert len(artifact["resolved_manifest_sha256"][name]) == 64
    saved = json.loads((tmp_path / mod.OUTPUT_REL_PATH).read_text(encoding="utf-8"))
    assert saved == artifact


def test_req_bench_2863_blocks_on_checksum_mismatch(tmp_path: Path) -> None:
    """REQ-BENCH-2863: checksum mismatch prevents a ready contract verdict."""

    artifact_path = _materialization_artifact(tmp_path)
    payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    payload["manifest_sha256"]["fever"] = "0" * 64
    _write_json(artifact_path, payload)

    artifact = mod.build_contract_artifact(
        repo_root=tmp_path,
        tests_run=[],
        started_at=10.0,
        clock=lambda: 10.5,
    )

    assert artifact["honest_verdict"] == "blocked_eval_manifest_contract"
    assert artifact["manifest_contract_ready"] is False
    assert artifact["halueval_ready"] is True
    assert artifact["fever_ready"] is False
    assert artifact["checksum_verified"]["fever"] is False
    assert artifact["resolved_manifest_paths"]["fever"].endswith("fever_20260522.jsonl")
