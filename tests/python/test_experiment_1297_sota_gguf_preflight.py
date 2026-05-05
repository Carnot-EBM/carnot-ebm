"""Tests for the Exp 1297 SOTA GGUF cache/provenance preflight.

Spec: REQ-INFER-SOTA-004,
      SCENARIO-INFER-SOTA-004-001,
      SCENARIO-INFER-SOTA-004-002,
      SCENARIO-INFER-SOTA-004-003
"""

from __future__ import annotations

import json
from pathlib import Path

from carnot.reporting.sota_gguf_preflight import (
    MANDATED_SOTA_MODEL_IDS,
    REQUIRED_ARTIFACT_FIELDS,
    build_preflight_artifact,
    run_experiment,
)


def _write_prior_coverage(project_root: Path, *, ok: bool = True) -> Path:
    results = project_root / "results"
    results.mkdir(exist_ok=True)
    path = results / "experiment_1296_prior_failures_activation_audit.json"
    path.write_text(
        json.dumps(
            {
                "status": "complete",
                "run_date": "20260505",
                "prior_failures_coverage_ok": ok,
                "n_prior_failures_checks": 13,
                "n_prior_failures_missing": 0 if ok else 2,
                "honest_verdict": "activation_audit_passed" if ok else "activation_audit_blocked",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def _ready_paths() -> dict[str, str]:
    return {
        "unsloth/Qwen3.6-35B-A3B-GGUF": "/cache/Qwen3.6-35B-A3B-Q4_K_M.gguf",
        "unsloth/gemma-4-26B-A4B-it-GGUF": "/cache/gemma-4-26B-A4B-it-UD-Q4_K_M.gguf",
    }


def test_exp1297_ready_artifact_uses_local_cached_pair(tmp_path: Path) -> None:
    """REQ-INFER-SOTA-004 / SCENARIO-INFER-SOTA-004-001: cached pair is headline-ready."""
    _write_prior_coverage(tmp_path, ok=True)
    calls: list[tuple[tuple[int, int], str]] = []
    paths = _ready_paths()

    def resolver(hf_id: str, preferred_quant: str) -> str | None:
        assert preferred_quant == "Q4_K_M"
        return paths.get(hf_id)

    def cached_pair(*, gpu_indices: tuple[int, int], preferred_quant: str) -> list[dict]:
        calls.append((gpu_indices, preferred_quant))
        return [
            {
                "name": "Qwen3.6-35B-A3B",
                "hf_id": "unsloth/Qwen3.6-35B-A3B-GGUF",
                "gpu": gpu_indices[0],
                "model_path": paths["unsloth/Qwen3.6-35B-A3B-GGUF"],
            },
            {
                "name": "Gemma4-26B-A4B-it",
                "hf_id": "unsloth/gemma-4-26B-A4B-it-GGUF",
                "gpu": gpu_indices[1],
                "model_path": paths["unsloth/gemma-4-26B-A4B-it-GGUF"],
            },
        ]

    artifact = build_preflight_artifact(
        project_root=tmp_path,
        run_date="20260505",
        resolver_fn=resolver,
        cached_pair_fn=cached_pair,
    )

    assert set(REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert calls == [((0, 1), "Q4_K_M")]
    assert artifact["status"] == "complete"
    assert artifact["cached_sota_ready"] is True
    assert artifact["headline_result_possible"] is True
    assert artifact["provenance_ok"] is True
    assert artifact["models_used"] == [
        "unsloth/Qwen3.6-35B-A3B-GGUF",
        "unsloth/gemma-4-26B-A4B-it-GGUF",
    ]
    assert artifact["missing_models"] == ["unsloth/gemma-4-31B-it-GGUF"]
    assert [row["hf_id"] for row in artifact["model_specs_preview"]] == list(
        MANDATED_SOTA_MODEL_IDS
    )
    assert artifact["model_specs_preview"][0]["cached"] is True
    assert artifact["model_specs_preview"][0]["quantization_suffix"] == "Q4_K_M"
    assert artifact["model_specs_preview"][2]["quantization_suffix"] == "UD-Q4_K_M"
    assert artifact["cached_sota_pair_returned_two_loadable_specs"] is True
    assert artifact["honest_verdict"] == "sota_gguf_cache_ready"


def test_exp1297_missing_cache_blocks_headline_result(tmp_path: Path) -> None:
    """SCENARIO-INFER-SOTA-004-002: missing local files keep headline work blocked."""
    _write_prior_coverage(tmp_path, ok=True)

    artifact = build_preflight_artifact(
        project_root=tmp_path,
        run_date="20260505",
        resolver_fn=lambda hf_id, preferred_quant: None,
        cached_pair_fn=lambda *, gpu_indices, preferred_quant: None,
    )

    assert artifact["cached_sota_ready"] is False
    assert artifact["headline_result_possible"] is False
    assert artifact["provenance_ok"] is True
    assert artifact["models_used"] == []
    assert artifact["missing_models"] == list(MANDATED_SOTA_MODEL_IDS)
    assert all(row["model_path"] is None for row in artifact["model_specs_preview"])
    assert artifact["cached_sota_pair_returned_two_loadable_specs"] is False
    assert artifact["honest_verdict"] == "sota_gguf_cache_not_ready"


def test_exp1297_prior_failure_coverage_gates_provenance(tmp_path: Path) -> None:
    """SCENARIO-INFER-SOTA-004-003: failed Exp 1296 coverage blocks provenance."""
    _write_prior_coverage(tmp_path, ok=False)
    paths = _ready_paths()

    artifact = build_preflight_artifact(
        project_root=tmp_path,
        run_date="20260505",
        resolver_fn=lambda hf_id, preferred_quant: paths.get(hf_id),
        cached_pair_fn=lambda *, gpu_indices, preferred_quant: [
            {
                "name": "Qwen3.6-35B-A3B",
                "hf_id": "unsloth/Qwen3.6-35B-A3B-GGUF",
                "gpu": gpu_indices[0],
                "model_path": paths["unsloth/Qwen3.6-35B-A3B-GGUF"],
            },
            {
                "name": "Gemma4-26B-A4B-it",
                "hf_id": "unsloth/gemma-4-26B-A4B-it-GGUF",
                "gpu": gpu_indices[1],
                "model_path": paths["unsloth/gemma-4-26B-A4B-it-GGUF"],
            },
        ],
    )

    assert artifact["cached_sota_ready"] is True
    assert artifact["provenance_ok"] is False
    assert artifact["headline_result_possible"] is False
    assert artifact["prior_failure_coverage"]["artifact_found"] is True
    assert artifact["prior_failure_coverage"]["prior_failures_coverage_ok"] is False
    assert artifact["honest_verdict"] == "sota_gguf_cache_ready_but_provenance_blocked"


def test_exp1297_run_experiment_writes_required_artifact(tmp_path: Path) -> None:
    """REQ-INFER-SOTA-004: runner writes the stable v2 JSON artifact schema."""
    _write_prior_coverage(tmp_path, ok=True)
    output_path = tmp_path / "results" / "experiment_1297_sota_gguf_cache_provenance_preflight_v2.json"

    artifact = run_experiment(
        project_root=tmp_path,
        run_date="20260505",
        output_path=output_path,
        resolver_fn=lambda hf_id, preferred_quant: None,
        cached_pair_fn=lambda *, gpu_indices, preferred_quant: None,
    )
    written = json.loads(output_path.read_text(encoding="utf-8"))

    assert written == artifact
    assert written["artifact_metadata"]["project_root"] == str(tmp_path)
    assert written["artifact_metadata"]["run_date"] == "20260505"
    assert written["status"] == "complete"
    assert written["honest_verdict"] == "sota_gguf_cache_not_ready"
