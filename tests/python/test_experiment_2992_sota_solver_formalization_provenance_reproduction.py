"""Tests for Exp 2992 SOTA solver provenance reproduction.

Spec refs: REQ-VERIFY-2992, SCENARIO-VERIFY-2992.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from carnot.eval import sota_solver_formalization_provenance_reproduction_v1 as exp


REPO_ROOT = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO_ROOT / "openspec" / "capabilities" / "verification" / "spec.md"
MANDATED = "unsloth/gemma-4-26B-A4B-it-GGUF"


def _write_prior_exp2980(root: Path) -> None:
    path = root / "results" / exp.EXP2980_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "honest_verdict": "complete: prior fixture",
                "formalization_feedback_clean": True,
                "headline_result": True,
                "n_items": 6,
                "parseability_rate": 1.0,
                "z3_execution_rate": 1.0,
                "solver_verified_accuracy": 1.0,
                "feedback_repair_delta": 0.833333,
                "tautology_flag_rate": 0.0,
                "duration_s": 42.854308,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _model_file(tmp_path: Path) -> Path:
    path = tmp_path / "models" / "gemma-4-26B-A4B-it-UD-Q4_K_M.gguf"
    path.parent.mkdir(parents=True)
    path.write_text("tiny checksum fixture\n", encoding="utf-8")
    return path


def _runtime_probe() -> dict[str, Any]:
    return {
        "cuda_available": True,
        "cuda_device_count": 2,
        "llama_cpp_import_ok": True,
        "llama_cpp_supports_gpu_offload": True,
        "detail": "fixture runtime ready",
    }


def _config(tmp_path: Path, *, elapsed: float = 70.0) -> exp.ExperimentConfig:
    return exp.ExperimentConfig(
        repo_root=tmp_path,
        output_path=tmp_path / "results" / exp.OUTPUT_FILENAME,
        raw_response_dir=tmp_path / "raw",
        z3_transcript_dir=tmp_path / "z3",
        started_at=10.0,
        clock=lambda: 10.0 + elapsed,
        monotonic_clock=lambda: 20.0,
    )


def _collect_unparseable(
    spec: dict[str, Any],
    items: list[exp.FeedbackFrontierItem],
    config: exp.ExperimentConfig,
) -> dict[str, Any]:
    rows = []
    for index, item in enumerate(items):
        prompt = exp.feedback_prompt(item)
        rows.append(
            {
                "item_id": item.item_id,
                "model_hf_id": spec["hf_id"],
                "model_name": spec["name"],
                "model_path": spec["model_path"],
                "gpu_index": spec.get("gpu", 0),
                "prompt_hash": exp.sha256_text(prompt),
                "prompt_text": prompt,
                "per_item_seed": config.random_seed + index,
                "generation_source": "live_provenance_reproduction",
                "output_text": "not json",
                "raw_response_path": f"/tmp/{item.item_id}.json",
                "raw_response_sha256": exp.sha256_text("not json"),
                "elapsed_seconds": 1.0,
                "blocker": None,
            }
        )
    return {
        "summary": {
            "hf_id": spec["hf_id"],
            "model_name": spec["name"],
            "model_path": spec["model_path"],
            "model_used": True,
            "blocker": None,
            "live_inference_duration_s": float(len(items)),
        },
        "rows": rows,
    }


def test_req_verify_2992_spec_anchor_exists() -> None:
    """REQ-VERIFY-2992: the reproduction runner is OpenSpec anchored."""
    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-VERIFY-2992" in spec
    assert "SCENARIO-VERIFY-2992" in spec
    assert exp.OUTPUT_FILENAME in spec
    assert "solver_provenance_reproduced" in spec


def test_scenario_verify_2992_reproduces_with_hashes_and_larger_item_set(
    tmp_path: Path,
) -> None:
    """SCENARIO-VERIFY-2992: a 12-item live run records replayable solver evidence."""
    _write_prior_exp2980(tmp_path)
    model_path = _model_file(tmp_path)

    artifact = exp.run_experiment(
        _config(tmp_path),
        cached_pair_provider=lambda **_: None,
        individual_model_resolver=lambda hf_id: str(model_path) if hf_id == MANDATED else None,
        runtime_probe_fn=_runtime_probe,
        collect_model_outputs_fn=_collect_unparseable,
    )
    saved = json.loads((tmp_path / "results" / exp.OUTPUT_FILENAME).read_text(encoding="utf-8"))

    assert saved == artifact
    assert set(exp.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert artifact["honest_verdict"].startswith("reproduced:")
    assert artifact["solver_provenance_reproduced"] is True
    assert artifact["formalization_clean"] is True
    assert artifact["n_items"] == exp.FIXED_ITEM_COUNT
    assert artifact["parseability"] == pytest.approx(1.0)
    assert artifact["z3_execution_rate"] == pytest.approx(1.0)
    assert artifact["solver_verified_accuracy"] == pytest.approx(1.0)
    assert artifact["feedback_repair_delta"] == pytest.approx(1.0)
    assert artifact["tautology_rate"] == pytest.approx(0.0)
    assert artifact["prompt_hashes_recorded"] is True
    assert artifact["z3_transcript_hashes_recorded"] is True
    assert artifact["model_checksums_recorded"] is True
    assert artifact["duration_seconds"] == pytest.approx(70.0)
    assert artifact["inference_substrate"] == exp.INFERENCE_SUBSTRATE
    assert artifact["models_used"] == [MANDATED]
    assert artifact["comparison_to_exp2980"]["prior_n_items"] == 6
    assert artifact["comparison_to_exp2980"]["n_item_delta"] == exp.FIXED_ITEM_COUNT - 6
    assert artifact["per_item_results"][0]["initial_result"]["failure_category"] == "unparseable"
    assert artifact["per_item_results"][0]["final_result"]["failure_category"] == "solver_verified_correct"
    assert artifact["per_item_results"][0]["z3_transcript_sha256"]
    assert artifact["per_item_results"][0]["final_z3_input_sha256"]
    assert artifact["raw_model_outputs_recorded"] is True
    exp.validate_artifact(artifact)


def test_req_verify_2992_blocks_without_headline_preconditions(tmp_path: Path) -> None:
    """REQ-VERIFY-2992: missing prior/model/runtime evidence fails closed."""
    calls: list[str] = []

    artifact = exp.run_experiment(
        exp.ExperimentConfig(
            repo_root=tmp_path,
            output_path=tmp_path / "results" / exp.OUTPUT_FILENAME,
            raw_response_dir=tmp_path / "raw",
            z3_transcript_dir=tmp_path / "z3",
            max_items=3,
            started_at=10.0,
            clock=lambda: 10.25,
        ),
        cached_pair_provider=lambda **_: (_ for _ in ()).throw(RuntimeError("cache down")),
        individual_model_resolver=lambda _hf_id: None,
        runtime_probe_fn=lambda: {
            "cuda_available": False,
            "cuda_device_count": 0,
            "llama_cpp_import_ok": False,
            "llama_cpp_supports_gpu_offload": False,
        },
        collect_model_outputs_fn=lambda *_args: calls.append("unexpected") or {},
    )

    assert calls == []
    assert artifact["honest_verdict"].startswith("blocked:")
    assert artifact["solver_provenance_reproduced"] is False
    assert artifact["formalization_clean"] is False
    assert artifact["n_items"] == 3
    assert artifact["prompt_hashes_recorded"] is False
    assert artifact["z3_transcript_hashes_recorded"] is False
    assert artifact["model_checksums_recorded"] is False
    assert artifact["inference_substrate"] == "blocked_precondition"
    assert "fixed_item_set_below_12" in artifact["blocking_reasons"]
    assert any("cache down" in str(row["detail"]) for row in artifact["preconditions_checked"])
    assert _config(tmp_path).response_dir() == tmp_path / "raw"
    exp.validate_artifact(artifact)


def test_req_verify_2992_duration_policy_and_validation(tmp_path: Path) -> None:
    """REQ-VERIFY-2992: implausibly short live inference cannot be reproduced."""
    _write_prior_exp2980(tmp_path)
    model_path = _model_file(tmp_path)

    artifact = exp.run_experiment(
        _config(tmp_path, elapsed=12.0),
        cached_pair_provider=lambda **_: [
            {"name": "Gemma4-26B-A4B-it", "hf_id": MANDATED, "gpu": 0, "model_path": str(model_path)},
            {
                "name": "Qwen3.6-35B-A3B",
                "hf_id": "unsloth/Qwen3.6-35B-A3B-GGUF",
                "gpu": 1,
                "model_path": str(model_path),
            },
        ],
        individual_model_resolver=lambda _hf_id: None,
        runtime_probe_fn=_runtime_probe,
        collect_model_outputs_fn=_collect_unparseable,
    )

    assert artifact["formalization_clean"] is True
    assert artifact["solver_provenance_reproduced"] is False
    assert artifact["honest_verdict"].startswith("flagged:")
    assert "duration_below_live_headline_floor" in artifact["non_reproduction_reasons"]
    exp.validate_artifact(artifact)

    with pytest.raises(ValueError, match="missing required fields"):
        exp.validate_artifact({"honest_verdict": "blocked: x"})
    with pytest.raises(ValueError, match="implausible duration"):
        exp.validate_artifact(artifact | {"solver_provenance_reproduced": True})

    reproduced_base = artifact | {
        "solver_provenance_reproduced": True,
        "duration_seconds": 70.0,
        "n_items": exp.FIXED_ITEM_COUNT,
        "formalization_clean": True,
        "prompt_hashes_recorded": True,
        "z3_transcript_hashes_recorded": True,
        "model_checksums_recorded": True,
        "z3_execution_rate": 1.0,
        "inference_substrate": exp.INFERENCE_SUBSTRATE,
        "honest_verdict": "reproduced: fixture",
    }
    exp.validate_artifact(reproduced_base)
    with pytest.raises(ValueError, match="honest_verdict"):
        exp.validate_artifact(reproduced_base | {"honest_verdict": "success: wrong prefix"})
    with pytest.raises(ValueError, match="inference_substrate"):
        exp.validate_artifact(reproduced_base | {"inference_substrate": "blocked_precondition"})
    with pytest.raises(ValueError, match="at least 12"):
        exp.validate_artifact(reproduced_base | {"n_items": 11})
    with pytest.raises(ValueError, match="formalization_clean"):
        exp.validate_artifact(reproduced_base | {"formalization_clean": False})
    with pytest.raises(ValueError, match="prompt_hashes_recorded"):
        exp.validate_artifact(reproduced_base | {"prompt_hashes_recorded": False})
    with pytest.raises(ValueError, match="complete Z3 execution"):
        exp.validate_artifact(reproduced_base | {"z3_execution_rate": 0.99})


def test_req_verify_2992_helper_edges_are_deterministic(tmp_path: Path) -> None:
    """REQ-VERIFY-2992: helper edge cases remain deterministic and fail closed."""
    missing = tmp_path / "missing.gguf"
    large = tmp_path / "large.gguf"
    large.write_text("abcdef", encoding="utf-8")
    model_path = _model_file(tmp_path)

    pair_specs, pair_used, cache_error = exp.resolve_headline_model_specs(
        lambda **_: [
            {"name": "Gemma4-26B-A4B-it", "hf_id": MANDATED, "gpu": 0, "model_path": str(model_path)},
            {"name": "ignored", "hf_id": "legacy", "gpu": 1, "model_path": str(model_path)},
        ],
        lambda _hf_id: None,
    )
    assert pair_used is True
    assert cache_error is None
    assert [spec["hf_id"] for spec in pair_specs] == [MANDATED]
    assert exp.model_checksum(None)["status"] == "missing"
    assert exp.model_checksum(missing)["path"] == str(missing)
    assert exp.model_checksum(large, full_sha_max_bytes=1)["bounded_sha256"]
    assert exp.honest_verdict(False, False, None) == "flagged: insufficient evidence for reproduction"

    reasons = exp.non_reproduction_reasons(
        n_items=1,
        models_used=[],
        formalization_clean=False,
        prompt_hashes_recorded=False,
        z3_transcript_hashes_recorded=False,
        model_checksums_recorded=False,
        duration_seconds=0.1,
        z3_execution_rate=0.5,
        comparison={"regression": True},
    )
    assert reasons == [
        "fixed_item_set_below_12",
        "no_headline_model_live_output",
        "formalization_not_clean",
        "prompt_hashes_missing",
        "z3_transcript_hashes_missing",
        "model_checksums_missing",
        "duration_below_live_headline_floor",
        "z3_execution_not_complete",
        "solver_accuracy_regressed_vs_exp2980",
    ]

    _write_prior_exp2980(tmp_path)
    blocked = exp.check_preconditions(
        _config(tmp_path),
        cached_pair_provider=lambda **_: None,
        individual_model_resolver=lambda _hf_id: str(missing),
        runtime_probe_fn=_runtime_probe,
        z3_module=None,
    )
    assert "z3_unavailable" in blocked.blocking_reasons
    assert "headline_model_checksum_missing" in blocked.blocking_reasons
