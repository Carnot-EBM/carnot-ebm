"""Tests for Exp5708 raw-response SOTA exact-constraint canary.

Spec refs: REQ-VERIFY-5708, SCENARIO-VERIFY-5708.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_5708_sota_exact_constraint_canary as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec/capabilities/verification/spec.md"
TEST_COMMAND = (
    ".venv/bin/pytest tests/python/test_experiment_5708_sota_exact_constraint_canary.py "
    "-q --no-cov -n 0"
)
COVERAGE_COMMAND = (
    ".venv/bin/coverage run "
    "--include=python/carnot/experiment_5708_sota_exact_constraint_canary.py "
    "-m pytest tests/python/test_experiment_5708_sota_exact_constraint_canary.py "
    "-q --no-cov -n 0 && .venv/bin/coverage report "
    "--include=python/carnot/experiment_5708_sota_exact_constraint_canary.py "
    "--fail-under=100"
)
FULL_TEST_COMMAND = ".venv/bin/pytest tests/python -q"
SPEC_COMMAND = ".venv/bin/python scripts/check_spec_coverage.py"
ADVERSARIAL_COMMAND = (
    ".venv/bin/python scripts/adversarial_verify.py "
    "results/experiment_5708_sota_exact_constraint_canary.json"
)
ROOT_CLUTTER_COMMAND = ".venv/bin/python scripts/root_clutter_sweep.py"
TESTS_ADDED_OR_REUSED = [
    TEST_COMMAND,
    COVERAGE_COMMAND,
    FULL_TEST_COMMAND,
    SPEC_COMMAND,
    ADVERSARIAL_COMMAND,
    ROOT_CLUTTER_COMMAND,
]


def _fake_model_path(tmp_path: Path) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    path = tmp_path / "gemma-4-26B-A4B-it-Q4_K_M.gguf"
    path.write_bytes(b"GGUF-fixture-exp5708-" + mod.MODEL_REPO_ID.encode("utf-8"))
    return path


def _runner(
    model_spec: dict[str, Any],
    panel: list[dict[str, Any]],
    generation_config: dict[str, Any],
    random_seeds: dict[str, int],
) -> dict[str, Any]:
    rows = []
    for index, row in enumerate(panel):
        raw_text = f"ANSWER: {mod.expected_answer_text(row)}"
        rows.append(
            {
                "row_id": row["row_id"],
                "prompt": row["prompt"],
                "raw_text": raw_text,
                "finish_reason": "stop",
                "token_counts": {
                    "prompt_tokens": len(row["prompt"].split()),
                    "completion_tokens": len(raw_text.split()),
                    "total_tokens": len(row["prompt"].split()) + len(raw_text.split()),
                },
                "timing": {"load_s": 0.0, "generation_s": round(0.01 + index / 10000, 6)},
                "seed": random_seeds["base_seed"] + index,
                "generation_config": dict(generation_config),
                "model_hash": model_spec["model_hash"],
                "telemetry": {"gpu_memory_peak_mb": 6144, "n_gpu_layers_offloaded": 42},
            }
        )
    return {
        "llama_cpp_version": "0.3.99-fixture",
        "llama_cpp_build_info": {
            "cuda_backend": True,
            "system_info": "CUDA = 1 | ggml-cuda present",
            "module": "llama_cpp",
        },
        "cuda_device_receipt": {
            "devices": [
                {
                    "index": 0,
                    "name": "NVIDIA GeForce RTX 3090",
                    "driver_version": "610.43.03",
                    "memory_total_mb": 24576,
                    "memory_free_mb": 22000,
                }
            ],
            "ram_reservation_mb": 32768,
            "vram_reservation_mb": 4096,
        },
        "n_gpu_layers_requested": -1,
        "n_gpu_layers_offloaded": 42,
        "gpu_memory_before_mb": 128,
        "gpu_memory_peak_mb": 6144,
        "gpu_memory_after_mb": 160,
        "cuda_offload_authenticated": True,
        "offload_log_excerpt": "llama_model_load_tensors: offloaded 42/42 layers to GPU",
        "rows": rows,
    }


def _blocked_runner(
    model_spec: dict[str, Any],
    panel: list[dict[str, Any]],
    generation_config: dict[str, Any],
    random_seeds: dict[str, int],
) -> dict[str, Any]:
    del model_spec, generation_config, random_seeds
    return {
        "llama_cpp_version": "0.3.99-fixture",
        "llama_cpp_build_info": {"cuda_backend": False, "system_info": "CPU-only"},
        "cuda_device_receipt": {"devices": []},
        "n_gpu_layers_requested": -1,
        "n_gpu_layers_offloaded": 0,
        "gpu_memory_before_mb": 0,
        "gpu_memory_peak_mb": 0,
        "gpu_memory_after_mb": 0,
        "cuda_offload_authenticated": False,
        "offload_log_excerpt": "",
        "rows": [
            {
                "row_id": row["row_id"],
                "prompt": row["prompt"],
                "raw_text": f"ANSWER: {mod.expected_answer_text(row)}",
                "finish_reason": "stop",
                "token_counts": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                "timing": {"load_s": 0.0, "generation_s": 0.01},
                "seed": 5708,
                "generation_config": {},
                "model_hash": "",
                "telemetry": {},
            }
            for row in panel
        ],
    }


def _run_fixture(tmp_path: Path, runner: mod.GenerationRunner = _runner) -> dict[str, Any]:
    return mod.run(
        result_path=tmp_path / mod.RESULT_RELATIVE_PATH.name,
        row_manifest_path=tmp_path / mod.ROW_MANIFEST_RELATIVE_PATH.name,
        resolved_model_path=_fake_model_path(tmp_path),
        generation_runner=runner,
        tests_added_or_reused=TESTS_ADDED_OR_REUSED,
        write=True,
    )


def test_req_verify_5708_spec_declares_raw_canary_contract() -> None:
    """REQ-VERIFY-5708: OpenSpec anchors raw CUDA canary fields and gates."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("### REQ-VERIFY-5708") : spec.index("### REQ-VERIFY-5615")]
    normalized = " ".join(section.split())

    for marker in (
        "REQ-VERIFY-5708",
        "SCENARIO-VERIFY-5708",
        str(mod.RESULT_RELATIVE_PATH),
        mod.INFERENCE_SUBSTRATE,
        mod.MODEL_REPO_ID,
        "without JSON grammar",
        "TrapQA-style",
        "`cuda_offload_authenticated_score=0.0`",
        "`sota_canary_ready_score=1.0`",
    ):
        assert marker in section
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert f"`{field}`" in section
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert " ".join(principle.split()) in normalized


def test_scenario_verify_5708_complete_artifact_and_manifest(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-5708: raw responses seal when all gates pass."""

    artifact = _run_fixture(tmp_path)
    manifest_rows = mod.read_manifest_rows(tmp_path / mod.ROW_MANIFEST_RELATIVE_PATH.name)

    assert mod.validate_artifact(artifact) is True
    assert mod.verify_manifest_rows(manifest_rows, artifact) is True
    assert mod.verify_commitments(artifact, manifest_rows) is True
    assert artifact["MODEL_SPECS"][0]["headline"] == mod.MODEL_HEADLINE
    assert artifact["model_repo_id"] == mod.MODEL_REPO_ID
    assert artifact["gguf_filename"].endswith("Q4_K_M.gguf")
    assert artifact["quantization"] == "Q4_K_M"
    assert artifact["headline_model_count"] == 1
    assert artifact["cuda_offload_authenticated"] is True
    assert artifact["cuda_offload_authenticated_score"] == 1.0
    assert artifact["sota_canary_ready_score"] == 1.0
    assert artifact["missing_row_count"] == 0
    assert artifact["parse_failure_count"] == 0
    assert artifact["validator_disagreement_count"] == 0
    assert artifact["legacy_smoke_only"] is True
    assert artifact["native_json_grammar_used"] is False
    assert artifact["external_scorer_used"] is False
    assert artifact["honest_verdict"].startswith("complete:")
    assert len(manifest_rows) == len(artifact["preregistered_panel"])
    assert set(artifact["family_counts"]) == set(mod.REQUIRED_FAMILIES)
    assert all(count == 10 for count in artifact["family_counts"].values())
    assert len(artifact["raw_response_hashes"]) == 50

    loaded = json.loads((tmp_path / mod.RESULT_RELATIVE_PATH.name).read_text(encoding="utf-8"))
    assert loaded == artifact
    assert artifact["tests_added_or_reused"] == TESTS_ADDED_OR_REUSED


def test_req_verify_5708_panel_contains_required_families_and_controls() -> None:
    """REQ-VERIFY-5708: panel is frozen before outcomes and covers controls."""

    panel = mod.freeze_preregistered_panel()
    families = mod.family_counts(panel)

    assert len(panel) == 50
    assert all(families[family] == 10 for family in mod.REQUIRED_FAMILIES)
    assert any(row["expected"]["kind"] == "abstain" for row in panel)
    assert any(row["control_tags"]["contradiction"] for row in panel)
    assert any(row["control_tags"]["shift"] for row in panel)
    assert any(row["family"] == "trapqa_shortcut" for row in panel)

    row = next(row for row in panel if row["family"] == "hard_soft_preference")
    answer = mod.expected_answer_text(row)
    parsed = mod.parse_raw_answer(f"ANSWER: {answer}")
    assert mod.primary_validate_row(row, parsed)["parse_ok"] is True
    assert mod.secondary_validate_row(row, parsed)["validator_version"].endswith("v1")
    assert mod.primary_validate_row(row, {"parse_ok": False, "answer": ""})["parse_ok"] is False


def test_req_verify_5708_cpu_only_or_missing_offload_blocks(tmp_path: Path) -> None:
    """REQ-VERIFY-5708: CPU-only evidence cannot unlock the headline gate."""

    artifact = _run_fixture(tmp_path, runner=_blocked_runner)

    assert artifact["cuda_offload_authenticated"] is False
    assert artifact["cuda_offload_authenticated_score"] == 0.0
    assert artifact["n_gpu_layers_offloaded"] == 0
    assert artifact["sota_canary_ready_score"] == 0.0
    assert artifact["honest_verdict"].startswith("blocked:")
    assert mod.validate_artifact(artifact) is True


def test_req_verify_5708_parse_failure_and_validator_disagreement_block(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-VERIFY-5708: unparsable rows or validator disagreements fail closed."""

    def parse_failure_runner(
        model_spec: dict[str, Any],
        panel: list[dict[str, Any]],
        generation_config: dict[str, Any],
        random_seeds: dict[str, int],
    ) -> dict[str, Any]:
        receipt = _runner(model_spec, panel, generation_config, random_seeds)
        receipt["rows"][0]["raw_text"] = "I cannot comply without a final answer."
        return receipt

    parse_failure = _run_fixture(tmp_path / "parse", runner=parse_failure_runner)
    assert parse_failure["parse_failure_count"] == 1
    assert parse_failure["sota_canary_ready_score"] == 0.0
    assert parse_failure["honest_verdict"].startswith("blocked:")
    assert mod.validate_artifact(parse_failure) is True

    original_secondary = mod.secondary_validate_row

    def disagree(row: dict[str, Any], parsed: dict[str, Any]) -> dict[str, Any]:
        result = original_secondary(row, parsed)
        if row["row_id"] == "efs-00":
            result = dict(result)
            result["label"] = not result["label"]
        return result

    monkeypatch.setattr(mod, "secondary_validate_row", disagree)
    disagreement = _run_fixture(tmp_path / "disagree")
    assert disagreement["validator_disagreement_count"] == 1
    assert disagreement["sota_canary_ready_score"] == 0.0
    assert disagreement["honest_verdict"].startswith("blocked:")
    assert mod.validate_artifact(disagreement) is True


def test_req_verify_5708_manifest_and_artifact_validation_fail_closed(tmp_path: Path) -> None:
    """REQ-VERIFY-5708: tampered manifests and unsupported claims are rejected."""

    artifact = _run_fixture(tmp_path)
    manifest_rows = mod.read_manifest_rows(tmp_path / mod.ROW_MANIFEST_RELATIVE_PATH.name)

    tampered_rows = deepcopy(manifest_rows)
    tampered_rows[0]["raw_text"] = "ANSWER: TAMPERED"
    with pytest.raises(ValueError, match="raw_response_hash"):
        mod.verify_manifest_rows(tampered_rows, artifact)

    tampered_rows = deepcopy(manifest_rows)
    tampered_rows[0]["previous_row_hash"] = "sha256:bad"
    with pytest.raises(ValueError, match="previous_row_hash"):
        mod.verify_manifest_rows(tampered_rows, artifact)

    bad = deepcopy(artifact)
    bad["native_json_grammar_used"] = True
    bad["sota_canary_ready_score"] = mod.sota_canary_ready_score(bad)
    bad["honest_verdict"] = mod.honest_verdict(bad)
    bad["reproducibility_checksum"] = mod.reproducibility_checksum(bad)
    with pytest.raises(ValueError, match="native_json_grammar_used"):
        mod.validate_artifact(bad)

    bad = deepcopy(artifact)
    bad["reproducibility_checksum"] = "sha256:bad"
    with pytest.raises(ValueError, match="reproducibility_checksum"):
        mod.validate_artifact(bad)

    bad = deepcopy(artifact)
    bad["field_principles"] = {}
    with pytest.raises(ValueError, match="field_principles"):
        mod.validate_artifact(bad)


def test_req_verify_5708_edge_branches_are_fail_closed(tmp_path: Path) -> None:
    """REQ-VERIFY-5708: helper and validation edge cases remain deterministic."""

    artifact = _run_fixture(tmp_path)
    manifest_rows = mod.read_manifest_rows(tmp_path / mod.ROW_MANIFEST_RELATIVE_PATH.name)

    assert mod.sha256_bytes(b"edge").startswith("sha256:")
    assert mod.parse_raw_answer("ANSWER:    ") == {
        "parse_ok": False,
        "answer": "",
        "error": "empty_answer",
    }

    bad_family = deepcopy(mod.freeze_preregistered_panel()[0])
    bad_family["family"] = "unknown"
    with pytest.raises(ValueError, match="unknown family"):
        mod.primary_validate_row(bad_family, {"parse_ok": True, "answer": "A"})

    missing = mod.run(
        result_path=tmp_path / "missing.json",
        row_manifest_path=tmp_path / "missing.jsonl",
        resolved_model_path=tmp_path / "not-present-Q4_K_M.gguf",
        generation_runner=_runner,
        write=False,
    )
    assert missing["missing_row_count"] == 50
    assert missing["parse_failure_count"] == 50
    assert "missing_rows" in missing["blocked_reasons"]

    def truncated_runner(
        model_spec: dict[str, Any],
        panel: list[dict[str, Any]],
        generation_config: dict[str, Any],
        random_seeds: dict[str, int],
    ) -> dict[str, Any]:
        receipt = _runner(model_spec, panel, generation_config, random_seeds)
        receipt["rows"][0]["finish_reason"] = "length"
        return receipt

    truncated = _run_fixture(tmp_path / "truncated", runner=truncated_runner)
    assert truncated["parse_failure_count"] == 1
    assert "parse_failures" in truncated["blocked_reasons"]

    expected_hash_mismatch = deepcopy(artifact)
    first_id = manifest_rows[0]["row_id"]
    expected_hash_mismatch["raw_response_hashes"][first_id] = "sha256:wrong"
    with pytest.raises(ValueError, match="raw_response_hash"):
        mod.verify_manifest_rows(manifest_rows, expected_hash_mismatch)

    row_hash_mismatch = deepcopy(manifest_rows)
    row_hash_mismatch[0]["row_hash"] = "sha256:wrong"
    with pytest.raises(ValueError, match="row_hash"):
        mod.verify_manifest_rows(row_hash_mismatch, artifact)

    bad_commit = deepcopy(artifact)
    bad_commit["shadow_prefix_hash"] = "sha256:wrong"
    with pytest.raises(ValueError, match="shadow_prefix_hash"):
        mod.verify_commitments(bad_commit, manifest_rows)

    bad_order = deepcopy(manifest_rows)
    bad_order[0], bad_order[1] = bad_order[1], bad_order[0]
    with pytest.raises(ValueError, match="manifest_order"):
        mod.verify_commitments(artifact, bad_order)

    bad_preimage = deepcopy(manifest_rows)
    bad_preimage[0]["pre_outcome_hash"] = "sha256:wrong"
    with pytest.raises(ValueError, match="pre_outcome_hash"):
        mod.verify_commitments(artifact, bad_preimage)

    reason_probe = deepcopy(artifact)
    reason_probe["commitments_verified"] = False
    reason_probe["external_scorer_used"] = True
    reason_probe["retired_runtime_used"] = True
    reasons = mod._blocked_reasons(reason_probe)
    assert "commitments_unverified" in reasons
    assert "external_scorer_used" in reasons
    assert "retired_runtime_used" in reasons

    validation_cases: list[tuple[str, dict[str, Any]]] = [("missing required fields", {})]
    for field, value, expected in (
        ("field_principles", [], "field_principles"),
        ("model_repo_id", "wrong/model", "model_repo_id"),
        ("inference_substrate", "cpu", "inference_substrate"),
        ("headline_model_count", 2, "headline_model_count"),
        ("legacy_smoke_only", False, "legacy_smoke_only"),
        ("external_scorer_used", True, "external_scorer_used"),
        ("cuda_offload_authenticated_score", 0.0, "cuda_offload_authenticated_score"),
        ("shadow_prefix_hash", "sha256:wrong", "shadow_prefix_hash"),
        ("sota_canary_ready_score", 0.0, "sota_canary_ready_score"),
    ):
        bad = deepcopy(artifact)
        bad[field] = value
        validation_cases.append((expected, bad))

    bad = deepcopy(artifact)
    bad["honest_verdict"] = "blocked: wrong"
    validation_cases.append(("honest_verdict", bad))

    bad = deepcopy(missing)
    bad["honest_verdict"] = "complete: wrong"
    validation_cases.append(("honest_verdict", bad))

    for expected, bad_artifact in validation_cases:
        with pytest.raises(ValueError, match=expected):
            mod.validate_artifact(bad_artifact)
