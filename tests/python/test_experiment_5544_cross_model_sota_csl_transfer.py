"""Tests for Exp5544 cross-model SOTA CSL transfer.

Spec refs: REQ-LEARN-5544,
SCENARIO-LEARN-5544-UPSTREAM-GATE,
SCENARIO-LEARN-5544-CROSS-FAMILY,
SCENARIO-LEARN-5544-NO-WEIGHT-MUTATION.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_5544_cross_model_sota_csl_transfer as exp


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec/capabilities/self-learning/spec.md"
TEST_COMMAND = (
    ".venv/bin/pytest "
    "tests/python/test_experiment_5544_cross_model_sota_csl_transfer.py "
    "-q --no-cov -n 0"
)
COVERAGE_COMMAND = (
    ".venv/bin/coverage run "
    "--include=python/carnot/experiment_5544_cross_model_sota_csl_transfer.py "
    "-m pytest tests/python/test_experiment_5544_cross_model_sota_csl_transfer.py "
    "-q --no-cov -n 0 && .venv/bin/coverage report "
    "--include=python/carnot/experiment_5544_cross_model_sota_csl_transfer.py "
    "--fail-under=100"
)
FULL_TEST_COMMAND = ".venv/bin/pytest tests/python -q"
TESTS_ADDED_OR_REUSED = [TEST_COMMAND, COVERAGE_COMMAND, FULL_TEST_COMMAND]


def _gguf_paths(tmp_path: Path) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for hf_id in exp.MANDATED_HF_IDS:
        path = tmp_path / f"{hf_id.split('/')[-1]}.gguf"
        path.write_bytes(f"GGUF fixture for {hf_id}".encode("utf-8"))
        paths[hf_id] = path
    return paths


def _resolver(paths: dict[str, Path]):
    def resolve(hf_id: str, _quantization: str) -> str | None:
        return str(paths[hf_id]) if hf_id in paths else None

    return resolve


def _cached_pair(paths: dict[str, Path]):
    return lambda: [
        {
            "name": "Qwen3.6-35B-A3B",
            "hf_id": exp.QWEN_HF_ID,
            "gpu": 0,
            "model_path": str(paths[exp.QWEN_HF_ID]),
        },
        {
            "name": "Gemma4-26B-A4B-it",
            "hf_id": exp.GEMMA_26_HF_ID,
            "gpu": 1,
            "model_path": str(paths[exp.GEMMA_26_HF_ID]),
        },
    ]


def _runtime_receipt(*, offload: bool = True) -> dict[str, Any]:
    return {
        "runtime_backend": "llama_cpp_python_cuda_gguf",
        "cuda_visible": offload,
        "llama_cpp_import_ok": True,
        "gpu_offload_supported": offload,
        "offload_evidence": offload,
        "n_gpu_layers": -1,
        "gpu_memory_delta_mb": 2048 if offload else 0,
        "blocked_preconditions": [] if offload else ["gpu_offload_evidence_missing"],
    }


def _fake_generation(**kwargs: Any) -> dict[str, Any]:
    task = kwargs["task"]
    stage = kwargs["stage"]
    arm = kwargs.get("arm")
    if stage == "source_attempt":
        answer = task["expected_answer"]
    elif arm in {exp.SAME_FAMILY_ARM, exp.CROSS_FAMILY_ARM}:
        answer = task["expected_answer"]
    elif arm == exp.SHUFFLED_ARM:
        answer = task["decoy_answer"]
    else:
        answer = "unknown"
    return {
        "output_text": f"Final answer: {answer}",
        "prompt_token_count": len(str(kwargs["prompt"]).split()),
        "generated_token_count": 4,
        "duration_s": 0.01,
        "backend_details": {"fake_generation_runner": True},
    }


def _complete_artifact(tmp_path: Path) -> dict[str, Any]:
    paths = _gguf_paths(tmp_path)
    return exp.run(
        root=REPO,
        result_path=tmp_path / exp.RESULT_RELATIVE_PATH.name,
        tests_added_or_reused=TESTS_ADDED_OR_REUSED,
        model_resolver=_resolver(paths),
        cached_pair_fn=_cached_pair(paths),
        runtime_probe=lambda **_kwargs: _runtime_receipt(),
        generation_runner=_fake_generation,
        write=True,
    )


def test_req_learn_5544_spec_declares_cross_model_transfer_contract() -> None:
    """REQ-LEARN-5544: OpenSpec anchors the cross-model transfer artifact."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("## REQ-LEARN-5544") :]

    for marker in (
        "REQ-LEARN-5544",
        "SCENARIO-LEARN-5544-UPSTREAM-GATE",
        "SCENARIO-LEARN-5544-CROSS-FAMILY",
        "SCENARIO-LEARN-5544-NO-WEIGHT-MUTATION",
        str(exp.RESULT_RELATIVE_PATH),
        str(exp.UPSTREAM_FIVE_ARM_PATH),
        exp.INFERENCE_SUBSTRATE,
        "cached_sota_pair()",
        "AutoTokenizer.from_pretrained",
    ):
        assert marker in section
    for hf_id in exp.MANDATED_HF_IDS:
        assert hf_id in section
    for field in exp.REQUIRED_ARTIFACT_FIELDS:
        assert f"`{field}`" in section


def test_scenario_learn_5544_upstream_gate_and_model_family_selection(
    tmp_path: Path,
) -> None:
    """SCENARIO-LEARN-5544-UPSTREAM-GATE: parse gates and select families."""

    paths = _gguf_paths(tmp_path)
    model_specs, cache_receipt = exp.resolve_model_specs(
        model_resolver=_resolver(paths),
        cached_pair_fn=_cached_pair(paths),
    )
    roles = exp.select_transfer_roles(model_specs)
    upstream = exp.load_upstream_five_arm(REPO)

    assert upstream["loadable"] is True
    assert upstream["csl_five_arm_ready"] is True
    assert cache_receipt["cached_sota_pair_attempted"] is True
    assert cache_receipt["cached_sota_pair_available"] is True
    assert {spec["hf_id"] for spec in model_specs} == set(exp.MANDATED_HF_IDS)
    assert roles["cross_source"]["hf_id"] == exp.QWEN_HF_ID
    assert roles["same_source"]["hf_id"] == exp.GEMMA_26_HF_ID
    assert roles["target"]["hf_id"] == exp.GEMMA_31_HF_ID
    for spec in model_specs:
        assert spec["model_path"].endswith(".gguf")
        assert spec["legacy_smoke_only"] is False
        assert spec["file_receipt"]["exists"] is True


def test_scenario_learn_5544_cross_family_scores_and_writes_artifact(
    tmp_path: Path,
) -> None:
    """SCENARIO-LEARN-5544-CROSS-FAMILY: cross memory beats controls."""

    artifact = _complete_artifact(tmp_path)
    written = json.loads((tmp_path / exp.RESULT_RELATIVE_PATH.name).read_text(encoding="utf-8"))

    assert written == artifact
    assert exp.validate_artifact(artifact) is True
    for field in exp.REQUIRED_ARTIFACT_FIELDS:
        assert field in artifact
        assert artifact["field_principles"][field]

    assert artifact["upstream_gate_evidence"]["csl_five_arm_ready"] is True
    assert artifact["source_models"] == [exp.QWEN_HF_ID, exp.GEMMA_26_HF_ID]
    assert artifact["target_models"] == [exp.GEMMA_31_HF_ID]
    assert artifact["no_memory_score"] == pytest.approx(0.0)
    assert artifact["shuffled_memory_score"] == pytest.approx(0.0)
    assert artifact["same_family_memory_score"] == pytest.approx(1.0)
    assert artifact["cross_family_memory_score"] == pytest.approx(1.0)
    assert artifact["cross_family_delta_over_shuffled"] == pytest.approx(1.0)
    assert artifact["heldout_delta"] == pytest.approx(1.0)
    assert artifact["stale_evidence_rejection_rate"] == pytest.approx(1.0)
    assert artifact["negative_transfer_rate"] == pytest.approx(0.0)
    assert artifact["no_weight_mutation"] is True
    assert artifact["gpu_offload_evidence"]["offload_evidence"] is True
    assert artifact["csl_claim_allowed"] is True
    assert artifact["inference_substrate"] == exp.INFERENCE_SUBSTRATE
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["tests_added_or_reused"] == TESTS_ADDED_OR_REUSED

    label_sets = {
        arm: tuple(row["query_id"] for row in rows)
        for arm, rows in artifact["target_evaluations"].items()
    }
    assert set(label_sets) == set(exp.TARGET_ARMS)
    assert len(set(label_sets.values())) == 1
    assert len(artifact["source_attempts"]) == 2 * len(exp.build_fixture()["heldout_tasks"])
    for row in artifact["source_attempts"] + artifact["row_results"]:
        assert row["prompt_hash"].startswith("sha256:")
        assert row["output_hash"].startswith("sha256:")
        assert row["row_checksum"] == exp.row_checksum(row)


def test_scenario_learn_5544_no_weight_mutation_and_validation_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SCENARIO-LEARN-5544-NO-WEIGHT-MUTATION: receipts fail closed."""

    artifact = _complete_artifact(tmp_path)
    assert artifact["weight_mutation_evidence"]["no_weight_mutation"] is True
    assert artifact["weight_mutation_evidence"]["changed_model_files"] == []

    drift_cases = [
        ("cross_family_delta_over_shuffled", 0.0, "cross_family_delta_over_shuffled"),
        ("heldout_delta", 0.0, "heldout_delta"),
        ("stale_evidence_rejection_rate", 0.0, "stale_evidence_rejection_rate"),
        ("negative_transfer_rate", 1.0, "negative_transfer_rate"),
        ("no_weight_mutation", False, "no_weight_mutation"),
        ("csl_claim_allowed", False, "csl_claim_allowed"),
        ("inference_substrate", "aggregation_from_upstream_artifacts", "inference_substrate"),
        ("honest_verdict", "ready", "honest_verdict"),
    ]
    for field, value, expected in drift_cases:
        bad = deepcopy(artifact)
        bad[field] = value
        bad["reproducibility_checksum"] = exp.reproducibility_checksum(bad)
        with pytest.raises(ValueError, match=expected):
            exp.validate_artifact(bad)

    bad_gpu = deepcopy(artifact)
    bad_gpu["gpu_offload_evidence"]["offload_evidence"] = False
    bad_gpu["reproducibility_checksum"] = exp.reproducibility_checksum(bad_gpu)
    with pytest.raises(ValueError, match="gpu_offload_evidence"):
        exp.validate_artifact(bad_gpu)

    bad_weight = deepcopy(artifact)
    target_id = artifact["target_models"][0]
    bad_weight["weight_mutation_evidence"]["after_receipts"][target_id]["mtime_ns"] += 1
    bad_weight["weight_mutation_evidence"]["changed_model_files"] = [target_id]
    bad_weight["reproducibility_checksum"] = exp.reproducibility_checksum(bad_weight)
    with pytest.raises(ValueError, match="weight_mutation_evidence"):
        exp.validate_artifact(bad_weight)

    bad_row_checksum = deepcopy(artifact)
    bad_row_checksum["row_results"][0]["row_checksum"] = "sha256:bad"
    bad_row_checksum["reproducibility_checksum"] = exp.reproducibility_checksum(
        bad_row_checksum
    )
    with pytest.raises(ValueError, match="row_checksum"):
        exp.validate_artifact(bad_row_checksum)

    divergent_queries = deepcopy(artifact)
    divergent_queries["target_evaluations"][exp.SHUFFLED_ARM] = divergent_queries[
        "target_evaluations"
    ][exp.SHUFFLED_ARM][1:]
    divergent_queries["reproducibility_checksum"] = exp.reproducibility_checksum(
        divergent_queries
    )
    with pytest.raises(ValueError, match="same_heldout_query_set"):
        exp.validate_artifact(divergent_queries)

    missing_principle = deepcopy(artifact)
    missing_principle["field_principles"].pop("cross_family_memory_score")
    missing_principle["reproducibility_checksum"] = exp.reproducibility_checksum(
        missing_principle
    )
    with pytest.raises(ValueError, match="field_principles"):
        exp.validate_artifact(missing_principle)

    no_tests = deepcopy(artifact)
    no_tests["tests_added_or_reused"] = []
    no_tests["reproducibility_checksum"] = exp.reproducibility_checksum(no_tests)
    with pytest.raises(ValueError, match="tests_added_or_reused"):
        exp.validate_artifact(no_tests)

    missing = deepcopy(artifact)
    missing.pop("cross_family_memory_score")
    missing["reproducibility_checksum"] = exp.reproducibility_checksum(missing)
    with pytest.raises(ValueError, match="missing required fields"):
        exp.validate_artifact(missing)

    blocked = deepcopy(artifact)
    blocked["upstream_gate_evidence"]["csl_five_arm_ready"] = False
    blocked["csl_claim_allowed"] = False
    blocked["honest_verdict"] = exp.honest_verdict(blocked)
    blocked["reproducibility_checksum"] = exp.reproducibility_checksum(blocked)
    assert blocked["honest_verdict"].startswith("blocked:")
    assert exp.validate_artifact(blocked) is True

    assert exp.score_rows([]) == 0.0
    assert exp.extract_answer("none", ["a", "b"]) is None
    assert exp.select_transfer_roles([]) == {
        "cross_source": None,
        "same_source": None,
        "target": None,
    }
    blocked_preconditions = exp.evaluate_preconditions(
        upstream_gate={"csl_five_arm_ready": False},
        model_specs=[],
        roles={"cross_source": None, "same_source": None, "target": None},
        runtime_receipt={
            "runtime_backend": "unavailable",
            "cuda_visible": False,
            "offload_evidence": False,
            "blocked_preconditions": ["probe_failed"],
        },
    )
    assert blocked_preconditions["all_passed"] is False
    assert blocked_preconditions["blocked_preconditions"] == [
        "cuda_not_visible",
        "gpu_offload_evidence_missing",
        "llama_cpp_gguf_runtime_missing",
        "mandated_model_specs_missing",
        "probe_failed",
        "required_cross_model_family_roles_unavailable",
        "upstream_csl_five_arm_not_ready",
    ]
    assert exp.memory_for_arm(artifact["memory_entries"], exp.CROSS_FAMILY_ARM, "missing") is None
    assert exp.same_heldout_query_set(None) is False
    assert exp.weight_evidence_consistent(None) is False
    assert exp.model_file_receipt(None) == {"exists": False, "path": None}
    assert exp.model_file_receipt(tmp_path / "missing.gguf") == {
        "exists": False,
        "path": str(tmp_path / "missing.gguf"),
    }

    bad_seed = deepcopy(artifact)
    bad_seed["random_seed"] = 1
    bad_seed["reproducibility_checksum"] = exp.reproducibility_checksum(bad_seed)
    with pytest.raises(ValueError, match="random_seed"):
        exp.validate_artifact(bad_seed)

    bad_checksum = deepcopy(artifact)
    bad_checksum["reproducibility_checksum"] = "sha256:bad"
    with pytest.raises(ValueError, match="reproducibility_checksum"):
        exp.validate_artifact(bad_checksum)

    def raise_oserror(*_args: Any, **_kwargs: Any) -> None:
        raise OSError("nvidia-smi unavailable")

    monkeypatch.setattr(exp.subprocess, "run", raise_oserror)
    assert exp.total_gpu_memory_used_mb() == 0
