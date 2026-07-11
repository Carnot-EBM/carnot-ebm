"""Tests for Exp5559 causal cross-model SOTA CSL transfer v2.

Spec refs: REQ-LEARN-5559,
SCENARIO-LEARN-5559-UPSTREAM-GATE,
SCENARIO-LEARN-5559-CROSS-FAMILY,
SCENARIO-LEARN-5559-STALE-AND-NEGATIVE-GATES,
SCENARIO-LEARN-5559-NO-WEIGHT-MUTATION,
SCENARIO-LEARN-5559-ARTIFACT.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_5559_cross_model_sota_csl_transfer_v2 as exp


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec/capabilities/self-learning/spec.md"
RESULT_PATH = REPO / exp.RESULT_RELATIVE_PATH
TEST_COMMAND = (
    ".venv/bin/pytest "
    "tests/python/test_experiment_5559_cross_model_sota_csl_transfer_v2.py "
    "-q --no-cov -n 0"
)
COVERAGE_COMMAND = (
    ".venv/bin/coverage run "
    "--include=python/carnot/experiment_5559_cross_model_sota_csl_transfer_v2.py "
    "-m pytest tests/python/test_experiment_5559_cross_model_sota_csl_transfer_v2.py "
    "-q --no-cov -n 0 && .venv/bin/coverage report "
    "--include=python/carnot/experiment_5559_cross_model_sota_csl_transfer_v2.py "
    "--fail-under=100"
)
FULL_TEST_COMMAND = ".venv/bin/pytest tests/python -q"
TESTS_ADDED_OR_REUSED = [TEST_COMMAND, COVERAGE_COMMAND, FULL_TEST_COMMAND]


def _gguf_paths(tmp_path: Path) -> dict[str, Path]:
    tmp_path.mkdir(parents=True, exist_ok=True)
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
            "name": "Gemma4-31B-it",
            "hf_id": exp.GEMMA_31_HF_ID,
            "gpu": 1,
            "model_path": str(paths[exp.GEMMA_31_HF_ID]),
        },
    ]


def _runtime_receipt(*, offload: bool = True) -> dict[str, Any]:
    return {
        "runtime_backend": "llama_cpp_python_cuda_gguf" if offload else "unavailable",
        "cuda_visible": offload,
        "llama_cpp_import_ok": offload,
        "gpu_offload_supported": offload,
        "offload_evidence": offload,
        "n_gpu_layers": -1,
        "gpu_memory_delta_mb": 2048 if offload else 0,
        "blocked_preconditions": [] if offload else ["gpu_offload_evidence_missing"],
    }


def _fake_generation(**kwargs: Any) -> dict[str, Any]:
    stage = kwargs["stage"]
    if stage == "source_attempt":
        answer = kwargs["candidate_memory"]["selected_action"]
    else:
        arm = kwargs["arm"]
        decision = kwargs["decision"]
        memory = kwargs.get("memory_entry")
        if arm == exp.ALIGNED_CAUSAL_MEMORY_ARM and memory is not None:
            answer = decision["expected_action"]
        elif arm in {exp.NO_MEMORY_ARM, exp.ALIGNED_CAUSAL_MEMORY_ARM}:
            answer = decision["baseline_action"]
        else:
            answer = "unknown"
    return {
        "output_text": f"Final answer: {answer}",
        "prompt_token_count": len(str(kwargs["prompt"]).split()),
        "generated_token_count": 4,
        "duration_s": 0.01,
        "backend_details": {"fake_generation_runner": True},
    }


def _bad_negative_generation(**kwargs: Any) -> dict[str, Any]:
    if kwargs["stage"] == "target_evaluation" and kwargs["arm"] == exp.ALIGNED_CAUSAL_MEMORY_ARM:
        answer = "unknown"
        return {"output_text": answer, "backend_details": {"bad_negative": True}}
    return _fake_generation(**kwargs)


def _bad_shuffled_generation(**kwargs: Any) -> dict[str, Any]:
    if kwargs["stage"] == "target_evaluation" and kwargs["arm"] == exp.SHUFFLED_MEMORY_ARM:
        answer = kwargs["decision"]["expected_action"]
        return {"output_text": answer, "backend_details": {"bad_shuffled": True}}
    return _fake_generation(**kwargs)


def _complete_artifact(tmp_path: Path, generation_runner=_fake_generation) -> dict[str, Any]:
    paths = _gguf_paths(tmp_path)
    return exp.run(
        root=REPO,
        result_path=tmp_path / exp.RESULT_RELATIVE_PATH.name,
        tests_added_or_reused=TESTS_ADDED_OR_REUSED,
        model_resolver=_resolver(paths),
        cached_pair_fn=_cached_pair(paths),
        runtime_probe=lambda **_kwargs: _runtime_receipt(),
        generation_runner=generation_runner,
        write=True,
    )


def test_req_learn_5559_spec_declares_causal_cross_model_transfer_contract() -> None:
    """REQ-LEARN-5559: OpenSpec anchors the causal transfer v2 receipt."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("## REQ-LEARN-5559") :]

    for marker in (
        "REQ-LEARN-5559",
        "SCENARIO-LEARN-5559-UPSTREAM-GATE",
        "SCENARIO-LEARN-5559-CROSS-FAMILY",
        "SCENARIO-LEARN-5559-STALE-AND-NEGATIVE-GATES",
        "SCENARIO-LEARN-5559-NO-WEIGHT-MUTATION",
        "SCENARIO-LEARN-5559-ARTIFACT",
        str(exp.RESULT_RELATIVE_PATH),
        str(exp.UPSTREAM_CAUSAL_MEMORY_PATH),
        exp.INFERENCE_SUBSTRATE,
        "cached_sota_pair()",
        "AutoTokenizer.from_pretrained",
    ):
        assert marker in section
    for hf_id in exp.MANDATED_HF_IDS:
        assert hf_id in section
    for field in exp.REQUIRED_ARTIFACT_FIELDS:
        assert f"`{field}`" in section
        assert exp.FIELD_PRINCIPLES[field]


def test_scenario_learn_5559_cross_family_scores_and_writes_artifact(tmp_path: Path) -> None:
    """SCENARIO-LEARN-5559-CROSS-FAMILY: aligned causal memory beats controls."""

    artifact = _complete_artifact(tmp_path)
    written = json.loads((tmp_path / exp.RESULT_RELATIVE_PATH.name).read_text(encoding="utf-8"))

    assert written == artifact
    assert exp.validate_artifact(artifact) is True
    for field in exp.REQUIRED_ARTIFACT_FIELDS:
        assert field in artifact
        assert artifact["field_principles"][field]

    assert artifact["upstream_causal_status"]["csl_claim_allowed"] is True
    assert artifact["live_model_invoked"] is True
    assert artifact["source_models"] == [exp.QWEN_HF_ID]
    assert artifact["target_models"] == [exp.GEMMA_31_HF_ID]
    assert artifact["unavailable_models"] == []
    assert artifact["no_memory_score"] == pytest.approx(0.1666666667)
    assert artifact["shuffled_memory_score"] == pytest.approx(0.0)
    assert artifact["stale_memory_score"] == pytest.approx(0.0)
    assert artifact["aligned_memory_score"] == pytest.approx(1.0)
    assert artifact["cross_family_delta_over_shuffled"] == pytest.approx(1.0)
    assert artifact["stale_evidence_rejection_rate"] == pytest.approx(1.0)
    assert artifact["negative_transfer_rate"] == pytest.approx(0.0)
    assert artifact["no_weight_mutation"] is True
    assert artifact["gpu_offload_evidence"]["offload_evidence"] is True
    assert artifact["csl_claim_allowed"] is True
    assert artifact["inference_substrate"] == exp.INFERENCE_SUBSTRATE
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["tests_added_or_reused"] == TESTS_ADDED_OR_REUSED

    decision_sets = {
        arm: tuple(row["decision_id"] for row in rows)
        for arm, rows in artifact["target_evaluations"].items()
    }
    assert set(decision_sets) == set(exp.TARGET_ARMS)
    assert len(set(decision_sets.values())) == 1
    assert len(artifact["source_attempts"]) == len(exp.candidate_memory_rows(artifact["fixture"]))
    for row in artifact["source_attempts"] + artifact["row_results"]:
        assert row["prompt_hash"].startswith("sha256:")
        assert row["output_hash"].startswith("sha256:")
        assert row["row_checksum"] == exp.row_checksum(row)


def test_scenario_learn_5559_stale_and_negative_gates_fail_closed(tmp_path: Path) -> None:
    """SCENARIO-LEARN-5559-STALE-AND-NEGATIVE-GATES: unsafe transfer is blocked."""

    negative = _complete_artifact(tmp_path / "negative", _bad_negative_generation)
    assert negative["negative_transfer_rate"] > 0.0
    assert negative["csl_claim_allowed"] is False
    assert negative["honest_verdict"].startswith("blocked:")
    assert exp.validate_artifact(negative) is True

    shuffled = _complete_artifact(tmp_path / "shuffled", _bad_shuffled_generation)
    assert shuffled["cross_family_delta_over_shuffled"] == pytest.approx(0.0)
    assert shuffled["csl_claim_allowed"] is False
    assert exp.validate_artifact(shuffled) is True


def test_scenario_learn_5559_no_weight_mutation_and_validation_drift(tmp_path: Path) -> None:
    """SCENARIO-LEARN-5559-NO-WEIGHT-MUTATION: receipts and gates validate."""

    artifact = _complete_artifact(tmp_path)
    assert artifact["weight_mutation_evidence"]["no_weight_mutation"] is True
    assert artifact["weight_mutation_evidence"]["changed_model_files"] == []

    drift_cases = [
        ("upstream_causal_memory", "results/wrong.json", "upstream_causal_memory"),
        ("random_seed", 0, "random_seed"),
        ("cross_family_delta_over_shuffled", 0.0, "cross_family_delta_over_shuffled"),
        ("negative_transfer_rate", 1.0, "negative_transfer_rate"),
        ("stale_evidence_rejection_rate", 0.0, "stale_evidence_rejection_rate"),
        ("no_weight_mutation", False, "no_weight_mutation"),
        ("csl_claim_allowed", False, "csl_claim_allowed"),
        ("inference_substrate", "aggregation_from_upstream_artifacts", "inference_substrate"),
        ("honest_verdict", "ready", "honest_verdict"),
        ("live_model_invoked", False, "live_model_invoked"),
    ]
    for field, value, expected in drift_cases:
        bad = deepcopy(artifact)
        bad[field] = value
        bad["reproducibility_checksum"] = exp.reproducibility_checksum(bad)
        with pytest.raises(ValueError, match=expected):
            exp.validate_artifact(bad)

    bad_weight = deepcopy(artifact)
    target_id = artifact["target_models"][0]
    bad_weight["weight_mutation_evidence"]["after_receipts"][target_id]["mtime_ns"] += 1
    bad_weight["weight_mutation_evidence"]["changed_model_files"] = [target_id]
    bad_weight["reproducibility_checksum"] = exp.reproducibility_checksum(bad_weight)
    with pytest.raises(ValueError, match="weight_mutation_evidence"):
        exp.validate_artifact(bad_weight)

    bad_gpu_claim = deepcopy(artifact)
    bad_gpu_claim["gpu_offload_evidence"]["offload_evidence"] = False
    bad_gpu_claim["reproducibility_checksum"] = exp.reproducibility_checksum(bad_gpu_claim)
    with pytest.raises(ValueError, match="gpu_offload_evidence"):
        exp.validate_artifact(bad_gpu_claim)

    bad_gpu_shape = deepcopy(artifact)
    bad_gpu_shape["gpu_offload_evidence"] = "not-a-mapping"
    bad_gpu_shape["reproducibility_checksum"] = exp.reproducibility_checksum(bad_gpu_shape)
    with pytest.raises(ValueError, match="gpu_offload_evidence"):
        exp.validate_artifact(bad_gpu_shape)

    bad_models = deepcopy(artifact)
    bad_models["model_specs"] = bad_models["model_specs"][1:]
    bad_models["reproducibility_checksum"] = exp.reproducibility_checksum(bad_models)
    with pytest.raises(ValueError, match="model_specs"):
        exp.validate_artifact(bad_models)

    bad_row_checksum = deepcopy(artifact)
    bad_row_checksum["row_results"][0]["row_checksum"] = "sha256:bad"
    bad_row_checksum["reproducibility_checksum"] = exp.reproducibility_checksum(bad_row_checksum)
    with pytest.raises(ValueError, match="row_checksum"):
        exp.validate_artifact(bad_row_checksum)

    divergent = deepcopy(artifact)
    divergent["target_evaluations"][exp.STALE_MEMORY_ARM] = divergent["target_evaluations"][
        exp.STALE_MEMORY_ARM
    ][1:]
    divergent["reproducibility_checksum"] = exp.reproducibility_checksum(divergent)
    with pytest.raises(ValueError, match="same_decision_set"):
        exp.validate_artifact(divergent)

    missing_principle = deepcopy(artifact)
    missing_principle["field_principles"].pop("aligned_memory_score")
    missing_principle["reproducibility_checksum"] = exp.reproducibility_checksum(
        missing_principle
    )
    with pytest.raises(ValueError, match="field_principles"):
        exp.validate_artifact(missing_principle)

    missing = deepcopy(artifact)
    missing.pop("aligned_memory_score")
    missing["reproducibility_checksum"] = exp.reproducibility_checksum(missing)
    with pytest.raises(ValueError, match="missing required fields"):
        exp.validate_artifact(missing)

    bad_checksum = deepcopy(artifact)
    bad_checksum["reproducibility_checksum"] = "sha256:bad"
    with pytest.raises(ValueError, match="reproducibility_checksum"):
        exp.validate_artifact(bad_checksum)


def test_scenario_learn_5559_upstream_or_model_gate_skip_is_valid(tmp_path: Path) -> None:
    """SCENARIO-LEARN-5559-UPSTREAM-GATE: blocked preconditions skip live calls."""

    blocked = exp.run(
        root=tmp_path,
        result_path=tmp_path / exp.RESULT_RELATIVE_PATH.name,
        tests_added_or_reused=TESTS_ADDED_OR_REUSED,
        model_resolver=lambda _hf_id, _quantization: None,
        cached_pair_fn=lambda: None,
        runtime_probe=lambda **_kwargs: _runtime_receipt(offload=False),
        generation_runner=_fake_generation,
        write=True,
    )

    assert blocked["upstream_causal_status"]["loadable"] is False
    assert blocked["live_model_invoked"] is False
    assert blocked["csl_claim_allowed"] is False
    assert blocked["no_weight_mutation"] is True
    assert blocked["unavailable_models"] == list(exp.MANDATED_HF_IDS)
    assert blocked["honest_verdict"].startswith("blocked:")
    assert exp.validate_artifact(blocked) is True
    assert json.loads((tmp_path / exp.RESULT_RELATIVE_PATH.name).read_text(encoding="utf-8")) == blocked

    assert exp.load_upstream_causal_memory(tmp_path)["loadable"] is False
    assert exp.select_cross_family_roles([]) == {"source": None, "target": None}
    fallback_roles = exp.select_cross_family_roles(
        [
            {"hf_id": "local-alpha", "family": "alpha", "local_path_available": True},
            {"hf_id": "local-beta", "family": "beta", "local_path_available": True},
        ]
    )
    assert fallback_roles["source"]["hf_id"] == "local-alpha"
    assert fallback_roles["target"]["hf_id"] == "local-beta"
    same_family_roles = {
        "source": {"hf_id": "a", "family": "gemma"},
        "target": {"hf_id": "b", "family": "gemma"},
    }
    preconditions = exp.evaluate_preconditions(
        upstream={"csl_claim_allowed": True, "csl_memory_ready": True},
        fixture={"decisions": [{}], "active_entries": [{}]},
        model_specs=[],
        roles=same_family_roles,
        runtime_receipt={
            "runtime_backend": "llama_cpp_python_cuda_gguf",
            "cuda_visible": True,
            "offload_evidence": True,
            "blocked_preconditions": [],
        },
    )
    assert "mandated_model_specs_missing" in preconditions["blocked_preconditions"]
    assert "source_target_family_not_distinct" in preconditions["blocked_preconditions"]
    assert exp.shuffled_memory_entries([], []) == []
    assert exp.score_rows([]) == 0.0
    assert exp.safe_rate(0, 0) == 0.0
    assert exp.extract_answer("none", ["a", "b"]) is None
    assert exp.same_decision_set(None) is False


def test_req_learn_5559_repository_artifact_is_valid_when_present() -> None:
    """REQ-LEARN-5559-6: committed JSON remains a valid v2 transfer receipt."""

    if not RESULT_PATH.exists():
        pytest.skip("Exp5559 artifact has not been emitted yet")
    artifact = json.loads(RESULT_PATH.read_text(encoding="utf-8"))

    assert exp.validate_artifact(artifact) is True
    assert set(exp.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert artifact["upstream_causal_memory"] == str(exp.UPSTREAM_CAUSAL_MEMORY_PATH)
    assert artifact["inference_substrate"] == exp.INFERENCE_SUBSTRATE
