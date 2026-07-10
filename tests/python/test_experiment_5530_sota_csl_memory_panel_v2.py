"""Tests for Exp5530 SOTA GGUF CSL memory panel v2.

Spec refs: REQ-LEARN-5530,
SCENARIO-LEARN-5530-UPSTREAM-GATES,
SCENARIO-LEARN-5530-CONTROLS,
SCENARIO-LEARN-5530-NO-WEIGHT-MUTATION.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_5530_sota_csl_memory_panel_v2 as exp


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec/capabilities/self-learning/spec.md"
TEST_COMMAND = (
    ".venv/bin/pytest "
    "tests/python/test_experiment_5530_sota_csl_memory_panel_v2.py "
    "-q --no-cov -n 0"
)
COVERAGE_COMMAND = (
    ".venv/bin/coverage run "
    "--include=python/carnot/experiment_5530_sota_csl_memory_panel_v2.py "
    "-m pytest tests/python/test_experiment_5530_sota_csl_memory_panel_v2.py "
    "-q --no-cov -n 0 && .venv/bin/coverage report "
    "--include=python/carnot/experiment_5530_sota_csl_memory_panel_v2.py "
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
            "hf_id": "unsloth/Qwen3.6-35B-A3B-GGUF",
            "gpu": 0,
            "model_path": str(paths["unsloth/Qwen3.6-35B-A3B-GGUF"]),
        },
        {
            "name": "Gemma4-26B-A4B-it",
            "hf_id": "unsloth/gemma-4-26B-A4B-it-GGUF",
            "gpu": 1,
            "model_path": str(paths["unsloth/gemma-4-26B-A4B-it-GGUF"]),
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
        "gpu_memory_delta_mb": 1536 if offload else 0,
        "load_receipt": {
            "offload_evidence": offload,
            "gpu_memory_delta_mb": 1536 if offload else 0,
        },
        "blocked_preconditions": [] if offload else ["gpu_offload_evidence_missing"],
    }


def _fake_generation(**kwargs: Any) -> dict[str, Any]:
    task = kwargs["task"]
    condition = kwargs["condition"]
    if condition == exp.FRESH_MEMORY_CONDITION:
        answer = task["expected_answer"]
    elif condition == exp.STALE_MEMORY_CONDITION:
        answer = task["stale_answer"]
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


def test_req_learn_5530_spec_declares_sota_csl_panel_contract() -> None:
    """REQ-LEARN-5530: OpenSpec anchors the SOTA CSL memory panel v2."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("## REQ-LEARN-5530") :]

    for marker in (
        "REQ-LEARN-5530",
        "SCENARIO-LEARN-5530-UPSTREAM-GATES",
        "SCENARIO-LEARN-5530-CONTROLS",
        "SCENARIO-LEARN-5530-NO-WEIGHT-MUTATION",
        str(exp.RESULT_RELATIVE_PATH),
        exp.INFERENCE_SUBSTRATE,
        "cached_sota_pair()",
        "AutoTokenizer.from_pretrained",
    ):
        assert marker in section
    for hf_id in exp.MANDATED_HF_IDS:
        assert hf_id in section
    for field in exp.REQUIRED_ARTIFACT_FIELDS:
        assert f"`{field}`" in section


def test_scenario_learn_5530_upstream_gates_and_model_specs(tmp_path: Path) -> None:
    """SCENARIO-LEARN-5530-UPSTREAM-GATES: parse gates and GGUF receipts."""

    paths = _gguf_paths(tmp_path)
    model_specs, cache_receipt = exp.resolve_model_specs(
        model_resolver=_resolver(paths),
        cached_pair_fn=_cached_pair(paths),
    )
    upstream = exp.load_upstream_gates(REPO)

    assert upstream["all_required_gates_true"] is True
    assert upstream["exp5528"]["path"] == exp.UPSTREAM_5528_PATH.as_posix()
    assert upstream["exp5529"]["path"] == exp.UPSTREAM_5529_PATH.as_posix()
    assert cache_receipt["cached_sota_pair_attempted"] is True
    assert cache_receipt["cached_sota_pair_available"] is True
    assert {spec["hf_id"] for spec in model_specs} == set(exp.MANDATED_HF_IDS)
    assert exp.select_panel_model(model_specs)["hf_id"] == "unsloth/gemma-4-26B-A4B-it-GGUF"
    for spec in model_specs:
        assert spec["model_path"].endswith(".gguf")
        assert spec["local_path_available"] is True
        assert spec["legacy_smoke_only"] is False
        assert spec["file_receipt"]["exists"] is True


def test_scenario_learn_5530_controls_scores_and_writes_artifact(tmp_path: Path) -> None:
    """SCENARIO-LEARN-5530-CONTROLS: held-out controls use independent labels."""

    artifact = _complete_artifact(tmp_path)
    written = json.loads((tmp_path / exp.RESULT_RELATIVE_PATH.name).read_text(encoding="utf-8"))

    assert written == artifact
    assert exp.validate_artifact(artifact) is True
    for field in exp.REQUIRED_ARTIFACT_FIELDS:
        assert field in artifact
        assert artifact["field_principles"][field]
    assert artifact["model_specs"] and len(artifact["model_specs"]) == 3
    assert artifact["models_attempted"][0]["hf_id"] == "unsloth/gemma-4-26B-A4B-it-GGUF"
    assert artifact["no_memory_score"] == pytest.approx(0.0)
    assert artifact["fresh_memory_score"] == pytest.approx(1.0)
    assert artifact["stale_memory_score"] == pytest.approx(0.0)
    assert artifact["heldout_delta"] == pytest.approx(1.0)
    assert artifact["negative_transfer_rate"] == pytest.approx(0.0)
    assert artifact["stale_evidence_rejection_rate"] == pytest.approx(1.0)
    assert artifact["memory_hash_before"] != artifact["memory_hash_after"]
    assert artifact["no_model_weight_mutation"] is True
    assert artifact["gpu_offload_evidence"]["offload_evidence"] is True
    assert artifact["continuous_self_learning_evidence"] is True
    assert artifact["csl_claim_allowed"] is True
    assert artifact["inference_substrate"] == exp.INFERENCE_SUBSTRATE
    assert artifact["honest_verdict"].startswith("complete:")

    label_sets = {
        condition: tuple(row["label_id"] for row in rows)
        for condition, rows in artifact["condition_results"].items()
    }
    assert set(label_sets) == set(exp.CONDITIONS)
    assert len(set(label_sets.values())) == 1
    assert artifact["independent_label_source"] == exp.INDEPENDENT_LABEL_SOURCE
    assert artifact["metric_independence_clean"] is True
    assert artifact["utility_deltas"]["fresh_vs_no_memory"] == pytest.approx(1.0)
    assert artifact["cost_deltas"]["fresh_minus_no_memory_prompt_tokens"] > 0
    assert artifact["cost_deltas"]["fresh_minus_no_memory_verifier_units"] == 1.0
    for row in artifact["row_results"]:
        assert row["exact_verifier_witness"]["authority"] == "independent_label_table"
        assert row["final_authority_bypassed"] is False
        assert row["row_checksum"] == exp.row_checksum(row)


def test_scenario_learn_5530_no_weight_mutation_and_validation_drift(
    tmp_path: Path,
) -> None:
    """SCENARIO-LEARN-5530-NO-WEIGHT-MUTATION: receipts fail closed."""

    artifact = _complete_artifact(tmp_path)
    assert artifact["model_weight_receipt"]["model_file_receipt_before"] == artifact[
        "model_weight_receipt"
    ]["model_file_receipt_after"]
    assert exp.honest_verdict(artifact).startswith("complete:")

    drift_cases = [
        ("heldout_delta", 0.0, "heldout_delta"),
        ("negative_transfer_rate", 1.0, "negative_transfer_rate"),
        ("stale_evidence_rejection_rate", 0.0, "stale_evidence_rejection_rate"),
        ("memory_hash_after", artifact["memory_hash_before"], "memory_hash"),
        ("no_model_weight_mutation", False, "no_model_weight_mutation"),
        ("continuous_self_learning_evidence", False, "continuous_self_learning_evidence"),
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
    bad_weight["model_weight_receipt"]["model_file_receipt_after"]["mtime_ns"] += 1
    bad_weight["reproducibility_checksum"] = exp.reproducibility_checksum(bad_weight)
    with pytest.raises(ValueError, match="model_weight_receipt"):
        exp.validate_artifact(bad_weight)

    bad_row_checksum = deepcopy(artifact)
    bad_row_checksum["row_results"][0]["row_checksum"] = "sha256:bad"
    bad_row_checksum["reproducibility_checksum"] = exp.reproducibility_checksum(bad_row_checksum)
    with pytest.raises(ValueError, match="row_checksum"):
        exp.validate_artifact(bad_row_checksum)

    bad_row_type = deepcopy(artifact)
    bad_row_type["row_results"] = "not-a-list"
    bad_row_type["reproducibility_checksum"] = exp.reproducibility_checksum(bad_row_type)
    with pytest.raises(ValueError, match="row_results"):
        exp.validate_artifact(bad_row_type)

    no_attempted = deepcopy(artifact)
    no_attempted["models_attempted"] = []
    no_attempted["reproducibility_checksum"] = exp.reproducibility_checksum(no_attempted)
    with pytest.raises(ValueError, match="models_attempted"):
        exp.validate_artifact(no_attempted)

    missing = deepcopy(artifact)
    missing.pop("model_specs")
    missing["reproducibility_checksum"] = exp.reproducibility_checksum(missing)
    with pytest.raises(ValueError, match="missing required fields"):
        exp.validate_artifact(missing)

    missing_principle = deepcopy(artifact)
    missing_principle["field_principles"].pop("heldout_delta")
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

    bad_checksum = deepcopy(artifact)
    bad_checksum["reproducibility_checksum"] = "sha256:bad"
    with pytest.raises(ValueError, match="reproducibility_checksum"):
        exp.validate_artifact(bad_checksum)


def test_req_learn_5530_blocked_preconditions_do_not_call_generation(
    tmp_path: Path,
) -> None:
    """REQ-LEARN-5530-1/2/6/7: failed preconditions block before generation."""

    calls: list[str] = []
    artifact = exp.run(
        root=REPO,
        result_path=tmp_path / "blocked.json",
        tests_added_or_reused=TESTS_ADDED_OR_REUSED,
        model_resolver=lambda _hf_id, _quantization: None,
        cached_pair_fn=lambda: None,
        runtime_probe=lambda **_kwargs: _runtime_receipt(offload=False),
        generation_runner=lambda **kwargs: calls.append(kwargs["condition"]) or {},
        write=True,
    )

    assert calls == []
    assert artifact["status"] == "blocked"
    assert artifact["models_attempted"] == []
    assert artifact["continuous_self_learning_evidence"] is False
    assert artifact["csl_claim_allowed"] is False
    assert artifact["honest_verdict"].startswith("blocked:")
    assert "no_mandated_local_sota_gguf_available" in artifact["precondition_details"][
        "blocked_preconditions"
    ]
    exp.validate_artifact(artifact)
    assert json.loads((tmp_path / "blocked.json").read_text(encoding="utf-8")) == artifact

    assert exp.extract_answer("nothing usable", ["a", "unknown"]) is None
    assert exp.extract_answer("Use queue-beta, not queue-alpha", ["queue-alpha", "queue-beta"]) == (
        "queue-beta"
    )
    assert exp.estimate_tokens("") == 1
    assert exp.honest_verdict({"csl_claim_allowed": False, "precondition_details": {}}).startswith(
        "blocked: sota_csl_memory_panel_v2_claim_not_allowed"
    )
    missing_runtime = exp.evaluate_preconditions(
        upstream_gates={"all_required_gates_true": False},
        model_specs=[],
        selected_model=None,
        runtime_receipt={"runtime_backend": "transformers"},
    )
    assert set(missing_runtime["blocked_preconditions"]) >= {
        "upstream_csl_gates_not_ready",
        "mandated_model_specs_missing",
        "llama_cpp_gguf_runtime_missing",
    }
    assert exp.model_file_receipt(tmp_path / "missing.gguf") == {
        "path": str(tmp_path / "missing.gguf"),
        "exists": False,
    }
    with pytest.raises(ValueError, match="unknown condition"):
        exp.prompt_for_condition(exp.build_fixture()["heldout_tasks"][0], "bad")
