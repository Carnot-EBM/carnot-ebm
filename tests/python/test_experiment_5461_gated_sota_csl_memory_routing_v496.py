"""Tests for Exp5461 gated local SOTA CSL memory routing panel.

Spec refs: REQ-LEARN-5461,
SCENARIO-LEARN-5461-PRECONDITIONS,
SCENARIO-LEARN-5461-CONDITIONS,
SCENARIO-LEARN-5461-VERIFIERS,
SCENARIO-LEARN-5461-NO-WEIGHT-MUTATION.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_5460_csl_policy_bandit_v496 as exp5460
from carnot import experiment_5461_gated_sota_csl_memory_routing_v496 as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec/capabilities/self-learning/spec.md"
RESULT_PATH = REPO / mod.RESULT_RELATIVE_PATH
TEST_COMMAND = (
    ".venv/bin/pytest "
    "tests/python/test_experiment_5461_gated_sota_csl_memory_routing_v496.py "
    "-q --no-cov -n 0"
)
COVERAGE_COMMAND = (
    ".venv/bin/coverage run "
    "--include=python/carnot/experiment_5461_gated_sota_csl_memory_routing_v496.py "
    "-m pytest tests/python/test_experiment_5461_gated_sota_csl_memory_routing_v496.py "
    "-q --no-cov -n 0 && .venv/bin/coverage report "
    "--include=python/carnot/experiment_5461_gated_sota_csl_memory_routing_v496.py "
    "--fail-under=100"
)


def _minimal_gguf(path: Path) -> Path:
    path.write_bytes(b"GGUF-fixture")
    return path


def _paths(tmp_path: Path) -> dict[str, Path]:
    return {
        hf_id: _minimal_gguf(tmp_path / f"{hf_id.split('/')[-1]}.gguf")
        for hf_id in mod.MANDATED_HF_IDS
    }


def _resolver(paths: dict[str, Path]):
    def resolve(hf_id: str, _quantization: str) -> str | None:
        return str(paths[hf_id]) if hf_id in paths else None

    return resolve


def _runtime_receipt(*, offload: bool = True) -> dict[str, Any]:
    return {
        "runtime_backend": "llama_cpp_python_cuda_gguf",
        "llama_cpp_import_ok": True,
        "cuda_visible": offload,
        "gpu_offload_supported": offload,
        "n_gpu_layers": -1,
        "offload_evidence": offload,
        "gpu_memory_delta_mb": 2048 if offload else 0,
        "blocked_preconditions": [] if offload else ["gpu_offload_evidence_missing"],
        "load_receipt": {"offload_evidence": offload, "gpu_memory_delta_mb": 2048 if offload else 0},
    }


def _policy_artifact() -> dict[str, Any]:
    return exp5460.build_artifact(
        root=REPO,
        tests_run=[
            {
                "command": "upstream-exp5460-replay",
                "outcome": "passed",
            }
        ],
    )


def _fake_generation(**kwargs: Any) -> dict[str, Any]:
    task = kwargs["task"]
    condition = kwargs["condition"]
    if condition == mod.NAIVE_CONDITION and task["negative_transfer_candidate"]:
        answer = task["decoy_answer"]
    elif condition == mod.NO_MEMORY_CONDITION and task["requires_memory"]:
        answer = "unknown"
    else:
        answer = task["expected_answer"]
    return {
        "output_text": f"Final answer: {answer}",
        "duration_s": 0.05,
        "generated_token_count": 5,
        "prompt_token_count": len(str(kwargs["prompt"]).split()),
        "backend_details": {"mocked_live_runtime": True},
    }


def _complete_artifact(tmp_path: Path) -> dict[str, Any]:
    return mod.run(
        root=tmp_path,
        artifact_path=tmp_path / mod.RESULT_RELATIVE_PATH,
        row_results_path=tmp_path / mod.ROW_RESULTS_RELATIVE_PATH,
        policy_artifact=_policy_artifact(),
        model_resolver=_resolver(_paths(tmp_path)),
        runtime_probe=lambda **_kwargs: _runtime_receipt(),
        generation_runner=_fake_generation,
        max_tasks=4,
        tests_run=[TEST_COMMAND, COVERAGE_COMMAND],
        write=True,
    )


def test_req_learn_5461_spec_declares_sota_memory_routing_contract() -> None:
    """REQ-LEARN-5461: OpenSpec anchors the live SOTA routing panel."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("## REQ-LEARN-5461") :]
    normalized = " ".join(section.split())

    for marker in (
        "REQ-LEARN-5461",
        "SCENARIO-LEARN-5461-PRECONDITIONS",
        "SCENARIO-LEARN-5461-CONDITIONS",
        "SCENARIO-LEARN-5461-VERIFIERS",
        "SCENARIO-LEARN-5461-NO-WEIGHT-MUTATION",
        str(mod.RESULT_RELATIVE_PATH),
        str(mod.ROW_RESULTS_RELATIVE_PATH),
        "no-memory, naive-ICL, governed-memory, and policy-selected conditions",
        "Exact deterministic task verifiers or witnesses",
        mod.INFERENCE_SUBSTRATE,
    ):
        assert marker in section
    for hf_id in mod.MANDATED_HF_IDS:
        assert hf_id in section
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert " ".join(principle.split()) in normalized


def test_scenario_learn_5461_preconditions_block_before_generation(tmp_path: Path) -> None:
    """SCENARIO-LEARN-5461-PRECONDITIONS: failed gates do not call the model."""

    calls: list[str] = []
    artifact = mod.run(
        root=tmp_path,
        artifact_path=tmp_path / "blocked.json",
        row_results_path=tmp_path / "blocked_rows.jsonl",
        policy_artifact={"csl_policy_ready": False, "policy_snapshot": {}},
        model_resolver=lambda _hf_id, _quantization: None,
        runtime_probe=lambda **_kwargs: _runtime_receipt(offload=False),
        generation_runner=lambda **kwargs: calls.append(kwargs["condition"]) or {},
        tests_run=[TEST_COMMAND],
        write=True,
    )

    assert calls == []
    assert json.loads((tmp_path / "blocked.json").read_text(encoding="utf-8")) == artifact
    assert artifact["preconditions_checked"] is True
    assert artifact["gpu_offload_verified"] is False
    assert artifact["csl_sota_memory_routing_ready"] is False
    assert artifact["honest_verdict"].startswith("blocked:")
    assert not (tmp_path / "blocked_rows.jsonl").exists()
    assert set(artifact["precondition_details"]["blocked_preconditions"]) >= {
        "exp5460_csl_policy_not_ready",
        "non_empty_mandated_model_paths_missing",
        "no_mandated_local_gguf_model_path",
        "cuda_not_visible",
        "gpu_offload_evidence_missing",
    }
    mod.validate_artifact(artifact)


def test_scenario_learn_5461_complete_run_writes_rows_and_policy_receipts(
    tmp_path: Path,
) -> None:
    """SCENARIO-LEARN-5461-CONDITIONS: complete panel writes scored row evidence."""

    artifact = _complete_artifact(tmp_path)
    rows = [
        json.loads(line)
        for line in Path(artifact["row_results_path"]).read_text(encoding="utf-8").splitlines()
    ]

    mod.validate_artifact(artifact)
    assert json.loads((tmp_path / mod.RESULT_RELATIVE_PATH).read_text(encoding="utf-8")) == artifact
    assert artifact["status"] == "complete"
    assert artifact["preconditions_checked"] is True
    assert artifact["gpu_offload_verified"] is True
    assert artifact["runtime_backend"] == "llama_cpp_python_cuda_gguf"
    assert artifact["condition_names"] == list(mod.CONDITION_NAMES)
    assert artifact["row_results_path"] == str(tmp_path / mod.ROW_RESULTS_RELATIVE_PATH)
    assert artifact["policy_state_checksum"] == mod.policy_state_checksum(_policy_artifact())
    assert artifact["quality_delta_vs_no_memory"] > 0.0
    assert artifact["quality_delta_vs_naive_icl"] > 0.0
    assert artifact["context_efficiency_delta"] > 0.0
    assert artifact["verifier_cost_delta"] > 0.0
    assert artifact["negative_transfer_deflection_rate"] == pytest.approx(1.0)
    assert artifact["stale_memory_deflection_rate"] == pytest.approx(1.0)
    assert artifact["no_weight_mutation"] is True
    assert artifact["csl_sota_memory_routing_ready"] is True
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["field_principles"] == mod.FIELD_PRINCIPLES
    assert artifact["research_conductor_modified"] is False
    assert len(rows) == artifact["task_count"] * len(mod.CONDITION_NAMES)
    assert rows == artifact["row_results"]
    assert {row["condition"] for row in rows} == set(mod.CONDITION_NAMES)
    assert {row["task_id"] for row in rows} == {
        row["task_id"] for row in mod.build_task_stream(_policy_artifact())[:4]
    }

    for row in rows:
        assert row["model_hf_id"] in mod.MANDATED_HF_IDS
        assert row["model_path"].endswith(".gguf")
        assert row["runtime_backend"] == artifact["runtime_backend"]
        assert row["gpu_offload_evidence"] is True
        assert row["prompt_hash"]
        assert row["prompt_text"]
        assert row["memory_receipt"]["condition"] == row["condition"]
        assert row["policy_receipt"]["policy_state_checksum"] == artifact["policy_state_checksum"]
        assert row["exact_verifier_witness"]["authority"] == "exact_task_verifier"
        assert row["final_authority_bypassed"] is False
        assert row["row_checksum"] == mod.row_checksum(row)


def test_scenario_learn_5461_exact_verifier_rejects_self_verdict_and_decoy() -> None:
    """SCENARIO-LEARN-5461-VERIFIERS: exact witnesses, not model verdicts, score rows."""

    task = mod.build_task_stream(_policy_artifact())[1]
    self_verdict = mod.exact_task_verifier(task, "I verified myself as correct.")
    decoy = mod.exact_task_verifier(task, f"Final answer: {task['decoy_answer']}")
    correct = mod.exact_task_verifier(task, f"Final answer: {task['expected_answer']}")

    assert self_verdict["accepted"] is False
    assert "answer_not_found" in self_verdict["failure_reasons"]
    assert decoy["accepted"] is False
    assert decoy["selected_answer"] == task["decoy_answer"]
    assert decoy["negative_transfer_detected"] is True
    assert correct["accepted"] is True
    assert correct["selected_answer"] == task["expected_answer"]


def test_req_learn_5461_validation_fails_closed_for_drift(tmp_path: Path) -> None:
    """REQ-LEARN-5461: validation rejects model, row, and readiness drift."""

    artifact = _complete_artifact(tmp_path)

    missing_model = deepcopy(artifact)
    missing_model["model_specs"] = [
        spec
        for spec in missing_model["model_specs"]
        if spec["hf_id"] != "unsloth/Qwen3.6-35B-A3B-GGUF"
    ]
    with pytest.raises(ValueError, match="mandated model_specs"):
        mod.validate_artifact(missing_model)

    cpu_headline = deepcopy(artifact)
    for spec in cpu_headline["model_specs"]:
        if spec["ran_headline"]:
            spec["gpu_offload_verified"] = False
    with pytest.raises(ValueError, match="CPU-only headline"):
        mod.validate_artifact(cpu_headline)

    bad_authority = deepcopy(artifact)
    bad_authority["row_results"][0]["exact_verifier_witness"]["authority"] = "model_self_verdict"
    bad_authority["row_results"][0]["final_authority_bypassed"] = True
    with pytest.raises(ValueError, match="exact task verifier"):
        mod.validate_artifact(bad_authority)

    bad_path = deepcopy(artifact)
    bad_path["row_results_path"] = str(tmp_path / "missing_rows.jsonl")
    with pytest.raises(ValueError, match="row_results_path"):
        mod.validate_artifact(bad_path)

    self_validating = deepcopy(artifact)
    self_validating["metric_dependency_graph"]["csl_sota_memory_routing_ready"] = [
        "csl_sota_memory_routing_ready"
    ]
    with pytest.raises(ValueError, match="self-validating readiness"):
        mod.validate_artifact(self_validating)

    blocked_ready = deepcopy(artifact)
    blocked_ready["status"] = "blocked"
    with pytest.raises(ValueError, match="complete status"):
        mod.validate_artifact(blocked_ready)


def test_req_learn_5461_defensive_helpers_and_live_runner_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-LEARN-5461: defensive branches fail closed with concrete errors."""

    kwargs: dict[str, Any] = {
        "root": tmp_path,
        "artifact_path": tmp_path / "default.json",
        "row_results_path": tmp_path / "default_rows.jsonl",
        "policy_artifact": _policy_artifact(),
        "model_resolver": _resolver(_paths(tmp_path)),
        "runtime_probe": lambda **_kwargs: _runtime_receipt(),
        "max_tasks": 1,
        "tests_run": [{"command": "mapped", "outcome": "passed"}],
    }

    class FakeLiveRunner:
        def __init__(self, **_kwargs: Any) -> None:
            self.load_receipt = {"offload_evidence": True, "gpu_memory_delta_mb": 1024}

        def __call__(self, **call_kwargs: Any) -> dict[str, Any]:
            return _fake_generation(**call_kwargs)

    monkeypatch.setattr(mod, "LlamaCslGenerationRunner", FakeLiveRunner)
    artifact = mod.run(**kwargs, write=True)
    assert artifact["gpu_offload_verified"] is True
    assert mod.run(
        **(kwargs | {"artifact_path": tmp_path / "write_false.json"}),
        write=False,
    )["metric_independence_checks_passed"] is True

    class CpuOnlyRunner:
        def __init__(self, **_kwargs: Any) -> None:
            self.load_receipt = {"offload_evidence": False, "gpu_memory_delta_mb": 0}

    monkeypatch.setattr(mod, "LlamaCslGenerationRunner", CpuOnlyRunner)
    cpu_only = mod.run(**(kwargs | {"artifact_path": tmp_path / "cpu.json"}), write=True)
    assert cpu_only["honest_verdict"].startswith("blocked:")
    assert "gpu_offload_not_observed_after_load" in cpu_only["precondition_details"][
        "blocked_preconditions"
    ]

    class RaisingRunner:
        def __init__(self, **_kwargs: Any) -> None:
            raise RuntimeError("load failed")

    monkeypatch.setattr(mod, "LlamaCslGenerationRunner", RaisingRunner)
    failed_load = mod.run(**(kwargs | {"artifact_path": tmp_path / "raise.json"}), write=True)
    assert "llama_cpp_model_load_failed" in failed_load["honest_verdict"]

    preconditions = mod.evaluate_preconditions(
        policy_artifact={},
        model_specs=[],
        selected_model=None,
        runtime_receipt={"runtime_backend": "transformers", "cuda_visible": False},
    )
    assert set(preconditions["blocked_preconditions"]) >= {
        "exp5460_csl_policy_not_ready",
        "mandated_model_specs_missing",
        "non_empty_mandated_model_paths_missing",
        "no_mandated_local_gguf_model_path",
        "cuda_not_visible",
        "gpu_offload_evidence_missing",
        "llama_cpp_gguf_runtime_missing",
    }

    with pytest.raises(ValueError, match="unknown condition"):
        mod.build_prompt_and_receipts(
            mod.build_task_stream(_policy_artifact())[0],
            condition="bad",
            policy_checksum=mod.policy_state_checksum(_policy_artifact()),
        )
    with pytest.raises(ValueError, match="unknown effective condition"):
        mod._memory_lines({}, "bad")  # noqa: SLF001

    assert mod.model_file_receipt(tmp_path / "missing.gguf") == {
        "path": str(tmp_path / "missing.gguf"),
        "exists": False,
    }
    assert mod._normalise_tests_run([]) == [  # noqa: SLF001
        {"command": "not_recorded", "outcome": "not_recorded"}
    ]
    assert mod._read_json(tmp_path / "missing.json") == {}  # noqa: SLF001
    bad_json = tmp_path / "bad.json"
    bad_json.write_text("{bad", encoding="utf-8")
    assert mod._read_json(bad_json) == {}  # noqa: SLF001
    scalar_json = tmp_path / "scalar.json"
    scalar_json.write_text("7", encoding="utf-8")
    assert mod._read_json(scalar_json) == {}  # noqa: SLF001
    assert mod._list_of_mappings("bad") == []  # noqa: SLF001
    assert mod._float_close("bad", 1.0) is False  # noqa: SLF001

    no_prompt_tokens = mod.score_candidate_row(
        task=mod.build_task_stream(_policy_artifact())[0],
        condition=mod.NO_MEMORY_CONDITION,
        model_spec=artifact["selected_model_spec"],
        runtime_backend=artifact["runtime_backend"],
        runtime_receipt=artifact["runtime_receipt"],
        generation={"output_text": "unknown"},
        prompt_text="short prompt",
        memory_receipt={"condition": mod.NO_MEMORY_CONDITION, "effective_condition": mod.NO_MEMORY_CONDITION},
        policy_receipt={"policy_state_checksum": artifact["policy_state_checksum"]},
        seed=1,
        fallback_duration_s=0.0,
    )
    assert no_prompt_tokens["context_cost"] == 2

    assert "quality_regressed_vs_no_memory" in mod._honest_verdict(  # noqa: SLF001
        False,
        [{"row": "present"}],
        {},
        {
            "quality_delta_vs_no_memory": -1.0,
            "quality_delta_vs_naive_icl": -1.0,
            "negative_transfer_deflection_rate": 0.0,
        },
    )

    schema_cases: list[tuple[str, Any, str]] = [
        ("field_principles", {}, "field_principles"),
        ("preconditions_checked", "yes", "preconditions_checked"),
        ("headline_required_any_of", [], "headline_required_any_of"),
        ("condition_names", [], "condition_names"),
        ("gpu_offload_verified", "yes", "gpu_offload_verified"),
        ("no_weight_mutation", "yes", "no_weight_mutation"),
        ("csl_sota_memory_routing_ready", "yes", "csl_sota_memory_routing_ready"),
        ("metric_independence_checks_passed", "yes", "metric_independence_checks_passed"),
        ("runtime_backend", 7, "runtime_backend"),
        ("policy_state_checksum", "bad", "policy_state_checksum"),
        ("honest_verdict", "done\n", "honest_verdict"),
        ("research_conductor_modified", True, "research_conductor.py"),
        ("row_results", "bad", "row_results"),
        ("model_specs", "bad", "model_specs"),
    ]
    for field, value, expected in schema_cases:
        bad = deepcopy(artifact)
        bad[field] = value
        assert expected in "; ".join(mod.artifact_schema_errors(bad))

    missing = deepcopy(artifact)
    del missing["runtime_backend"]
    assert "missing required fields" in "; ".join(mod.artifact_schema_errors(missing))

    aggregate_cases: list[tuple[str, Any, str]] = [
        ("quality_delta_vs_no_memory", -99.0, "quality_delta_vs_no_memory"),
        ("quality_delta_vs_naive_icl", -99.0, "quality_delta_vs_naive_icl"),
        ("context_efficiency_delta", -99.0, "context_efficiency_delta"),
        ("verifier_cost_delta", -99.0, "verifier_cost_delta"),
        ("negative_transfer_deflection_rate", -99.0, "negative_transfer_deflection_rate"),
        ("stale_memory_deflection_rate", -99.0, "stale_memory_deflection_rate"),
        ("condition_metrics", {}, "condition_metrics"),
        ("policy_regret_proxy", {}, "policy_regret_proxy"),
        ("metric_independence_checks_passed", False, "metric_independence_checks_passed"),
        ("metric_details", {}, "metric_details"),
        ("task_count", artifact["task_count"] + 1, "task_count"),
        ("row_count", artifact["row_count"] + 1, "row_count"),
        ("row_checksums", [], "row_checksums"),
    ]
    for field, value, expected in aggregate_cases:
        bad = deepcopy(artifact)
        bad[field] = value
        assert expected in "; ".join(mod.artifact_schema_errors(bad))

    ready_dir = tmp_path / "ready"
    ready_dir.mkdir()
    ready_artifact = _complete_artifact(ready_dir)
    ready_cases: list[tuple[str, Any, str]] = [
        ("inference_substrate", mod.BLOCKED_INFERENCE_SUBSTRATE, "live frozen-policy substrate"),
        ("gpu_offload_verified", False, "gpu_offload_verified"),
        ("quality_delta_vs_no_memory", -1.0, "quality_delta_vs_no_memory"),
        ("quality_delta_vs_naive_icl", -1.0, "quality_delta_vs_naive_icl"),
        ("context_efficiency_delta", 0.0, "context_efficiency_delta"),
        ("verifier_cost_delta", 0.0, "verifier_cost_delta"),
        ("negative_transfer_deflection_rate", 0.0, "negative_transfer_deflection_rate"),
        ("no_weight_mutation", False, "no_weight_mutation"),
    ]
    for field, value, expected in ready_cases:
        bad = deepcopy(ready_artifact)
        bad[field] = value
        assert expected in "; ".join(mod.artifact_schema_errors(bad))

    no_headline = deepcopy(ready_artifact)
    for spec in no_headline["model_specs"]:
        spec["ran_headline"] = False
    assert "at least one mandated model ran" in "; ".join(mod.artifact_schema_errors(no_headline))

    ready_without_rows = deepcopy(ready_artifact)
    ready_without_rows["row_results"] = []
    assert "row evidence" in "; ".join(mod.artifact_schema_errors(ready_without_rows))

    legacy_headline = deepcopy(ready_artifact)
    for spec in legacy_headline["model_specs"]:
        if spec["ran_headline"]:
            spec["legacy_smoke_only"] = True
    assert "legacy smoke" in "; ".join(mod.artifact_schema_errors(legacy_headline))

    row_cases: list[tuple[str, Any, str]] = [
        ("model_hf_id", "legacy/not-headline", "row model_hf_id"),
        ("model_path", "/tmp/model.bin", "row model_path"),
        ("condition", "unknown", "row condition"),
        ("gpu_offload_evidence", "yes", "row gpu_offload_evidence"),
    ]
    for field, value, expected in row_cases:
        bad = deepcopy(artifact)
        bad["row_results"][0][field] = value
        assert expected in "; ".join(mod.artifact_schema_errors(bad))

    bad_checksum = deepcopy(artifact)
    bad_checksum["row_results"][0]["row_checksum"] = "0" * 64
    assert "row checksum" in "; ".join(mod.artifact_schema_errors(bad_checksum))

    bad_policy_receipt = deepcopy(artifact)
    bad_policy_receipt["row_results"][0]["policy_receipt"]["policy_state_checksum"] = "bad"
    assert "policy receipt" in "; ".join(mod.artifact_schema_errors(bad_policy_receipt))

    missing_path = deepcopy(artifact)
    missing_path["row_results_path"] = str(tmp_path / "missing.jsonl")
    assert "row_results_path must point" in "; ".join(mod.artifact_schema_errors(missing_path))

    bad_path_type = deepcopy(artifact)
    bad_path_type["row_results_path"] = 7
    assert "row_results_path must be" in "; ".join(mod.artifact_schema_errors(bad_path_type))

    invalid_jsonl = tmp_path / "invalid.jsonl"
    invalid_jsonl.write_text("{bad}\n", encoding="utf-8")
    invalid_rows = deepcopy(artifact)
    invalid_rows["row_results_path"] = str(invalid_jsonl)
    assert "row_results_path is unreadable" in "; ".join(mod.artifact_schema_errors(invalid_rows))

    bad_graph = deepcopy(artifact)
    bad_graph["metric_dependency_graph"] = "bad"
    assert "metric_dependency_graph must be a dict" in "; ".join(
        mod.artifact_schema_errors(bad_graph)
    )

    missing_graph_deps = deepcopy(artifact)
    missing_graph_deps["metric_dependency_graph"] = {}
    assert "readiness dependencies" in "; ".join(
        mod.artifact_schema_errors(missing_graph_deps)
    )


def test_req_learn_5461_repository_artifact_is_valid() -> None:
    """REQ-LEARN-5461-6: checked-in deliverable has valid rows and receipts."""

    result = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
    rows = [
        json.loads(line)
        for line in Path(result["row_results_path"]).read_text(encoding="utf-8").splitlines()
    ]

    mod.validate_artifact(result)
    assert rows == result["row_results"]
    assert result["csl_sota_memory_routing_ready"] is True
    assert result["no_weight_mutation"] is True
    assert result["inference_substrate"] == mod.INFERENCE_SUBSTRATE
