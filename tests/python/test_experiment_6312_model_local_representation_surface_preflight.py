"""Tests for Exp6312 model-local representation surface preflight.

Spec refs: REQ-INFRA-6312, SCENARIO-INFRA-6312-COMPLETE-NULL,
SCENARIO-INFRA-6312-SURFACE-SELECTION, SCENARIO-INFRA-6312-CONTROLS.
"""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_6312_model_local_representation_surface_preflight as mod


REPO = Path(__file__).resolve().parents[2]
SPEC = REPO / "openspec/capabilities/llm-ebm-inference/spec.md"
TEST_COMMAND = (
    ".venv/bin/pytest "
    "tests/python/test_experiment_6312_model_local_representation_surface_preflight.py "
    "-q --no-cov -n 0"
)
COVERAGE_COMMAND = (
    ".venv/bin/coverage run --rcfile=/dev/null "
    "--include=python/carnot/experiment_6312_model_local_representation_surface_preflight.py "
    "-m pytest tests/python/test_experiment_6312_model_local_representation_surface_preflight.py "
    "-q --no-cov -n 0 && .venv/bin/coverage report --rcfile=/dev/null "
    "--include=python/carnot/experiment_6312_model_local_representation_surface_preflight.py "
    "--fail-under=100"
)
FULL_TEST_COMMAND = ".venv/bin/pytest tests/python -q"
SPEC_COMMAND = (
    ".venv/bin/python scripts/check_spec_coverage.py "
    "tests/python/test_experiment_6312_model_local_representation_surface_preflight.py"
)
RUN_COMMAND = (
    ".venv/bin/python -m carnot.experiment_6312_model_local_representation_surface_preflight "
    "--date 20260811"
)
TEST_COMMANDS = [TEST_COMMAND, COVERAGE_COMMAND, FULL_TEST_COMMAND, SPEC_COMMAND, RUN_COMMAND]
TEST_EXIT_CODES = {command: 0 for command in TEST_COMMANDS}


class FakeSurfaceBackend:
    """Deterministic backend with semantic headroom and no text generation."""

    def __init__(self, model_spec: Mapping[str, Any], config: Mapping[str, Any]) -> None:
        self.model_spec = dict(model_spec)
        self.config = dict(config)
        self.loaded = False
        self.generated_calls = 0

    def load(self) -> dict[str, Any]:
        self.loaded = True
        gpu = int(self.model_spec.get("gpu", 0))
        return {
            "loader_class": "FakeLlamaCppPrefixSurfaceBackend",
            "llama_cpp_version": "fixture",
            "requested_n_gpu_layers": -1,
            "requested_main_gpu": gpu,
            "observed_device_assignment": {
                "before": [{"index": gpu, "memory_used_mb": 1000}],
                "after": [{"index": gpu, "memory_used_mb": 1400}],
                "memory_delta_mb_by_gpu": {str(gpu): 400},
            },
            "embedding_mode": True,
            "output_logits_enabled": False,
            "generated_text_enabled": False,
            "cuda_offload_verified": True,
        }

    def tokenize(self, text: str) -> list[int]:
        return list(range(max(1, len(text.split()))))

    def embed(self, text: str) -> list[float]:
        if not self.loaded:
            raise RuntimeError("backend not loaded")
        prefix_stage = text.count("\n")
        vuln = 1.0 if " or owns_record" in text else 0.0
        fixed = 1.0 if " and owns_record" in text else 0.0
        duplicate = 1.0 if "duplicate-control" in text else 0.0
        model_offset = float(mod.MANDATED_MODEL_HF_IDS.index(str(self.model_spec["hf_id"]))) / 10.0
        return [
            round(vuln - fixed, 8),
            round(prefix_stage / 10.0, 8),
            round(duplicate, 8),
            round(model_offset, 8),
        ]

    def hidden_state_surface(self, text: str) -> list[list[float]]:
        embedding = self.embed(text)
        return [[value + 0.125, value - 0.125] for value in embedding]

    def close(self) -> None:
        self.loaded = False


class DegenerateSurfaceBackend(FakeSurfaceBackend):
    """Backend fixture that is reproducible but has no causal headroom."""

    def embed(self, text: str) -> list[float]:
        if not self.loaded:
            raise RuntimeError("backend not loaded")
        return [0.5, 0.5, 0.5, 0.5]

    def hidden_state_surface(self, text: str) -> list[list[float]]:
        return [[0.5, 0.5]]


def _model_specs(tmp_path: Path, *, missing_index: int | None = None) -> list[dict[str, Any]]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    specs: list[dict[str, Any]] = []
    for index, hf_id in enumerate(mod.MANDATED_MODEL_HF_IDS):
        path = tmp_path / f"model-{index}.gguf"
        if missing_index != index:
            path.write_bytes(f"GGUF fixture {hf_id}".encode("utf-8"))
        specs.append(
            {
                "name": hf_id.rsplit("/", 1)[-1].replace("-GGUF", ""),
                "hf_id": hf_id,
                "family": mod.model_family(hf_id),
                "role": "dense" if "31B" in hf_id else "moe",
                "gpu": index % 2,
                "model_path": str(path),
                "revision": f"fixture-revision-{index}",
                "quantization": "Q4_K_M",
                "context_length": 2048,
                "headline_eligible": True,
                "tokenizer_receipt": {
                    "source": "embedded_gguf_llama_cpp_vocab_only",
                    "loadable": missing_index != index,
                    "detail": "fixture tokenizer",
                    "tokenizer_hash": f"sha256:{index:064x}",
                },
            }
        )
    return specs


def _ready_preconditions(tmp_path: Path) -> dict[str, Any]:
    return {
        "preconditions_ready": True,
        "blocked_reasons": [],
        "python": {"available": True, "version": "fixture", "executable": "python"},
        "llama_cpp": {
            "available": True,
            "version": "fixture",
            "cuda_backend_available": True,
            "hidden_state_probe_safe": True,
        },
        "cuda": {"available": True, "backend": "CUDA", "genuine_offload_required": True},
        "gpu": {
            "gpu_count": 2,
            "devices": [
                {
                    "index": 0,
                    "name": "RTX 4090",
                    "memory_total_mb": 24576,
                    "memory_free_mb": 32768,
                    "memory_used_mb": 1000,
                },
                {
                    "index": 1,
                    "name": "RTX 4090",
                    "memory_total_mb": 24576,
                    "memory_free_mb": 32768,
                    "memory_used_mb": 1000,
                },
            ],
            "ok": True,
        },
        "resources": {
            "memory": {"available_mb": 65536, "required_mb": 8192, "ok": True},
            "disk": {"available_mb": 65536, "required_mb": 1024, "ok": True},
        },
        "output_paths": {
            "result_path": str(tmp_path / mod.RESULT_RELATIVE_PATH.name),
            "row_dir": str(tmp_path),
            "micro_fixture_path": str(tmp_path / mod.MICRO_FIXTURE_RELATIVE_PATH.name),
            "atomic_suffix": ".tmp",
            "ok": True,
        },
        "timeout_budget": {"available_s": 120, "estimated_required_s": 1, "ok": True},
        "random_seeds_checked": {"python": mod.DEFAULT_RANDOM_SEED, "ok": True},
        "protected_hashes_checked_before_model_construction": True,
        "legacy_tiny_models_policy": {"cannot_satisfy_readiness": True},
    }


def _protected_receipt() -> dict[str, Any]:
    return {
        "schema": mod.SCHEMA + ".protected_files_unchanged",
        "protected_files": [path.as_posix() for path in mod.PROTECTED_FILES],
        "records": {},
        "git_status_stdout": "",
        "unchanged": True,
    }


def _memory_receipt(model_spec: Mapping[str, Any], phase: str) -> list[dict[str, Any]]:
    gpu = int(model_spec.get("gpu", 0))
    used = {"before": 1000, "peak": 1500, "after": 1000}[phase]
    return [{"index": gpu, "memory_used_mb": used, "memory_free_mb": 24576 - used}]


def _run_ready(
    tmp_path: Path,
    *,
    backend_factory: type[FakeSurfaceBackend] = FakeSurfaceBackend,
    hidden_available: bool = False,
) -> dict[str, Any]:
    def hidden_probe(model_spec: Mapping[str, Any]) -> dict[str, Any]:
        available = hidden_available and model_spec["hf_id"] == mod.MANDATED_MODEL_HF_IDS[0]
        return {
            "surface": "hidden_state",
            "runtime": "fixture",
            "available": available,
            "tensor_provenance_available": available,
            "available_with_provenance": available,
            "failure": "" if available else "fixture_hidden_state_unavailable",
        }

    return mod.run(
        result_path=tmp_path / mod.RESULT_RELATIVE_PATH.name,
        row_dir=tmp_path,
        micro_fixture_path=tmp_path / mod.MICRO_FIXTURE_RELATIVE_PATH.name,
        model_specs=_model_specs(tmp_path),
        preconditions_checked=_ready_preconditions(tmp_path),
        surface_backend_factory=backend_factory,
        hidden_state_probe=hidden_probe,
        gpu_memory_probe=_memory_receipt,
        protected_files_receipt=_protected_receipt(),
        test_commands=TEST_COMMANDS,
        test_exit_codes=TEST_EXIT_CODES,
        write=True,
    )


def test_req_infra_6312_spec_declares_contract() -> None:
    """REQ-INFRA-6312: OpenSpec names fields, scenarios, models, and null closure."""

    text = SPEC.read_text(encoding="utf-8")
    section = text[text.index("### REQ-INFRA-6312") :]
    for marker in (
        "REQ-INFRA-6312",
        "SCENARIO-INFRA-6312-COMPLETE-NULL",
        "SCENARIO-INFRA-6312-SURFACE-SELECTION",
        "SCENARIO-INFRA-6312-CONTROLS",
        mod.RESULT_RELATIVE_PATH.as_posix(),
        "status=\"complete_null\"",
        "`duration_padding_count`",
        "`source_model_weight_mutation_count`",
    ):
        assert marker in section
    for hf_id in mod.MANDATED_MODEL_HF_IDS:
        assert hf_id in section
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert f"`{field}`" in section


def test_scenario_infra_6312_complete_null_missing_runtime_does_not_load(
    tmp_path: Path,
) -> None:
    """SCENARIO-INFRA-6312-COMPLETE-NULL: missing preconditions stop model construction."""

    calls: list[str] = []

    def forbidden_backend(
        model_spec: Mapping[str, Any], config: Mapping[str, Any]
    ) -> FakeSurfaceBackend:
        calls.append(str(model_spec["hf_id"]))
        raise AssertionError("backend must not load when preconditions are blocked")

    preconditions = _ready_preconditions(tmp_path)
    preconditions["preconditions_ready"] = False
    preconditions["blocked_reasons"] = ["cuda_offload_unavailable"]
    preconditions["cuda"]["available"] = False
    artifact = mod.run(
        result_path=tmp_path / mod.RESULT_RELATIVE_PATH.name,
        row_dir=tmp_path,
        micro_fixture_path=tmp_path / mod.MICRO_FIXTURE_RELATIVE_PATH.name,
        model_specs=_model_specs(tmp_path, missing_index=1),
        preconditions_checked=preconditions,
        surface_backend_factory=forbidden_backend,
        gpu_memory_probe=_memory_receipt,
        protected_files_receipt=_protected_receipt(),
        test_commands=TEST_COMMANDS,
        test_exit_codes=TEST_EXIT_CODES,
        write=True,
    )

    assert calls == []
    assert artifact["status"] == "complete_null"
    assert artifact["honest_verdict"].startswith("complete_null:")
    assert artifact["model_local_representation_surface_ready_score"] == 0.0
    assert artifact["duration_padding_count"] == 0
    assert type(artifact["duration_padding_count"]) is int
    assert artifact["source_model_weight_mutation_count"] == 0
    assert artifact["no_generation_receipt"]["generated_answers_enabled"] is False
    assert "mandated_model_unavailable" in artifact["preconditions_checked"]["blocked_reasons"]
    assert "preconditions_not_ready" in artifact["preconditions_checked"]["blocked_reasons"]
    assert mod.validate_artifact(artifact) is True


def test_scenario_infra_6312_surface_selection_prefers_hidden_only_with_provenance(
    tmp_path: Path,
) -> None:
    """SCENARIO-INFRA-6312-SURFACE-SELECTION: hidden state needs tensor provenance."""

    artifact = _run_ready(tmp_path, hidden_available=True)

    first_model = mod.MANDATED_MODEL_HF_IDS[0]
    assert artifact["status"] == "complete_ready"
    assert artifact["selected_surface_by_model"][first_model]["surface"] == "hidden_state"
    for hf_id in mod.MANDATED_MODEL_HF_IDS[1:]:
        assert artifact["selected_surface_by_model"][hf_id]["surface"] == (
            "prefix_trajectory_fallback"
        )
        assert artifact["hidden_state_runtime_receipts_by_model"][hf_id][
            "available_with_provenance"
        ] is False
    assert artifact["surface_selection_rule"]["frozen_before_label_observation"] is True
    assert artifact["surface_selection_rule"]["uses_fixture_labels"] is False
    assert artifact["prefix_trajectory_fallback_receipts_by_model"][first_model]["used"] is False
    assert mod.validate_artifact(artifact) is True


def test_scenario_infra_6312_controls_gate_per_model_readiness(tmp_path: Path) -> None:
    """SCENARIO-INFRA-6312-CONTROLS: every model passes independently."""

    artifact = _run_ready(tmp_path, hidden_available=False)

    assert artifact["status"] == "complete_ready"
    assert artifact["honest_verdict"].startswith("complete_ready:")
    assert artifact["model_local_representation_surface_ready_score"] == pytest.approx(1.0)
    assert artifact["models_used"] == list(mod.MANDATED_MODEL_HF_IDS)
    assert set(artifact["field_principles"]) == set(mod.REQUIRED_ARTIFACT_FIELDS)
    assert artifact["no_shared_adapter_receipt"]["cross_model_adapter_used"] is False
    assert artifact["no_shared_adapter_receipt"]["pooled_rescue_allowed"] is False
    assert artifact["no_generation_receipt"]["max_tokens_generated"] == 0

    for hf_id in mod.MANDATED_MODEL_HF_IDS:
        assert artifact["causal_intervention_results_by_model"][hf_id]["passed"] is True
        assert artifact["aa_noise_results_by_model"][hf_id]["passed"] is True
        assert artifact[
            "claim_flip_pair_swap_label_permutation_evaluator_swap_results_by_model"
        ][hf_id]["passed"] is True
        assert artifact[
            "norm_length_truncation_duplicate_and_identity_results_by_model"
        ][hf_id]["passed"] is True
        row_receipt = artifact["surface_tensor_shapes_and_hashes"][hf_id]["row_file"]
        assert Path(row_receipt["path"]).exists()
        assert mod.sha256_file(row_receipt["path"]) == row_receipt["sha256"]
    assert artifact["underpowered_or_missing_cells"] == []
    assert mod.validate_artifact(artifact) is True


def test_req_infra_6312_control_failures_close_null_and_validation_catches_tamper(
    tmp_path: Path,
) -> None:
    """REQ-INFRA-6312: degenerate surfaces and tampered artifacts fail closed."""

    artifact = _run_ready(tmp_path, backend_factory=DegenerateSurfaceBackend)

    assert artifact["status"] == "complete_null"
    assert artifact["honest_verdict"].startswith("complete_null:")
    assert artifact["model_local_representation_surface_ready_score"] == 0.0
    assert artifact["underpowered_or_missing_cells"]
    assert all(
        result["passed"] is False
        for result in artifact["causal_intervention_results_by_model"].values()
    )
    assert mod.validate_artifact(artifact) is True

    bad_score = deepcopy(artifact)
    bad_score["model_local_representation_surface_ready_score"] = 1.0
    bad_score["reproducibility_checksum"] = mod.reproducibility_checksum(bad_score)
    with pytest.raises(ValueError, match="model_local_representation_surface_ready_score"):
        mod.validate_artifact(bad_score)

    bad_generation = deepcopy(artifact)
    bad_generation["no_generation_receipt"]["generated_answers_enabled"] = True
    bad_generation["reproducibility_checksum"] = mod.reproducibility_checksum(bad_generation)
    with pytest.raises(ValueError, match="no_generation_receipt"):
        mod.validate_artifact(bad_generation)

    bad_checksum = deepcopy(artifact)
    bad_checksum["reproducibility_checksum"] = "sha256:bad"
    with pytest.raises(ValueError, match="reproducibility_checksum"):
        mod.validate_artifact(bad_checksum)


def test_req_infra_6312_defensive_helpers_and_replay(tmp_path: Path) -> None:
    """REQ-INFRA-6312: helper branches cover deterministic replay and blockers."""

    artifact = _run_ready(tmp_path / "ready")
    refreshed = mod.refresh_artifact_test_exit_codes(
        artifact_path=tmp_path / "ready" / mod.RESULT_RELATIVE_PATH.name,
        test_exit_codes=TEST_EXIT_CODES,
    )
    assert refreshed["reproducibility_checksum"] == mod.reproducibility_checksum(refreshed)

    fixture_rows = mod.build_micro_fixture()
    assert len(fixture_rows) == 2
    for row in fixture_rows:
        assert row["vulnerable_code_length"] == row["fixed_code_length"]
        assert row["row_hash"] == mod.micro_fixture_row_hash(row)
    fixture_path = tmp_path / "fixture.jsonl"
    sidecar_path = tmp_path / "fixture.sidecar.json"
    receipt = mod.write_micro_fixture(fixture_path, sidecar_path)
    assert receipt["ready"] is True
    assert mod.read_micro_fixture(fixture_path) == fixture_rows
    assert mod._read_jsonl(tmp_path / "missing.jsonl") == []
    blank_jsonl = tmp_path / "blank.jsonl"
    blank_jsonl.write_text("\n", encoding="utf-8")
    assert mod._read_jsonl(blank_jsonl) == []
    tampered_fixture = tmp_path / "tampered-fixture.jsonl"
    tampered = deepcopy(fixture_rows[0])
    tampered["pair_id"] = "tampered"
    tampered_fixture.write_text(mod.rows_to_jsonl([tampered]), encoding="utf-8")
    with pytest.raises(ValueError, match="micro_fixture_row_hash"):
        mod.read_micro_fixture(tampered_fixture)

    specs = mod.normalize_model_specs(_model_specs(tmp_path / "specs"))
    assert specs[0]["revision"] == "fixture-revision-0"
    assert specs[0]["tokenizer_hash"].startswith("sha256:")
    snapshot_path = tmp_path / "hub" / "snapshots" / "abc123" / "model.gguf"
    snapshot_path.parent.mkdir(parents=True)
    snapshot_path.write_bytes(b"GGUF")
    assert mod._revision_from_path(str(snapshot_path)) == "abc123"
    assert mod._revision_from_path("") == ""
    assert mod._revision_from_path(str(tmp_path / "plain.gguf")) == "local_file_no_hf_snapshot_revision"
    live_tokenizer_model = tmp_path / "live-tokenizer.gguf"
    live_tokenizer_model.write_bytes(b"GGUF")
    original_tokenizer_probe = mod.gguf_tokenizer_loadable
    mod.gguf_tokenizer_loadable = lambda path: (True, "fixture live tokenizer")  # type: ignore[assignment]
    try:
        live_specs = mod.normalize_model_specs(
            [{"hf_id": mod.MANDATED_MODEL_HF_IDS[0], "model_path": str(live_tokenizer_model)}]
        )
    finally:
        mod.gguf_tokenizer_loadable = original_tokenizer_probe  # type: ignore[assignment]
    assert live_specs[0]["tokenizer_receipt"]["loadable"] is True
    missing = mod.normalize_model_specs([{"hf_id": mod.MANDATED_MODEL_HF_IDS[0]}])
    assert missing[0]["local_model_present"] is False
    assert mod.model_family("example/custom-GGUF") == "custom"
    assert mod._output_path_receipt(tmp_path / "out.json", tmp_path, tmp_path / "micro.jsonl")[
        "ok"
    ] is True
    config = mod.deterministic_surface_config()
    assert {"n_batch", "n_ubatch", "normalize_embeddings"}.issubset(config)
    assert config["max_tokens_generated"] == 0
    assert mod._pad_code_pair("longer", "x")[1].endswith(" ")
    assert mod._flatten_numeric([[1, 2], [3]]) == [1.0, 2.0, 3.0]
    assert mod._shape([]) == [0]
    assert mod._vector_distance([1, 2], [1]) == pytest.approx(0.0)
    assert mod._variance([]) == pytest.approx(0.0)
    assert mod._variance([1.0, 3.0]) == pytest.approx(1.0)
    assert mod.sha256_text(mod._prompt_text(fixture_rows[0], "aa_left")) == mod.sha256_text(
        mod._prompt_text(fixture_rows[0], "aa_right")
    )
    with pytest.raises(ValueError, match="nonfinite_surface"):
        mod._round_floats([1.0, float("nan")])
    with pytest.raises(ValueError, match="unknown_case_kind"):
        mod._prompt_text(fixture_rows[0], "unknown")
    with pytest.raises(ValueError, match="JSON object required"):
        bad_json = tmp_path / "bad.json"
        bad_json.write_text("[]", encoding="utf-8")
        mod._read_json(bad_json)
    with pytest.raises(ValueError, match="JSONL object required"):
        bad_jsonl = tmp_path / "bad.jsonl"
        bad_jsonl.write_text("[]\n", encoding="utf-8")
        mod._read_jsonl(bad_jsonl)

    failed_codes = dict(TEST_EXIT_CODES)
    failed_codes[TEST_COMMAND] = 1
    failed = mod.run(
        result_path=tmp_path / "failed-tests.json",
        row_dir=tmp_path / "failed-tests",
        micro_fixture_path=tmp_path / "failed-tests" / mod.MICRO_FIXTURE_RELATIVE_PATH.name,
        model_specs=_model_specs(tmp_path / "failed-models"),
        preconditions_checked=_ready_preconditions(tmp_path / "failed-tests"),
        surface_backend_factory=FakeSurfaceBackend,
        gpu_memory_probe=_memory_receipt,
        protected_files_receipt=_protected_receipt(),
        test_commands=TEST_COMMANDS,
        test_exit_codes=failed_codes,
        write=True,
    )
    assert failed["status"] == "complete_null"
    assert "failed_test_exit_codes" in failed["underpowered_or_missing_cells"]
    assert mod.validate_artifact(failed) is True

    malformed = deepcopy(artifact)
    malformed.pop("status")
    with pytest.raises(ValueError, match="missing required fields"):
        mod.validate_artifact(malformed)

    original_run_command = mod._run_command
    original_protected = mod.PROTECTED_FILES
    protected_file = tmp_path / "protected.txt"
    protected_file.write_text("stable", encoding="utf-8")
    mod._run_command = lambda command, timeout_s: {  # type: ignore[assignment]
        "returncode": 0,
        "stdout": "",
    }
    mod.PROTECTED_FILES = (Path("protected.txt"),)
    try:
        protected = mod.protected_files_unchanged(tmp_path)
    finally:
        mod._run_command = original_run_command  # type: ignore[assignment]
        mod.PROTECTED_FILES = original_protected
    assert protected["unchanged"] is True

    blocker_preconditions = deepcopy(_ready_preconditions(tmp_path / "blockers"))
    blocker_preconditions.update(
        {
            "preconditions_ready": False,
            "blocked_reasons": ["manual"],
            "llama_cpp": {"available": False, "cuda_backend_available": False},
            "cuda": {"available": False},
            "gpu": {"ok": False, "devices": [{"index": 0, "memory_free_mb": 1}]},
            "resources": {"memory": {"ok": False}, "disk": {"ok": False}},
            "output_paths": {"ok": False},
            "timeout_budget": {"ok": False},
            "protected_hashes_checked_before_model_construction": False,
            "legacy_tiny_models_policy": {"cannot_satisfy_readiness": False},
        }
    )
    blocker_specs = deepcopy(specs)
    blocker_specs[0].pop("min_vram_gb", None)
    blocker_specs[1]["min_vram_gb"] = 99
    blockers = mod._precondition_blockers(blocker_preconditions, blocker_specs[:2])
    assert {
        "manual",
        "preconditions_not_ready",
        "mandated_model_order_mismatch",
        "insufficient_free_vram",
        "llama_cpp_unavailable",
        "llama_cpp_cuda_backend_unavailable",
        "cuda_offload_unavailable",
        "gpu_device_receipt_unavailable",
        "insufficient_free_ram",
        "insufficient_free_disk",
        "output_path_not_writable",
        "timeout_budget_unavailable",
        "protected_hashes_not_checked_before_model_construction",
        "legacy_smoke_policy_missing",
    }.issubset(set(blockers))

    sparse_causal = mod._causal_results([], fixture_rows)
    sparse_aa = mod._aa_results([], fixture_rows)
    assert sparse_causal["passed"] is False
    assert sparse_aa["passed"] is False

    readiness_mutations = [
        ("protected_files_unchanged", {"unchanged": False}),
        ("no_generation_receipt", {"generated_answers_enabled": True}),
        ("no_shared_adapter_receipt", {"cross_model_adapter_used": True}),
    ]
    for field, value in readiness_mutations:
        mutated = deepcopy(artifact)
        mutated[field].update(value)
        assert mod.model_local_representation_surface_ready_score(mutated) == 0.0
    for field in (
        "duration_padding_count",
        "source_model_weight_mutation_count",
    ):
        mutated = deepcopy(artifact)
        mutated[field] = 1
        assert mod.model_local_representation_surface_ready_score(mutated) == 0.0
    mutated = deepcopy(artifact)
    mutated["MODEL_SPECS"] = []
    assert mod.model_local_representation_surface_ready_score(mutated) == 0.0
    for result_field in (
        "selected_surface_by_model",
        "causal_intervention_results_by_model",
        "aa_noise_results_by_model",
        "claim_flip_pair_swap_label_permutation_evaluator_swap_results_by_model",
        "norm_length_truncation_duplicate_and_identity_results_by_model",
    ):
        mutated = deepcopy(artifact)
        if result_field == "selected_surface_by_model":
            mutated[result_field][mod.MANDATED_MODEL_HF_IDS[0]]["surface"] = "none"
        else:
            mutated[result_field][mod.MANDATED_MODEL_HF_IDS[0]]["passed"] = False
        assert mod.model_local_representation_surface_ready_score(mutated) == 0.0

    class FailingSurfaceBackend(FakeSurfaceBackend):
        def embed(self, text: str) -> list[float]:
            raise RuntimeError("fixture failure")

        def close(self) -> None:
            raise RuntimeError("fixture close failure")

    failed_extract = mod.run(
        result_path=tmp_path / "failed-extract.json",
        row_dir=tmp_path / "failed-extract",
        micro_fixture_path=tmp_path / "failed-extract" / mod.MICRO_FIXTURE_RELATIVE_PATH.name,
        model_specs=_model_specs(tmp_path / "failed-extract-models"),
        preconditions_checked=_ready_preconditions(tmp_path / "failed-extract"),
        surface_backend_factory=FailingSurfaceBackend,
        gpu_memory_probe=_memory_receipt,
        protected_files_receipt=_protected_receipt(),
        test_commands=TEST_COMMANDS,
        test_exit_codes=TEST_EXIT_CODES,
        write=True,
    )
    assert failed_extract["status"] == "complete_null"
    assert any("surface_collection_failed" in item for item in failed_extract["underpowered_or_missing_cells"])

    validation_cases = []
    bad_principles = deepcopy(artifact)
    bad_principles["field_principles"] = {}
    bad_principles["model_local_representation_surface_ready_score"] = 0.0
    bad_principles["status"] = "complete_null"
    bad_principles["reproducibility_checksum"] = mod.reproducibility_checksum(bad_principles)
    validation_cases.append((bad_principles, "field_principles"))
    bad_duration = deepcopy(artifact)
    bad_duration["duration_padding_count"] = 1
    bad_duration["model_local_representation_surface_ready_score"] = 0.0
    bad_duration["status"] = "complete_null"
    bad_duration["reproducibility_checksum"] = mod.reproducibility_checksum(bad_duration)
    validation_cases.append((bad_duration, "duration_padding_count"))
    bad_mutation = deepcopy(artifact)
    bad_mutation["source_model_weight_mutation_count"] = 1
    bad_mutation["model_local_representation_surface_ready_score"] = 0.0
    bad_mutation["status"] = "complete_null"
    bad_mutation["reproducibility_checksum"] = mod.reproducibility_checksum(bad_mutation)
    validation_cases.append((bad_mutation, "source_model_weight_mutation_count"))
    bad_tokens = deepcopy(artifact)
    bad_tokens["no_generation_receipt"]["max_tokens_generated"] = 1
    bad_tokens["model_local_representation_surface_ready_score"] = 0.0
    bad_tokens["status"] = "complete_null"
    bad_tokens["reproducibility_checksum"] = mod.reproducibility_checksum(bad_tokens)
    validation_cases.append((bad_tokens, "no_generation_receipt"))
    bad_adapter = deepcopy(artifact)
    bad_adapter["no_shared_adapter_receipt"]["cross_model_adapter_used"] = True
    bad_adapter["model_local_representation_surface_ready_score"] = 0.0
    bad_adapter["status"] = "complete_null"
    bad_adapter["reproducibility_checksum"] = mod.reproducibility_checksum(bad_adapter)
    validation_cases.append((bad_adapter, "no_shared_adapter_receipt"))
    bad_substrate = deepcopy(artifact)
    bad_substrate["inference_substrate"] = "wrong"
    bad_substrate["model_local_representation_surface_ready_score"] = 0.0
    bad_substrate["status"] = "complete_null"
    bad_substrate["reproducibility_checksum"] = mod.reproducibility_checksum(bad_substrate)
    validation_cases.append((bad_substrate, "inference_substrate"))
    bad_oracle = deepcopy(artifact)
    bad_oracle["verifier_is_oracle"] = True
    bad_oracle["model_local_representation_surface_ready_score"] = 0.0
    bad_oracle["status"] = "complete_null"
    bad_oracle["reproducibility_checksum"] = mod.reproducibility_checksum(bad_oracle)
    validation_cases.append((bad_oracle, "verifier_is_oracle"))
    bad_status = deepcopy(artifact)
    bad_status["status"] = "complete_null"
    bad_status["reproducibility_checksum"] = mod.reproducibility_checksum(bad_status)
    validation_cases.append((bad_status, "status"))
    bad_verdict = deepcopy(artifact)
    bad_verdict["honest_verdict"] = "unknown"
    bad_verdict["reproducibility_checksum"] = mod.reproducibility_checksum(bad_verdict)
    validation_cases.append((bad_verdict, "honest_verdict"))
    for bad_artifact, message in validation_cases:
        with pytest.raises(ValueError, match=message):
            mod.validate_artifact(bad_artifact)
