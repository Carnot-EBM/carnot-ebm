"""Tests for Exp6486 forced-candidate representation streams.

Spec refs: REQ-VERIFY-6486, SCENARIO-VERIFY-6486-PRECONDITIONS,
SCENARIO-VERIFY-6486-CANDIDATES, SCENARIO-VERIFY-6486-NO-GENERATION,
SCENARIO-VERIFY-6486-RAW-ROWS, SCENARIO-VERIFY-6486-FAMILY-HELD,
SCENARIO-VERIFY-6486-ARTIFACT.
"""

from __future__ import annotations

from collections.abc import Mapping
import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_6486_three_family_forced_candidate_representations as mod
import scripts.adversarial_verify as av


REPO = Path(__file__).resolve().parents[2]
SPEC = REPO / mod.SPEC_RELATIVE_PATH


class FakeRepresentationBackend:
    """Small deterministic backend that cannot generate text."""

    load_count = 0

    def __init__(self, model_spec: Mapping[str, Any], config: Mapping[str, Any]) -> None:
        self.model_spec = dict(model_spec)
        self.config = dict(config)
        self.loaded = False

    def load(self) -> dict[str, Any]:
        FakeRepresentationBackend.load_count += 1
        self.loaded = True
        return {
            "loader_class": "FakeLlamaCppEmbedding",
            "llama_cpp_version": "fixture",
            "requested_main_gpu": self.model_spec["gpu"],
            "requested_n_gpu_layers": -1,
            "embedding_mode": True,
            "fixed_sequence_forward": True,
            "generated_text_enabled": False,
            "output_logits_enabled": False,
            "observed_device_assignment": {
                "memory_delta_mb_by_gpu": {str(self.model_spec["gpu"]): 128}
            },
        }

    def tokenize(self, text: str) -> list[int]:
        return [index + 1 for index, _ in enumerate(text.split())]

    def embed(self, text: str) -> list[float]:
        if not self.loaded:
            raise RuntimeError("backend_not_loaded")
        digest = mod.sha256_text(f"{self.model_spec['hf_id']}|{text}")
        width = int(self.model_spec["test_native_dimension"])
        return [
            round((int(digest[7 + 2 * i : 9 + 2 * i], 16) + i) / 257.0, 8)
            for i in range(width)
        ]

    def close(self) -> None:
        self.loaded = False


class ExplodingBackend(FakeRepresentationBackend):
    """Backend that proves blocked preconditions do not load a model."""

    def load(self) -> dict[str, Any]:  # pragma: no cover - must not execute.
        raise AssertionError("model_load_should_not_run")


def _fake_model_specs(tmp_path: Path) -> list[dict[str, Any]]:
    dims = {
        "unsloth/Qwen3.6-35B-A3B-GGUF": 5,
        "unsloth/gemma-4-31B-it-GGUF": 7,
        "unsloth/gemma-4-26B-A4B-it-GGUF": 3,
    }
    specs = []
    for index, hf_id in enumerate(mod.MANDATED_MODEL_HF_IDS):
        path = tmp_path / f"model-{index}.gguf"
        path.write_bytes(f"GGUF fixture {hf_id}".encode("utf-8"))
        specs.append(
            {
                "name": hf_id.rsplit("/", 1)[-1].replace("-GGUF", ""),
                "hf_id": hf_id,
                "gpu": index % 2,
                "model_path": str(path),
                "quantization": "Q4_K_M",
                "headline_eligible": True,
                "tokenizer_receipt": {
                    "source": "fixture",
                    "loadable": True,
                    "detail": "fixture tokenizer",
                },
                "test_native_dimension": dims[hf_id],
            }
        )
    return specs


def _ready_preconditions(tmp_path: Path) -> dict[str, Any]:
    return {
        "preconditions_ready": True,
        "blocked_reasons": [],
        "checks": {
            "exp6482_ready": True,
            "exp6482_hashes_match": True,
            "exp6484_gate_passed": True,
            "dual_rtx3090_visible": True,
            "required_cache_paths_exist": True,
            "embedded_tokenizers_load": True,
            "disk_space_adequate": True,
            "memory_adequate": True,
            "no_retired_generated_answer_path": True,
            "llama_cpp_gpu_offload_supported": True,
        },
        "gpu": {
            "gpu_count": 2,
            "devices": [
                {"index": 0, "name": "NVIDIA GeForce RTX 3090", "memory_free_mb": 24000},
                {"index": 1, "name": "NVIDIA GeForce RTX 3090", "memory_free_mb": 24000},
            ],
        },
        "resources": {
            "disk": {"available_mb": 8192, "required_mb": 1024, "ok": True},
            "memory": {"available_mb": 8192, "required_mb": 1024, "ok": True},
        },
        "output_paths": {
            "result_path": str(tmp_path / "artifact.json"),
            "raw_vector_dir": str(tmp_path / "raw"),
            "ok": True,
        },
    }


def _run_fake(tmp_path: Path, *, unit_limit: int = 2) -> dict[str, Any]:
    FakeRepresentationBackend.load_count = 0
    return mod.run(
        root=REPO,
        result_path=tmp_path / "artifact.json",
        raw_vector_dir=tmp_path / "raw",
        model_specs=_fake_model_specs(tmp_path),
        preconditions_checked=_ready_preconditions(tmp_path),
        representation_backend_factory=FakeRepresentationBackend,
        tests_run=[{"command": "pytest fixture", "exit_code": 0}],
        unit_limit=unit_limit,
        write=True,
    )


def test_req_verify_6486_spec_declares_fields_and_scenarios() -> None:
    """REQ-VERIFY-6486: OpenSpec owns the forced representation contract."""

    text = SPEC.read_text(encoding="utf-8")
    section = text[text.index("### REQ-VERIFY-6486") : text.index("### REQ-VERIFY-6478")]
    normalized = " ".join(section.split())

    for marker in (
        "SCENARIO-VERIFY-6486-PRECONDITIONS",
        "SCENARIO-VERIFY-6486-CANDIDATES",
        "SCENARIO-VERIFY-6486-NO-GENERATION",
        "SCENARIO-VERIFY-6486-RAW-ROWS",
        "SCENARIO-VERIFY-6486-FAMILY-HELD",
        "SCENARIO-VERIFY-6486-ARTIFACT",
        mod.RESULT_RELATIVE_PATH.as_posix(),
        mod.INFERENCE_SUBSTRATE,
    ):
        assert marker in section
    for hf_id in mod.MANDATED_MODEL_HF_IDS:
        assert hf_id in section
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert " ".join(principle.split()) in normalized


def test_scenario_verify_6486_model_specs_call_cached_pair(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """REQ-VERIFY-6486: MODEL_SPECS use cached_sota_pair plus the third family."""

    calls: list[str] = []

    def fake_cached_sota_pair(*, gpu_indices: tuple[int, int] = (0, 1)) -> list[dict[str, Any]]:
        calls.append(f"cached:{gpu_indices}")
        return [
            {
                "name": "Qwen",
                "hf_id": "unsloth/Qwen3.6-35B-A3B-GGUF",
                "gpu": 0,
                "model_path": str(tmp_path / "qwen.gguf"),
            },
            {
                "name": "Gemma26",
                "hf_id": "unsloth/gemma-4-26B-A4B-it-GGUF",
                "gpu": 1,
                "model_path": str(tmp_path / "gemma26.gguf"),
            },
        ]

    def fake_resolve(hf_id: str, preferred_quant: str = "Q4_K_M") -> str:
        calls.append(f"resolve:{hf_id}:{preferred_quant}")
        return str(tmp_path / "gemma31.gguf")

    for name in ("qwen.gguf", "gemma26.gguf", "gemma31.gguf"):
        (tmp_path / name).write_bytes(b"fixture")
    monkeypatch.setattr(mod, "cached_sota_pair", fake_cached_sota_pair)
    monkeypatch.setattr(mod, "resolve_cached_gguf", fake_resolve)
    monkeypatch.setattr(
        mod,
        "gguf_tokenizer_loadable",
        lambda path: (True, f"tokenizer ok {path}"),
    )

    specs = mod.resolve_model_specs()

    assert calls[0] == "cached:(0, 1)"
    assert [spec["hf_id"] for spec in specs] == list(mod.MANDATED_MODEL_HF_IDS)
    assert all(spec["local_model_present"] is True for spec in specs)
    assert all(spec["tokenizer_receipt"]["loadable"] is True for spec in specs)
    assert specs[1]["hf_id"] == "unsloth/gemma-4-31B-it-GGUF"


def test_scenario_verify_6486_candidates_are_exact_and_frozen() -> None:
    """SCENARIO-VERIFY-6486-CANDIDATES: each unit gets frozen exact candidates."""

    upstream = mod.load_upstream_artifacts(root=REPO)
    manifest = mod.build_candidate_commitment_manifest(
        upstream.exp6482,
        unit_limit=3,
        commitment_monotonic_ns=1000,
        model_access_start_ns=2000,
    )

    assert manifest["unit_count"] == 3
    assert manifest["candidate_count"] == 9
    assert manifest["all_candidates_committed_before_model_access"] is True
    for unit in manifest["units"]:
        candidates = unit["candidates"]
        assert [row["candidate_kind"] for row in candidates] == [
            "exact_correct",
            "controlled_wrong_protected",
            "controlled_wrong_alternate",
        ]
        assert candidates[0]["exact_label"] is True
        assert candidates[1]["exact_label"] is False
        assert candidates[2]["exact_label"] is False
        assert candidates[1]["violated_protected_constraint_ids"]
        assert candidates[2]["violated_constraint_ids"]
        assert len({row["candidate_hash"] for row in candidates}) == 3
        assert all(row["candidate_byte_length"] > 0 for row in candidates)
        assert all(row["pre_model_commitment_ns"] < row["model_access_start_ns"] for row in candidates)


def test_scenario_verify_6486_no_generation_guard_fails_closed() -> None:
    """SCENARIO-VERIFY-6486-NO-GENERATION: prohibited calls are recorded."""

    guard = mod.NoGenerationCallGuard()
    guard.record_call("load_representation_backend")
    guard.record_call("embed_fixed_candidate")
    with pytest.raises(RuntimeError, match="generation_call_prohibited"):
        guard.record_call("generate")

    receipt = guard.receipt()
    assert receipt["generation_call_count"] == 1
    assert receipt["allowed_call_count"] == 2
    assert receipt["prohibited_method_calls"] == ["generate"]
    assert mod.validate_no_generation_receipts([receipt])["accepted"] is False


def test_scenario_verify_6486_fake_run_persists_once_and_recomputes(
    tmp_path: Path,
) -> None:
    """SCENARIO-VERIFY-6486-RAW-ROWS/FAMILY-HELD/ARTIFACT."""

    artifact = _run_fake(tmp_path, unit_limit=2)
    errors = mod.validate_artifact(artifact)
    rows = artifact["per_unit_rows"]

    assert errors == []
    assert json.loads((tmp_path / "artifact.json").read_text(encoding="utf-8")) == artifact
    assert artifact["status"] == "complete_representation_stream"
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["prospective_representation_stream_ready_score"] == pytest.approx(1.0)
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["verifier_is_oracle"] is False
    assert FakeRepresentationBackend.load_count == 3
    assert len(rows) == 2 * mod.CANDIDATES_PER_UNIT * len(mod.MANDATED_MODEL_HF_IDS)
    assert len({row["cell_id"] for row in rows}) == len(rows)
    assert len({entry["path"] for entry in artifact["raw_vector_manifest"]["vectors"]}) == len(rows)
    assert all(Path(entry["path"]).is_file() for entry in artifact["raw_vector_manifest"]["vectors"])
    assert all(entry["write_count"] == 1 for entry in artifact["raw_vector_manifest"]["vectors"])
    assert all(row["row_hash"] == mod.row_hash(row) for row in rows)
    assert all(row["raw_vector_hash"] == mod.sha256_file(row["raw_vector_path"]) for row in rows)
    assert artifact["aggregate_row_recomputation"]["ready_score_from_rows"] == pytest.approx(1.0)
    assert artifact["aggregate_row_recomputation"]["complete_unique_raw_rows"] is True
    assert artifact["aggregate_row_recomputation"]["no_generation_call_occurred"] is True
    assert artifact["candidate_commitment_manifest"]["manifest_hash"]
    assert artifact["no_generation_receipts"]["generation_call_count"] == 0
    assert artifact["family_separation_receipts"]["native_dimensions_by_family"] == {
        "gemma4_26b_a4b_it": 3,
        "gemma4_31b_it": 7,
        "qwen3_6_35b_a3b": 5,
    }
    assert artifact["held_isolation_receipt"]["held_vectors_inspected_during_transform_design"] == 0
    assert artifact["held_isolation_receipt"]["storage_roots_distinct"] is True
    assert all(
        receipt["monotonic_start_ns"] < receipt["monotonic_end_ns"]
        for receipt in artifact["phase_concurrency_receipts"]["phase_rows"]
    )
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert field in artifact
        assert field in artifact["field_principles"]
        assert field in artifact["field_provenance"]


def test_scenario_verify_6486_blocked_precondition_stops_before_model_load(
    tmp_path: Path,
) -> None:
    """SCENARIO-VERIFY-6486-PRECONDITIONS: blockers stop before model access."""

    preconditions = _ready_preconditions(tmp_path)
    preconditions["preconditions_ready"] = False
    preconditions["blocked_reasons"] = ["dual_rtx3090_visible"]
    preconditions["checks"]["dual_rtx3090_visible"] = False

    artifact = mod.run(
        root=REPO,
        result_path=tmp_path / "blocked.json",
        raw_vector_dir=tmp_path / "raw",
        model_specs=_fake_model_specs(tmp_path),
        preconditions_checked=preconditions,
        representation_backend_factory=ExplodingBackend,
        tests_run=[],
        unit_limit=1,
        write=True,
    )

    assert artifact["status"] == "blocked_precondition"
    assert artifact["prospective_representation_stream_ready_score"] == 0.0
    assert artifact["model_execution_receipts"] == []
    assert artifact["raw_vector_manifest"]["vectors"] == []
    assert artifact["gate_check_summary"]["all_gates_passed"] is False
    assert "dual_rtx3090_visible" in artifact["gate_check_summary"]["failed_gates"]
    assert mod.validate_artifact(artifact) == []


def test_scenario_verify_6486_substrate_is_adversarial_verify_recognized() -> None:
    """SCENARIO-VERIFY-6486-ARTIFACT: the requested substrate has a duration floor."""

    floor = av.duration_floor_for_artifact(
        {
            "inference_substrate": mod.INFERENCE_SUBSTRATE,
            "duration_s": 3.0,
            "MODEL_SPECS": [{"hf_id": mod.MANDATED_MODEL_HF_IDS[0]}],
            "random_seed": mod.RANDOM_SEED,
            "reproducibility_checksum": "sha256:" + "0" * 64,
        }
    )

    assert floor is not None
    assert floor["reason"] == "llm_embedding_extraction"
