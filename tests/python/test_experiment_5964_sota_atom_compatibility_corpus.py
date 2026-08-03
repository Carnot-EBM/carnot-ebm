"""Tests for Exp5964 SOTA atom compatibility corpus.

Spec refs: REQ-INFER-SOTA-5964, SCENARIO-INFER-SOTA-5964-BLOCKED,
SCENARIO-INFER-SOTA-5964-CORPUS, SCENARIO-INFER-SOTA-5964-CONTROLS.
"""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_5963_exact_atom_pair_fixture as exp5963
from carnot import experiment_5964_sota_atom_compatibility_corpus as mod


REPO = Path(__file__).resolve().parents[2]
LLM_SPEC = REPO / "openspec/capabilities/llm-ebm-inference/spec.md"
TEST_COMMAND = (
    ".venv/bin/pytest "
    "tests/python/test_experiment_5964_sota_atom_compatibility_corpus.py "
    "-q --no-cov -n 0"
)
COVERAGE_COMMAND = (
    ".venv/bin/coverage run --rcfile=/dev/null "
    "--include=python/carnot/experiment_5964_sota_atom_compatibility_corpus.py "
    "-m pytest tests/python/test_experiment_5964_sota_atom_compatibility_corpus.py "
    "-q --no-cov -n 0 && .venv/bin/coverage report --rcfile=/dev/null "
    "--include=python/carnot/experiment_5964_sota_atom_compatibility_corpus.py "
    "--fail-under=100"
)
FULL_TEST_COMMAND = ".venv/bin/pytest tests/python -q"
SPEC_COMMAND = (
    ".venv/bin/python scripts/check_spec_coverage.py "
    "tests/python/test_experiment_5964_sota_atom_compatibility_corpus.py"
)
ADVERSARIAL_COMMAND = (
    ".venv/bin/python scripts/adversarial_verify.py "
    "results/experiment_5964_sota_atom_compatibility_corpus.json"
)
ROOT_CLUTTER_COMMAND = ".venv/bin/python scripts/root_clutter_sweep.py"
PROTECTED_COMMAND = (
    "git status --short -- scripts/research_conductor.py ops/exclusion_manifest.yaml "
    "ops/changelog.md ops/status.md _bmad/traceability.md research-references.md"
)
TEST_COMMANDS = [
    TEST_COMMAND,
    COVERAGE_COMMAND,
    FULL_TEST_COMMAND,
    SPEC_COMMAND,
    ADVERSARIAL_COMMAND,
    ROOT_CLUTTER_COMMAND,
    PROTECTED_COMMAND,
]
TEST_EXIT_CODES = {command: 0 for command in TEST_COMMANDS}


class FakeCompatibilityBackend:
    """Deterministic output-free backend with representation-specific headroom."""

    def __init__(self, model_spec: Mapping[str, Any], config: Mapping[str, Any]) -> None:
        self.model_spec = dict(model_spec)
        self.config = dict(config)
        self.loaded = False

    def load(self) -> dict[str, Any]:
        self.loaded = True
        return {
            "loader_class": "FakeLlamaCppOutputFreeEmbeddingBackend",
            "llama_cpp_version": "fixture-llama-cpp",
            "requested_n_gpu_layers": -1,
            "requested_main_gpu": self.model_spec["gpu"],
            "observed_device_assignment": {
                "memory_delta_mb_by_gpu": {str(self.model_spec["gpu"]): 256}
            },
            "embedding_mode": True,
            "pooling_type": "LLAMA_POOLING_TYPE_LAST",
            "output_logits_enabled": False,
            "generated_text_enabled": False,
            "cuda_offload_verified": True,
        }

    def tokenize(self, text: str) -> list[int]:
        return list(range(max(1, len(text.split()))))

    def embed(self, text: str) -> list[float]:
        if not self.loaded:
            raise RuntimeError("backend not loaded")
        digest = mod.sha256_text(self.model_spec["hf_id"] + "|" + text)
        chunks = [int(digest[index : index + 4], 16) for index in range(7, 31, 4)]
        base = [((chunk % 997) / 997.0) for chunk in chunks]
        if "representation=context_then_claim_flip_atom" in text:
            base[0] += 4.0
        if "representation=atom_then_context" in text:
            base[1] += 3.0
        if "negative-visible" in text:
            base[2] += 2.0
        return [round(value, 8) for value in base]

    def close(self) -> None:
        self.loaded = False


class DegenerateBackend(FakeCompatibilityBackend):
    """Backend fixture that extracts deterministically but has no headroom."""

    def embed(self, text: str) -> list[float]:
        if not self.loaded:
            raise RuntimeError("backend not loaded")
        return [0.25, 0.25, 0.25, 0.25, 0.25, 0.25]


def _hash_row(row: dict[str, Any]) -> dict[str, Any]:
    row["row_hash"] = exp5963._row_hash(row)
    return row


def _fixture(tmp_path: Path, *, ready_score: float = 1.0) -> tuple[Path, Path, Path]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    contexts: list[dict[str, Any]] = []
    pairs: list[dict[str, Any]] = []
    previous_context = exp5963.INITIAL_PREFIX_HASH
    previous_pair = exp5963.INITIAL_PREFIX_HASH
    splits = ["train", "train", "dev", "heldout"]
    for index, split in enumerate(splits):
        context_text = f"Context {index} visible constraints only for {split}."
        context = {
            "schema": exp5963.CONTEXT_ROW_SCHEMA_VERSION,
            "sequence_index": index,
            "context_id": f"ctx-{index}",
            "semantic_instance_id": f"semantic-{index}",
            "context_view_id": "original",
            "source_row_id": f"source-{index}",
            "source_row_hash": mod.sha256_json({"source": index}),
            "source_group_id": "fixture-group",
            "family": "access_control" if index % 2 == 0 else "task_selection",
            "source_split": split,
            "variant_kind": "canonical",
            "model_visible_text": context_text,
            "model_visible_text_hash": mod.sha256_text(context_text),
            "previous_hash": previous_context,
            "row_hash": "",
        }
        _hash_row(context)
        previous_context = context["row_hash"]
        contexts.append(context)
        for label_index, visible in enumerate(("positive-visible", "negative-visible")):
            seq = len(pairs)
            candidate = {
                "atom_id": mod.sha256_json({"candidate": index, "label": label_index}),
                "atom_kind": "fact.assert",
                "payload": {
                    "args": [f"p{index}"],
                    "predicate": visible,
                    "truth": label_index == 0,
                },
                "schema_version": "carnot.constraint_ir.atomic_support.v1",
            }
            candidate_text = f"fact.assert {mod.canonical_json(candidate['payload'])}"
            pair = {
                "schema": exp5963.PAIR_ROW_SCHEMA_VERSION,
                "sequence_index": seq,
                "pair_id": f"ctx-{index}-pair-{label_index}",
                "context_id": context["context_id"],
                "semantic_instance_id": context["semantic_instance_id"],
                "context_view_id": "original",
                "source_row_id": context["source_row_id"],
                "source_group_id": context["source_group_id"],
                "family": context["family"],
                "variant_kind": context["variant_kind"],
                "candidate_atom": candidate,
                "candidate_text": candidate_text,
                "candidate_text_hash": mod.sha256_text(candidate_text),
                "label": "compatible" if label_index == 0 else "incompatible",
                "label_bool": label_index == 0,
                "python_label_bool": label_index == 0,
                "z3_label_bool": label_index == 0,
                "z3_check_status": "sat",
                "label_column_model_visible": False,
                "split_sealed_before_label": True,
                "label_opened_stage": "after_candidate_generation_and_split_seal",
                "previous_hash": previous_pair,
                "row_hash": "",
            }
            _hash_row(pair)
            previous_pair = pair["row_hash"]
            pairs.append(pair)
    context_path = tmp_path / "exp5963.contexts.jsonl"
    pair_path = tmp_path / "exp5963.pairs.jsonl"
    context_path.write_text("".join(mod.canonical_json(row) + "\n" for row in contexts))
    pair_path.write_text("".join(mod.canonical_json(row) + "\n" for row in pairs))
    artifact = {
        "schema": exp5963.ARTIFACT_SCHEMA_VERSION,
        "status": "complete_ready" if ready_score == 1.0 else "blocked",
        "pair_fixture_ready_score": ready_score,
        "row_paths_hashes_and_prefix_chain": {
            "context_rows": {
                "path": str(context_path),
                "sha256": mod.sha256_file(context_path),
                "row_count": len(contexts),
            },
            "pair_rows": {
                "path": str(pair_path),
                "sha256": mod.sha256_file(pair_path),
                "row_count": len(pairs),
            },
        },
    }
    artifact_path = tmp_path / "exp5963.json"
    artifact_path.write_text(json.dumps(artifact, sort_keys=True) + "\n", encoding="utf-8")
    return artifact_path, context_path, pair_path


def _model_specs(tmp_path: Path, *, missing_index: int | None = None) -> list[dict[str, Any]]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    specs: list[dict[str, Any]] = []
    for index, hf_id in enumerate(mod.MANDATED_MODEL_HF_IDS):
        path = tmp_path / f"model-{index}.gguf"
        if missing_index != index:
            path.write_bytes(b"GGUF" + hf_id.encode("utf-8"))
        specs.append(
            {
                "name": hf_id.rsplit("/", 1)[-1].replace("-GGUF", ""),
                "hf_id": hf_id,
                "role": "dense" if "31B" in hf_id else "moe",
                "gpu": index % 2,
                "model_path": str(path),
                "quantization": "Q4_K_M",
                "headline_eligible": True,
                "tokenizer_receipt": {
                    "source": "embedded_gguf_llama_cpp_vocab_only",
                    "loadable": missing_index != index,
                    "detail": "embedded tokenizer fixture",
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
            "version": "fixture-llama-cpp",
            "cuda_backend_available": True,
        },
        "cuda": {
            "available": True,
            "backend": "CUDA",
            "genuine_offload_required": True,
        },
        "gpu": {
            "gpu_count": 2,
            "devices": [
                {
                    "index": 0,
                    "name": "RTX 4090",
                    "memory_total_mb": 24576,
                    "memory_free_mb": 32768,
                    "temperature_c": 42,
                },
                {
                    "index": 1,
                    "name": "RTX 4090",
                    "memory_total_mb": 24576,
                    "memory_free_mb": 32768,
                    "temperature_c": 43,
                },
            ],
            "ok": True,
        },
        "resources": {
            "memory": {"available_mb": 65536, "required_mb": 16384, "ok": True},
            "disk": {"available_mb": 65536, "required_mb": 10240, "ok": True},
        },
        "output_paths": {
            "result_path": str(tmp_path / mod.RESULT_RELATIVE_PATH.name),
            "row_dir": str(tmp_path),
            "atomic_suffix": ".tmp",
            "ok": True,
        },
        "time_budget": {"estimated_required_s": 1, "available_s": 3600, "ok": True},
        "legacy_tiny_models_policy": {
            "legacy_smoke_model_ids": list(mod.LEGACY_SMOKE_MODEL_IDS),
            "smoke_only": True,
            "cannot_satisfy_readiness": True,
        },
    }


def _run_fake(
    tmp_path: Path, backend_factory: type[FakeCompatibilityBackend] = FakeCompatibilityBackend
) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
    artifact_path, context_path, pair_path = _fixture(tmp_path)
    artifact = mod.run(
        result_path=tmp_path / mod.RESULT_RELATIVE_PATH.name,
        row_dir=tmp_path,
        fixture_artifact_path=artifact_path,
        context_rows_path=context_path,
        pair_rows_path=pair_path,
        model_specs=_model_specs(tmp_path),
        preconditions_checked=_ready_preconditions(tmp_path),
        embedding_backend_factory=backend_factory,
        test_commands=TEST_COMMANDS,
        test_exit_codes=TEST_EXIT_CODES,
        write=True,
    )
    rows_by_model = {
        hf_id: mod.read_model_row_file(tmp_path / mod.model_row_relative_path(hf_id).name)
        for hf_id in mod.MANDATED_MODEL_HF_IDS
    }
    return artifact, rows_by_model


def test_req_infer_sota_5964_spec_declares_atom_compatibility_contract() -> None:
    """REQ-INFER-SOTA-5964: OpenSpec names fields, principles, models, and scenarios."""

    spec = LLM_SPEC.read_text(encoding="utf-8")
    section = spec[spec.index("### REQ-INFER-SOTA-5964") :]
    normalized = " ".join(section.split())

    for marker in (
        "REQ-INFER-SOTA-5964",
        "SCENARIO-INFER-SOTA-5964-BLOCKED",
        "SCENARIO-INFER-SOTA-5964-CORPUS",
        "SCENARIO-INFER-SOTA-5964-CONTROLS",
        mod.RESULT_RELATIVE_PATH.as_posix(),
        mod.INFERENCE_SUBSTRATE,
        "`atom_compatibility_corpus_ready_score`",
        "`llama_cpp.Llama(embedding=True, pooling_type=LAST)`",
    ):
        assert marker in section
    for hf_id in mod.MANDATED_MODEL_HF_IDS:
        assert hf_id in section
    for field, principle in mod.REQUIRED_FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert " ".join(principle.split()) in normalized


def test_scenario_infer_sota_5964_blocked_missing_model_does_not_extract(
    tmp_path: Path,
) -> None:
    """SCENARIO-INFER-SOTA-5964-BLOCKED: missing GGUF blocks before extraction."""

    artifact_path, context_path, pair_path = _fixture(tmp_path)
    calls: list[str] = []

    def forbidden_backend(
        model_spec: Mapping[str, Any], config: Mapping[str, Any]
    ) -> FakeCompatibilityBackend:
        calls.append(str(model_spec["hf_id"]))
        raise AssertionError("backend must not load when a mandated model is missing")

    artifact = mod.run(
        result_path=tmp_path / mod.RESULT_RELATIVE_PATH.name,
        row_dir=tmp_path,
        fixture_artifact_path=artifact_path,
        context_rows_path=context_path,
        pair_rows_path=pair_path,
        model_specs=_model_specs(tmp_path, missing_index=1),
        preconditions_checked=_ready_preconditions(tmp_path),
        embedding_backend_factory=forbidden_backend,
        test_commands=TEST_COMMANDS,
        test_exit_codes=TEST_EXIT_CODES,
        write=True,
    )

    assert calls == []
    assert artifact["status"] == "blocked"
    assert artifact["honest_verdict"].startswith("blocked:")
    assert artifact["atom_compatibility_corpus_ready_score"] == 0.0
    assert artifact["gate_replay_receipt"]["pair_fixture_ready_score"] == pytest.approx(1.0)
    assert "mandated_model_unavailable" in artifact["preconditions_checked"]["blocked_reasons"]
    assert mod.validate_artifact(artifact) is True
    for hf_id in mod.MANDATED_MODEL_HF_IDS:
        path = tmp_path / mod.model_row_relative_path(hf_id).name
        assert path.exists()
        assert mod.read_model_row_file(path) == []


def test_scenario_infer_sota_5964_corpus_fake_backend_emits_per_model_rows(
    tmp_path: Path,
) -> None:
    """SCENARIO-INFER-SOTA-5964-CORPUS: all family row corpora are immutable."""

    artifact, rows_by_model = _run_fake(tmp_path)

    assert mod.validate_artifact(artifact) is True
    assert json.loads((tmp_path / mod.RESULT_RELATIVE_PATH.name).read_text()) == artifact
    assert artifact["status"] == "complete_ready"
    assert artifact["honest_verdict"].startswith("complete_ready:")
    assert artifact["atom_compatibility_corpus_ready_score"] == pytest.approx(1.0)
    assert artifact["verifier_is_oracle"] is False
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["model_specs"][0]["hf_id"] == mod.MANDATED_MODEL_HF_IDS[0]
    assert artifact["model_file_hashes"]["all_mandated_files_present"] is True
    assert (
        artifact["embedded_tokenizer_and_llama_cpp_receipts"][
            "all_embedded_tokenizers_loadable"
        ]
        is True
    )
    assert artifact["cuda_offload_vram_thermal_and_cleanup_receipts"][
        "all_models_cuda_offloaded"
    ] is True
    assert artifact["raw_vs_standardized_feature_separation"][
        "raw_vectors_stored_only_in_per_model_files"
    ] is True
    assert artifact["split_and_label_secrecy_receipts"]["labels_in_prompts"] is False

    expected_representations = set(mod.REPRESENTATION_KINDS) | {mod.DUPLICATE_REPRESENTATION}
    for hf_id, rows in rows_by_model.items():
        assert rows
        assert {row["representation_kind"] for row in rows} == expected_representations
        assert all("label" not in row for row in rows)
        assert all(row["row_hash"] == mod.vector_row_hash(row) for row in rows)
        assert all(str(row["prompt_hash"]).startswith("sha256:") for row in rows)
        assert all(str(row["exp5963_pair_row_hash"]).startswith("sha256:") for row in rows)
        assert all(row["model_hf_id"] == hf_id for row in rows)
        assert mod.verify_model_row_file(
            rows,
            artifact["per_model_row_paths_hashes_counts_and_prefix_chains"]["models"][hf_id],
        )


def test_scenario_infer_sota_5964_controls_detect_tamper_and_retirement(
    tmp_path: Path,
) -> None:
    """SCENARIO-INFER-SOTA-5964-CONTROLS: controls gate readiness, not accuracy."""

    artifact, rows_by_model = _run_fake(tmp_path)

    assert artifact["finite_variance_duplicate_and_order_controls"]["all_families_pass"] is True
    assert artifact["claim_flip_and_per_family_headroom_controls"]["families_passing_count"] == 3
    shortcuts = artifact[
        "norm_length_frequency_label_pair_permutation_and_model_identity_controls"
    ]
    assert shortcuts["all_controls_measured"] is True
    assert shortcuts["raw_model_identity"]["model_identity_preserved_as_stratum"] is True

    tampered = deepcopy(artifact)
    tampered["test_exit_codes"][TEST_COMMAND] = 1
    assert mod.atom_compatibility_corpus_ready_score(tampered) == 0.0
    assert "failed_test_exit_codes" in mod._blocked_reasons(tampered)
    with pytest.raises(ValueError, match="atom_compatibility_corpus_ready_score"):
        mod.validate_artifact(tampered)

    row = deepcopy(next(iter(rows_by_model.values()))[0])
    row["embedding"][0] += 1.0
    with pytest.raises(ValueError, match="row_hash"):
        mod.verify_model_row_file(
            [row],
            {
                "path": "unused",
                "row_count": 1,
                "sha256": mod.sha256_text(mod.rows_to_jsonl([row])),
                "row_hashes": {row["vector_row_id"]: row["row_hash"]},
            },
        )

    retired, _rows = _run_fake(tmp_path / "degenerate", DegenerateBackend)
    assert retired["status"] == "retired"
    assert retired["honest_verdict"].startswith("retired:")
    assert retired["atom_compatibility_corpus_ready_score"] == 0.0
    assert retired["claim_flip_and_per_family_headroom_controls"]["families_passing_count"] == 0
    assert mod.validate_artifact(retired) is True


def test_scenario_infer_sota_5964_defensive_helper_branches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-INFER-SOTA-5964: corrupt gates, rows, specs, and prompts fail closed."""

    artifact_path, context_path, pair_path = _fixture(tmp_path, ready_score=0.0)
    artifact = mod.run(
        result_path=tmp_path / "blocked-gate.json",
        row_dir=tmp_path / "blocked-gate-rows",
        fixture_artifact_path=artifact_path,
        context_rows_path=context_path,
        pair_rows_path=pair_path,
        model_specs=_model_specs(tmp_path),
        preconditions_checked=_ready_preconditions(tmp_path),
        embedding_backend_factory=FakeCompatibilityBackend,
        test_commands=TEST_COMMANDS,
        test_exit_codes=TEST_EXIT_CODES,
        write=True,
    )
    assert artifact["status"] == "blocked"
    assert "exp5963_gate_not_ready" in artifact["preconditions_checked"]["blocked_reasons"]

    bad_json = tmp_path / "bad.json"
    bad_json.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="JSON object required"):
        mod._read_json(bad_json)
    missing_jsonl = tmp_path / "missing.jsonl"
    assert mod._read_jsonl(missing_jsonl) == []
    blank_jsonl = tmp_path / "blank.jsonl"
    blank_jsonl.write_text("\n", encoding="utf-8")
    assert mod._read_jsonl(blank_jsonl) == []
    bad_jsonl = tmp_path / "bad.jsonl"
    bad_jsonl.write_text("[]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="JSONL object required"):
        mod._read_jsonl(bad_jsonl)

    assert mod.model_row_relative_path("example/custom-GGUF").name.endswith("custom.rows.jsonl")
    assert mod.claim_flip_atom_text({"payload": {"truth": True}, "atom_kind": "fact.assert"})
    assert mod.claim_flip_atom_text({"payload": {"x": 1}, "atom_kind": "rule.body.atom"}).startswith(
        "claim.flip"
    )
    assert mod._auc([1, 1], [True, False]) == pytest.approx(0.5)
    assert mod._mean([]) == pytest.approx(0.0)
    assert mod._variance([]) == pytest.approx(0.0)
    assert mod._l2([1, 2], [1]) == pytest.approx(0.0)
    with pytest.raises(ValueError, match="nonfinite_embedding"):
        mod._round_embedding([1.0, float("nan")])
    assert mod._receipt_has_cuda_offload(
        {"observed_device_assignment": {"memory_delta_mb_by_gpu": {"0": 4}}}
    )
    assert mod._standardized_vector(
        {"model_hf_id": "missing", "embedding": [1.0], "representation_kind": "context_alone"},
        {},
    ) == [1.0]
    sparse_headroom = mod.claim_flip_and_per_family_headroom_controls(
        {
            mod.MANDATED_MODEL_HF_IDS[0]: [
                {
                    "exp5963_pair_sequence_index": 0,
                    "representation_kind": "context_then_atom",
                    "embedding": [0.0],
                    "model_hf_id": mod.MANDATED_MODEL_HF_IDS[0],
                }
            ]
        },
        {},
    )
    assert sparse_headroom["families"][mod.MANDATED_MODEL_HF_IDS[0]]["passed"] is False
    output_probe = mod._output_path_receipt(tmp_path / "out.json", tmp_path)
    assert output_probe["ok"] is True

    monkeypatch.setattr(mod, "gguf_tokenizer_loadable", lambda path: (True, "ok"))
    specs = []
    for hf_id in mod.MANDATED_MODEL_HF_IDS:
        path = tmp_path / f"{hf_id.split('/')[-1]}.gguf"
        path.write_bytes(b"GGUF")
        specs.append({"hf_id": hf_id, "model_path": str(path), "gpu": 0})
    assert all(
        spec["tokenizer_receipt"]["loadable"] for spec in mod.normalize_model_specs(specs)
    )
    missing_specs = mod.normalize_model_specs([{"hf_id": mod.MANDATED_MODEL_HF_IDS[0]}])
    assert missing_specs[0]["tokenizer_receipt"]["source"] == "missing_model_path"

    missing_gate = mod.gate_replay_receipt(
        fixture_artifact_path=tmp_path / "no-artifact.json",
        context_rows_path=tmp_path / "no-contexts.jsonl",
        pair_rows_path=tmp_path / "no-pairs.jsonl",
    )
    assert missing_gate["ready"] is False
    assert missing_gate["artifact_error"]

    bad_preconditions = {
        "preconditions_ready": False,
        "blocked_reasons": ["manual"],
        "llama_cpp": {"available": False, "cuda_backend_available": False},
        "cuda": {"available": False},
        "gpu": {"ok": False},
        "resources": {"memory": {"ok": False}, "disk": {"ok": False}},
        "output_paths": {"ok": False},
        "time_budget": {"ok": False},
        "legacy_tiny_models_policy": {},
    }
    bad_blockers = mod._precondition_blockers(
        bad_preconditions,
        list(reversed(missing_specs)),
        {"ready": False},
    )
    assert {
        "manual",
        "preconditions_not_ready",
        "exp5963_gate_not_ready",
        "mandated_model_order_mismatch",
        "mandated_model_unavailable",
        "llama_cpp_unavailable",
        "llama_cpp_cuda_backend_unavailable",
        "cuda_offload_unavailable",
        "gpu_device_receipt_unavailable",
        "insufficient_free_vram",
        "insufficient_free_ram",
        "insufficient_free_disk",
        "output_path_not_writable",
        "time_budget_unavailable",
        "legacy_smoke_policy_missing",
    }.issubset(set(bad_blockers))
    no_min_vram_specs = [
        {**spec, "min_vram_gb": None} for spec in mod.normalize_model_specs(specs)
    ]
    assert (
        "insufficient_free_vram"
        not in mod._precondition_blockers(
            _ready_preconditions(tmp_path),
            no_min_vram_specs,
            {"ready": True},
        )
    )


def test_scenario_infer_sota_5964_validation_error_branches(tmp_path: Path) -> None:
    """REQ-INFER-SOTA-5964: row and artifact validators reject every gate drift."""

    artifact, rows_by_model = _run_fake(tmp_path)
    row = deepcopy(next(iter(rows_by_model.values()))[0])
    valid_receipt = {
        "path": "unused",
        "row_count": 1,
        "sha256": mod.sha256_text(mod.rows_to_jsonl([row])),
        "row_hashes": {row["vector_row_id"]: row["row_hash"]},
        "final_prefix_checksum": row["row_hash"],
    }
    assert mod.verify_model_row_file([row], valid_receipt)

    for field, value, message in (
        ("sequence_index", 7, "sequence_index"),
        ("previous_hash", "sha256:bad", "previous_hash"),
    ):
        bad_row = deepcopy(row)
        bad_row[field] = value
        with pytest.raises(ValueError, match=message):
            mod.verify_model_row_file([bad_row], valid_receipt)

    for bad_receipt, message in (
        ({**valid_receipt, "row_count": 2}, "row_count"),
        ({**valid_receipt, "row_hashes": {}}, "row_hashes"),
        ({**valid_receipt, "sha256": "sha256:bad"}, "row_file_sha256"),
        ({**valid_receipt, "final_prefix_checksum": "sha256:bad"}, "final_prefix_checksum"),
    ):
        with pytest.raises(ValueError, match=message):
            mod.verify_model_row_file([row], bad_receipt)

    retired, _rows = _run_fake(tmp_path / "retired", DegenerateBackend)
    assert "claim_flip_and_per_family_headroom_controls" in mod._blocked_reasons(retired)

    malformed_cases = []
    missing = deepcopy(artifact)
    missing.pop("status")
    malformed_cases.append((missing, "missing required fields"))
    bad_provenance = deepcopy(artifact)
    bad_provenance["field_provenance"] = {}
    malformed_cases.append((bad_provenance, "field_provenance"))
    bad_substrate = deepcopy(artifact)
    bad_substrate["inference_substrate"] = "cpu"
    bad_substrate["atom_compatibility_corpus_ready_score"] = 0.0
    bad_substrate["reproducibility_checksum"] = mod.reproducibility_checksum(bad_substrate)
    malformed_cases.append((bad_substrate, "inference_substrate"))
    bad_oracle = deepcopy(artifact)
    bad_oracle["verifier_is_oracle"] = True
    bad_oracle["atom_compatibility_corpus_ready_score"] = 0.0
    bad_oracle["reproducibility_checksum"] = mod.reproducibility_checksum(bad_oracle)
    malformed_cases.append((bad_oracle, "verifier_is_oracle"))
    bad_models = deepcopy(artifact)
    bad_models["model_specs"][0]["hf_id"] = "wrong"
    bad_models["atom_compatibility_corpus_ready_score"] = 0.0
    bad_models["reproducibility_checksum"] = mod.reproducibility_checksum(bad_models)
    malformed_cases.append((bad_models, "model_specs"))
    bad_checksum = deepcopy(artifact)
    bad_checksum["reproducibility_checksum"] = "sha256:bad"
    malformed_cases.append((bad_checksum, "reproducibility_checksum"))
    bad_complete = deepcopy(artifact)
    bad_complete["honest_verdict"] = "blocked: wrong"
    bad_complete["reproducibility_checksum"] = mod.reproducibility_checksum(bad_complete)
    malformed_cases.append((bad_complete, "honest_verdict"))
    bad_retired = deepcopy(retired)
    bad_retired["honest_verdict"] = "blocked: wrong"
    bad_retired["reproducibility_checksum"] = mod.reproducibility_checksum(bad_retired)
    malformed_cases.append((bad_retired, "honest_verdict"))
    blocked = deepcopy(retired)
    blocked["status"] = "blocked"
    blocked["honest_verdict"] = "retired: wrong"
    blocked["preconditions_checked"]["blocked_reasons"] = ["manual"]
    blocked["reproducibility_checksum"] = mod.reproducibility_checksum(blocked)
    malformed_cases.append((blocked, "honest_verdict"))
    weird = deepcopy(retired)
    weird["status"] = "partial"
    weird["reproducibility_checksum"] = mod.reproducibility_checksum(weird)
    malformed_cases.append((weird, "status"))

    for bad_artifact, message in malformed_cases:
        with pytest.raises(ValueError, match=message):
            mod.validate_artifact(bad_artifact)

    artifact_path, _context_path, _pair_path = _fixture(tmp_path / "missing-rows")
    missing_rows_artifact = mod.run(
        result_path=tmp_path / "missing-rows.json",
        row_dir=tmp_path / "missing-rows-output",
        fixture_artifact_path=artifact_path,
        context_rows_path=tmp_path / "missing-contexts.jsonl",
        pair_rows_path=tmp_path / "missing-pairs.jsonl",
        model_specs=_model_specs(tmp_path / "missing-models"),
        preconditions_checked=_ready_preconditions(tmp_path),
        embedding_backend_factory=FakeCompatibilityBackend,
        test_commands=TEST_COMMANDS,
        test_exit_codes=TEST_EXIT_CODES,
        write=False,
    )
    assert "exp5963_rows_unavailable" in missing_rows_artifact["preconditions_checked"][
        "blocked_reasons"
    ]
