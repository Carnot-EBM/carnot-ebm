"""Tests for Exp5852 three-family paired GGUF embeddings.

Spec refs: REQ-VERIFY-5852, SCENARIO-VERIFY-5852-COMPLETE,
SCENARIO-VERIFY-5852-PARITY, SCENARIO-VERIFY-5852-RESUME,
SCENARIO-VERIFY-5852-BLOCKED.
"""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_5852_three_family_paired_embeddings as mod


REPO = Path(__file__).resolve().parents[2]
VERIFY_SPEC = REPO / "openspec/capabilities/verification/spec.md"
TEST_COMMAND = (
    ".venv/bin/pytest "
    "tests/python/test_experiment_5852_three_family_paired_embeddings.py "
    "-q --no-cov -n 0"
)
COVERAGE_COMMAND = (
    ".venv/bin/coverage run --rcfile=/dev/null "
    "--include=python/carnot/experiment_5852_three_family_paired_embeddings.py "
    "-m pytest tests/python/test_experiment_5852_three_family_paired_embeddings.py "
    "-q --no-cov -n 0 && .venv/bin/coverage report --rcfile=/dev/null "
    "--include=python/carnot/experiment_5852_three_family_paired_embeddings.py "
    "--fail-under=100"
)
FULL_TEST_COMMAND = ".venv/bin/pytest tests/python -q"
SPEC_COMMAND = ".venv/bin/python scripts/check_spec_coverage.py"
ADVERSARIAL_COMMAND = (
    ".venv/bin/python scripts/adversarial_verify.py "
    "results/experiment_5852_three_family_paired_embeddings.json"
)
ROOT_CLUTTER_COMMAND = ".venv/bin/python scripts/root_clutter_sweep.py"
PROTECTED_FILE_COMMAND = (
    '.venv/bin/python -c "from pathlib import Path; '
    "assert Path('scripts/research_conductor.py').exists()\""
)
TEST_COMMANDS = [
    TEST_COMMAND,
    COVERAGE_COMMAND,
    FULL_TEST_COMMAND,
    SPEC_COMMAND,
    ADVERSARIAL_COMMAND,
    ROOT_CLUTTER_COMMAND,
    PROTECTED_FILE_COMMAND,
]
TEST_EXIT_CODES = {command: 0 for command in TEST_COMMANDS}


class FakeEmbeddingBackend:
    """Deterministic output-free backend used by the Exp5852 unit tests."""

    def __init__(self, model_spec: Mapping[str, Any], config: Mapping[str, Any]) -> None:
        self.model_spec = dict(model_spec)
        self.config = dict(config)
        self.loaded = False

    def load(self) -> dict[str, Any]:
        self.loaded = True
        return {
            "loader_class": "FakeGemma4QuantizedLoader",
            "llama_cpp_version": "fixture-llama-cpp",
            "requested_n_gpu_layers": -1,
            "requested_main_gpu": self.model_spec["gpu"],
            "observed_device_assignment": {"gpu": self.model_spec["gpu"], "memory_delta_mb": 64},
            "embedding_mode": True,
            "output_logits_enabled": False,
            "generated_text_enabled": False,
        }

    def tokenize(self, text: str) -> list[int]:
        return list(range(len(text.split())))

    def embed(self, text: str) -> list[float]:
        if not self.loaded:
            raise RuntimeError("backend not loaded")
        digest = mod.sha256_text(self.model_spec["hf_id"] + "|" + text)
        chunks = [int(digest[index : index + 4], 16) for index in range(7, 27, 4)]
        token_count = len(self.tokenize(text))
        return [
            round((chunk % 997) / 997.0 + token_count * 0.001 + dim * 0.01, 8)
            for dim, chunk in enumerate(chunks)
        ]

    def close(self) -> None:
        self.loaded = False


class NoProgressBackend(FakeEmbeddingBackend):
    """Tokenizer fixture where padding can never increase token count."""

    def tokenize(self, text: str) -> list[int]:
        return [1]


class RaggedEmbeddingBackend(FakeEmbeddingBackend):
    """Embedding fixture that returns mismatched paired vector widths."""

    def embed(self, text: str) -> list[float]:
        base = super().embed(text)
        return base[:1] if "wrong" in text or "ablated" in text else base


def _source_row(
    *,
    row_id: str,
    axis: str,
    family: str,
    left_text: str,
    right_text: str,
) -> dict[str, Any]:
    labels = [True, False] if axis == "candidate_correctness" else [False, True]
    row: dict[str, Any] = {
        "schema": "carnot.experiment_5840.exact_counterfactual_embedding_fixture.v1.row",
        "row_id": row_id,
        "pair_id": row_id,
        "pair_group_id": f"group-{row_id}",
        "bootstrap_unit_id": mod.sha256_json({"bootstrap": row_id}),
        "split": "science" if "planning" in family else "train",
        "axis": axis,
        "family": family,
        "change": "addition",
        "surface_kind": "symbol_relabel",
        "solver_effort_bin": "low",
        "source_provenance": {"exp5826_row_id": row_id, "exp5826_row_hash": "source"},
        "feature_consumer_view": {"masked_pair_hash": mod.sha256_json({"pair_id": row_id})},
        "conditions": [
            {
                "condition_id": f"{row_id}-a",
                "condition_suffix": "a",
                "model_input": left_text,
                "model_input_hash": mod.sha256_text(left_text),
                "token_count": len(left_text.split()),
                "candidate_hash": mod.sha256_json({"candidate": row_id, "side": "a"}),
                "context_hash": mod.sha256_json({"context": row_id, "side": "a"}),
                "constraint_present": True,
                "exact_label": labels[0],
            },
            {
                "condition_id": f"{row_id}-b",
                "condition_suffix": "b",
                "model_input": right_text,
                "model_input_hash": mod.sha256_text(right_text),
                "token_count": len(right_text.split()),
                "candidate_hash": mod.sha256_json({"candidate": row_id, "side": "b"}),
                "context_hash": mod.sha256_json({"context": row_id, "side": "b"}),
                "constraint_present": axis == "candidate_correctness",
                "exact_label": labels[1],
            },
        ],
        "row_hash": "",
    }
    if axis == "candidate_correctness":
        row["conditions"][1]["context_hash"] = row["conditions"][0]["context_hash"]
    else:
        row["conditions"][1]["candidate_hash"] = row["conditions"][0]["candidate_hash"]
    row["row_hash"] = mod.source_row_hash(row)
    return row


def _write_fixture(tmp_path: Path) -> tuple[Path, Path, list[dict[str, Any]]]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    rows = [
        _source_row(
            row_id="exp5840-candidate-correctness-fixture-0000",
            axis="candidate_correctness",
            family="finite_domain_csp",
            left_text="case neutral same envelope correct side",
            right_text="case neutral same envelope wrong",
        ),
        _source_row(
            row_id="exp5840-constraint-ablation-fixture-0001",
            axis="constraint_ablation",
            family="finite_state_planning",
            left_text="case neutral present relation longer side",
            right_text="case neutral ablated side",
        ),
    ]
    row_text = mod.source_rows_to_jsonl(rows)
    row_path = tmp_path / "exp5840.rows.jsonl"
    row_path.write_text(row_text, encoding="utf-8")
    row_hashes = {row["row_id"]: row["row_hash"] for row in rows}
    artifact = {
        "status": "complete",
        "counterfactual_fixture_ready_score": 1.0,
        "split_definition_and_hashes": {"row_split_hashes": {"train": "x", "science": "y"}},
        "exact_label_and_minimality_receipts": {"validator_versions": ["primary", "independent"]},
        "row_file_receipt": {
            "path": "results/experiment_5840_exact_counterfactual_embedding_fixture.rows.jsonl",
            "row_count": len(rows),
            "sha256": mod.sha256_text(row_text),
            "row_hashes": row_hashes,
            "row_hash_root": mod.sha256_json(row_hashes),
        },
    }
    artifact_path = tmp_path / "exp5840.json"
    artifact_path.write_text(json.dumps(artifact, sort_keys=True) + "\n", encoding="utf-8")
    return artifact_path, row_path, rows


def _model_specs(tmp_path: Path, *, missing_index: int | None = None) -> list[dict[str, Any]]:
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
        "gpu": {
            "gpu_count": 2,
            "devices": [
                {"index": 0, "name": "RTX 3090", "memory_total_mb": 24576, "memory_free_mb": 24000},
                {"index": 1, "name": "RTX 3090", "memory_total_mb": 24576, "memory_free_mb": 24000},
            ],
            "ok": True,
        },
        "resources": {
            "memory": {"available_mb": 8192, "required_mb": 1024, "ok": True},
            "disk": {"available_mb": 8192, "required_mb": 1024, "ok": True},
        },
        "output_paths": {
            "result_path": str(tmp_path / mod.RESULT_RELATIVE_PATH.name),
            "row_file_path": str(tmp_path / mod.ROW_FILE_RELATIVE_PATH.name),
            "checkpoint_dir": str(tmp_path / "checkpoints"),
            "ok": True,
        },
        "legacy_tiny_models_policy": {
            "smoke_only": True,
            "cannot_satisfy_readiness": True,
        },
    }


def _run_fake(tmp_path: Path) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    fixture_artifact, fixture_rows, source_rows = _write_fixture(tmp_path)
    result_path = tmp_path / mod.RESULT_RELATIVE_PATH.name
    row_file_path = tmp_path / mod.ROW_FILE_RELATIVE_PATH.name
    artifact = mod.run(
        result_path=result_path,
        row_file_path=row_file_path,
        checkpoint_dir=tmp_path / "checkpoints",
        fixture_artifact_path=fixture_artifact,
        fixture_rows_path=fixture_rows,
        model_specs=_model_specs(tmp_path),
        preconditions_checked=_ready_preconditions(tmp_path),
        embedding_backend_factory=FakeEmbeddingBackend,
        test_commands=TEST_COMMANDS,
        test_exit_codes=TEST_EXIT_CODES,
        write=True,
    )
    rows = mod.read_row_file(row_file_path)
    return artifact, rows, source_rows


def test_req_verify_5852_spec_declares_three_family_embedding_contract() -> None:
    """REQ-VERIFY-5852: OpenSpec names fields, principles, models, and scenarios."""

    spec = VERIFY_SPEC.read_text(encoding="utf-8")
    section = spec[spec.index("### REQ-VERIFY-5852") :]
    normalized = " ".join(section.split())

    for marker in (
        "REQ-VERIFY-5852",
        "SCENARIO-VERIFY-5852-COMPLETE",
        "SCENARIO-VERIFY-5852-PARITY",
        "SCENARIO-VERIFY-5852-RESUME",
        "SCENARIO-VERIFY-5852-BLOCKED",
        mod.RESULT_RELATIVE_PATH.as_posix(),
        mod.ROW_FILE_RELATIVE_PATH.as_posix(),
        mod.INFERENCE_SUBSTRATE,
        "`paired_embedding_corpus_ready_score`",
    ):
        assert marker in section
    for hf_id in mod.MANDATED_MODEL_HF_IDS:
        assert hf_id in section
    for field, principle in mod.REQUIRED_FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert " ".join(principle.split()) in normalized


def test_scenario_verify_5852_complete_fake_backend_emits_aligned_model_rows(
    tmp_path: Path,
) -> None:
    """SCENARIO-VERIFY-5852-COMPLETE: all mandated model-row cells are complete."""

    artifact, rows, source_rows = _run_fake(tmp_path)

    assert mod.validate_artifact(artifact) is True
    assert mod.verify_row_file(rows, artifact) is True
    assert json.loads((tmp_path / mod.RESULT_RELATIVE_PATH.name).read_text()) == artifact
    assert artifact["status"] == "complete"
    assert artifact["honest_verdict"].startswith("ready:")
    assert artifact["paired_embedding_corpus_ready_score"] == pytest.approx(1.0)
    assert isinstance(artifact["paired_embedding_corpus_ready_score"], float)
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["verifier_is_oracle"] is True
    assert artifact["models_used"] == list(mod.MANDATED_MODEL_HF_IDS)
    assert [row["hf_id"] for row in artifact["model_specs"]] == list(mod.MANDATED_MODEL_HF_IDS)
    assert artifact["row_file_receipt"]["row_count"] == len(source_rows) * len(
        mod.MANDATED_MODEL_HF_IDS
    )
    assert artifact["model_axis_family_cell_counts"]["all_cells_complete"] is True
    assert artifact["embedding_shape_and_finiteness"]["all_finite"] is True
    assert (
        artifact["embedding_shape_and_finiteness"]["constant_dimensions_after_preprocessing"] == {}
    )
    assert artifact["pair_alignment_receipts"]["row_order_exact"] is True
    assert artifact["pair_alignment_receipts"]["duplicate_model_row_cells"] == []
    assert artifact["token_and_truncation_parity"]["all_pairs_token_count_matched"] is True
    assert artifact["token_and_truncation_parity"]["truncation_asymmetry_count"] == 0
    assert all(code == 0 for code in artifact["test_exit_codes"].values())

    expected_order = [
        (source["row_id"], hf_id) for source in source_rows for hf_id in mod.MANDATED_MODEL_HF_IDS
    ]
    assert [(row["source_row_id"], row["model_hf_id"]) for row in rows] == expected_order

    for row in rows:
        assert row["row_hash"] == mod.row_hash(row)
        assert row["embedding_cell_id"] == f"{row['source_row_id']}|{row['model_hf_id']}"
        features = row["feature_consumer_view"]
        forbidden_feature_keys = {
            "model_hf_id",
            "model_family",
            "family",
            "exact_label",
            "label",
            "oracle",
        }
        assert forbidden_feature_keys.isdisjoint(features)
        left, right = row["condition_embeddings"]
        assert left["token_count"] == right["token_count"]
        assert left["truncated"] is False
        assert right["truncated"] is False
        expected_difference = [
            pytest.approx(b - a) for a, b in zip(left["embedding"], right["embedding"], strict=True)
        ]
        assert row["paired_difference"] == expected_difference
        assert features["paired_difference"] == expected_difference


def test_scenario_verify_5852_parity_blocks_feature_shortcuts_and_tampering(
    tmp_path: Path,
) -> None:
    """SCENARIO-VERIFY-5852-PARITY: bad tokens, rows, or tests cannot be ready."""

    artifact, rows, _source_rows = _run_fake(tmp_path)

    tampered_exit = deepcopy(artifact)
    tampered_exit["test_exit_codes"][TEST_COMMAND] = 1
    assert mod.paired_embedding_corpus_ready_score(tampered_exit) == 0.0
    assert "failed_test_exit_codes" in mod._blocked_reasons(tampered_exit)
    with pytest.raises(ValueError, match="paired_embedding_corpus_ready_score"):
        mod.validate_artifact(tampered_exit)

    tampered_row = deepcopy(rows[0])
    tampered_row["feature_consumer_view"]["model_hf_id"] = rows[0]["model_hf_id"]
    checks = mod.feature_leakage_checks([tampered_row])
    assert checks["all_checks_passed"] is False
    assert checks["feature_identity_leakage_count"] == 1

    broken_token_row = deepcopy(rows[0])
    broken_token_row["condition_embeddings"][0]["token_count"] += 1
    parity = mod.token_and_truncation_parity([broken_token_row])
    assert parity["all_pairs_token_count_matched"] is False

    duplicate_rows = [*rows, deepcopy(rows[0])]
    alignment = mod.pair_alignment_receipts(
        duplicate_rows,
        source_rows=mod.load_fixture_rows(tmp_path / "exp5840.rows.jsonl"),
        model_hf_ids=mod.MANDATED_MODEL_HF_IDS,
    )
    assert alignment["duplicate_model_row_cells"] == [rows[0]["embedding_cell_id"]]


def test_scenario_verify_5852_resume_refuses_mixed_versions(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-5852-RESUME: checkpoint continuation is hash-bound."""

    artifact, rows, _source_rows = _run_fake(tmp_path)
    model_spec = artifact["model_specs"][0]
    row_group = artifact["checkpoint_resume_receipts"]["checkpoint_groups"][0]
    payload = mod.checkpoint_payload(
        model_spec=model_spec,
        row_group_id=row_group["row_group_id"],
        rows=[row for row in rows if row["model_hf_id"] == model_spec["hf_id"]],
        input_receipt=artifact["checkpoint_resume_receipts"]["input_receipt"],
    )
    accepted = mod.validate_checkpoint_payload(
        payload,
        model_spec=model_spec,
        row_group_id=row_group["row_group_id"],
        input_receipt=artifact["checkpoint_resume_receipts"]["input_receipt"],
    )
    stale = deepcopy(payload)
    stale["input_receipt_hash"] = mod.sha256_json({"stale": True})
    refused = mod.validate_checkpoint_payload(
        stale,
        model_spec=model_spec,
        row_group_id=row_group["row_group_id"],
        input_receipt=artifact["checkpoint_resume_receipts"]["input_receipt"],
    )

    assert accepted["accepted"] is True
    assert accepted["refusal_reasons"] == []
    assert refused["accepted"] is False
    assert "input_receipt_hash_mismatch" in refused["refusal_reasons"]


def test_scenario_verify_5852_blocked_missing_mandated_model_does_not_extract(
    tmp_path: Path,
) -> None:
    """SCENARIO-VERIFY-5852-BLOCKED: missing SOTA GGUF writes terminal blocked artifact."""

    fixture_artifact, fixture_rows, _source_rows = _write_fixture(tmp_path)
    calls: list[str] = []

    def forbidden_backend(
        model_spec: Mapping[str, Any], config: Mapping[str, Any]
    ) -> FakeEmbeddingBackend:
        calls.append(str(model_spec["hf_id"]))
        raise AssertionError("backend must not load when a mandated model is missing")

    artifact = mod.run(
        result_path=tmp_path / mod.RESULT_RELATIVE_PATH.name,
        row_file_path=tmp_path / mod.ROW_FILE_RELATIVE_PATH.name,
        checkpoint_dir=tmp_path / "checkpoints",
        fixture_artifact_path=fixture_artifact,
        fixture_rows_path=fixture_rows,
        model_specs=_model_specs(tmp_path, missing_index=2),
        preconditions_checked=_ready_preconditions(tmp_path),
        embedding_backend_factory=forbidden_backend,
        test_commands=TEST_COMMANDS,
        test_exit_codes=TEST_EXIT_CODES,
        write=True,
    )

    assert calls == []
    assert artifact["status"] == "blocked"
    assert artifact["honest_verdict"].startswith("blocked:")
    assert artifact["paired_embedding_corpus_ready_score"] == 0.0
    assert "mandated_model_unavailable" in artifact["preconditions_checked"]["blocked_reasons"]
    assert artifact["row_file_receipt"]["row_count"] == 0
    assert mod.read_row_file(tmp_path / mod.ROW_FILE_RELATIVE_PATH.name) == []
    assert mod.validate_artifact(artifact) is True


def test_scenario_verify_5852_defensive_validation_branches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-VERIFY-5852-BLOCKED/RESUME/PARITY: corrupt inputs fail closed."""

    artifact, rows, source_rows = _run_fake(tmp_path)

    assert mod.model_family("example/custom-GGUF") == "custom"
    assert mod._read_jsonl(tmp_path / "missing.jsonl") == []
    blank = tmp_path / "blank.jsonl"
    blank.write_text("\n", encoding="utf-8")
    assert mod._read_jsonl(blank) == []
    bad_json = tmp_path / "bad.json"
    bad_json.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="JSON object required"):
        mod._read_json(bad_json)
    bad_jsonl = tmp_path / "bad.jsonl"
    bad_jsonl.write_text("[]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="JSONL object required"):
        mod._read_jsonl(bad_jsonl)

    fixture_artifact, fixture_rows, fixture_source = _write_fixture(tmp_path / "defensive")
    stale_source = deepcopy(fixture_source[0])
    stale_source["row_hash"] = "sha256:stale"
    stale_rows_path = tmp_path / "stale.rows.jsonl"
    stale_rows_path.write_text(mod.source_rows_to_jsonl([stale_source]), encoding="utf-8")
    with pytest.raises(ValueError, match="exp5840_row_hash"):
        mod.load_fixture_rows(stale_rows_path)
    not_ready = tmp_path / "not-ready.json"
    not_ready.write_text(
        json.dumps({"counterfactual_fixture_ready_score": 0.0}) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="exp5840_counterfactual_fixture_not_ready"):
        mod.load_fixture_artifact(not_ready)

    monkeypatch.setattr(
        mod,
        "gguf_tokenizer_loadable",
        lambda model_path: (True, f"tokenizer ok for {model_path}"),
    )
    specs_without_receipt = []
    for index, hf_id in enumerate(mod.MANDATED_MODEL_HF_IDS):
        path = tmp_path / f"no-tokenizer-{index}.gguf"
        path.write_bytes(b"GGUF")
        specs_without_receipt.append({"hf_id": hf_id, "model_path": str(path), "gpu": 0})
    normalized = mod.normalize_model_specs(specs_without_receipt)
    assert all(spec["tokenizer_receipt"]["loadable"] is True for spec in normalized)
    with pytest.raises(ValueError, match="nonfinite_embedding"):
        mod._round_embedding([1.0, float("nan")])
    padded, count, added, ok = mod._pad_to_parity(
        text="x",
        target_count=3,
        backend=NoProgressBackend(
            artifact["model_specs"][0],
            artifact["deterministic_embedding_config"],
        ),
    )
    assert (padded, count, added, ok) == ("x", 1, [], False)

    blockers = mod._precondition_blockers(
        {
            "preconditions_ready": False,
            "blocked_reasons": ["manual"],
            "gpu": {"gpu_count": 0},
            "output_paths": {"ok": False},
            "legacy_tiny_models_policy": {},
        },
        list(reversed(artifact["model_specs"])),
    )
    assert {
        "manual",
        "preconditions_not_ready",
        "mandated_model_order_mismatch",
        "gpu_device_receipt_unavailable",
        "output_path_not_writable",
        "legacy_smoke_policy_missing",
    }.issubset(set(blockers))

    model_spec = artifact["model_specs"][0]
    input_receipt = artifact["checkpoint_resume_receipts"]["input_receipt"]
    malformed = {
        "schema": "wrong",
        "model_hf_id": "wrong",
        "model_file_sha256": "wrong",
        "tokenizer_receipt_hash": "wrong",
        "row_group_id": "wrong",
        "input_receipt_hash": "wrong",
        "row_count": 99,
        "row_hashes": {},
        "rows": [deepcopy(rows[0])],
    }
    malformed["rows"][0]["row_hash"] = "sha256:bad"
    refused = mod.validate_checkpoint_payload(
        malformed,
        model_spec=model_spec,
        row_group_id="expected-group",
        input_receipt=input_receipt,
    )
    assert refused["accepted"] is False
    assert {
        "schema",
        "model_hf_id_mismatch",
        "model_file_hash_mismatch",
        "tokenizer_receipt_hash_mismatch",
        "row_group_id_mismatch",
        "input_receipt_hash_mismatch",
        "row_count_mismatch",
        "row_hashes_mismatch",
        "row_hash_mismatch",
    }.issubset(set(refused["refusal_reasons"]))
    checkpoint_path = tmp_path / "checkpoint.json"
    checkpoint_path.write_text(json.dumps(malformed), encoding="utf-8")
    assert mod._load_checkpoint(checkpoint_path)["schema"] == "wrong"
    stale_checkpoint_dir = tmp_path / "stale-checkpoints"
    group_id = "rows-000000-000001"
    stale_path = mod._checkpoint_path(stale_checkpoint_dir, model_spec, group_id)
    stale_path.parent.mkdir(parents=True, exist_ok=True)
    stale_path.write_text(json.dumps(malformed), encoding="utf-8")
    extracted, stale_receipt = mod._extract_model_rows(
        model_spec=model_spec,
        source_rows=source_rows,
        config=artifact["deterministic_embedding_config"],
        checkpoint_dir=stale_checkpoint_dir,
        input_receipt=input_receipt,
        embedding_backend_factory=FakeEmbeddingBackend,
    )
    assert extracted == []
    assert stale_receipt["checkpoint_refused"] is True

    # A second run over the same checkpoint directory exercises accepted resume.
    resumed = mod.run(
        result_path=tmp_path / "resumed.json",
        row_file_path=tmp_path / "resumed.rows.jsonl",
        checkpoint_dir=tmp_path / "checkpoints",
        fixture_artifact_path=fixture_artifact,
        fixture_rows_path=fixture_rows,
        model_specs=_model_specs(tmp_path),
        preconditions_checked=_ready_preconditions(tmp_path),
        embedding_backend_factory=FakeEmbeddingBackend,
        test_commands=TEST_COMMANDS,
        test_exit_codes=TEST_EXIT_CODES,
        write=True,
    )
    assert any(
        group["resume_attempted"] is True
        for group in resumed["checkpoint_resume_receipts"]["checkpoint_groups"]
    )

    bad_shape = deepcopy(rows[0])
    bad_shape["condition_embeddings"][1]["embedding"] = [1.0]
    assert mod.embedding_shape_and_finiteness([bad_shape])["shape_failure_count"] == 1
    loaded_backend = FakeEmbeddingBackend(
        artifact["model_specs"][0],
        artifact["deterministic_embedding_config"],
    )
    loaded_backend.load()
    with pytest.raises(ValueError, match="source_row_must_have_two_conditions"):
        mod._build_output_row(
            source_index=0,
            source_row={**source_rows[0], "conditions": [source_rows[0]["conditions"][0]]},
            model_spec=artifact["model_specs"][0],
            backend=loaded_backend,
            config=artifact["deterministic_embedding_config"],
            loader_receipt={"loader": "fixture"},
        )
    ragged = RaggedEmbeddingBackend(
        artifact["model_specs"][0],
        artifact["deterministic_embedding_config"],
    )
    ragged.load()
    with pytest.raises(ValueError, match="embedding_pair_shape_mismatch"):
        mod._build_output_row(
            source_index=0,
            source_row=source_rows[0],
            model_spec=artifact["model_specs"][0],
            backend=ragged,
            config=artifact["deterministic_embedding_config"],
            loader_receipt={"loader": "fixture"},
        )
    nonfinite = deepcopy(rows[0])
    nonfinite["condition_embeddings"][0]["embedding"][0] = float("inf")
    assert mod.embedding_shape_and_finiteness([nonfinite])["nonfinite_embedding_count"] == 1
    missing_source = deepcopy(rows[0])
    missing_source["source_row_id"] = "missing"
    bad_suffix = deepcopy(rows[1])
    bad_suffix["condition_embeddings"][1]["condition_suffix"] = "z"
    bad_diff = deepcopy(rows[2])
    bad_diff["paired_difference"] = [0.0]
    alignment = mod.pair_alignment_receipts(
        [missing_source, bad_suffix, bad_diff],
        source_rows=source_rows,
        model_hf_ids=mod.MANDATED_MODEL_HF_IDS,
    )
    assert alignment["pair_join_failure_count"] == 2
    assert alignment["paired_difference_mismatch_count"] == 1

    truncated = deepcopy(rows[0])
    truncated["condition_embeddings"][0]["truncated"] = True
    assert mod.token_and_truncation_parity([truncated])["truncation_asymmetry_count"] == 1
    token_leak = deepcopy(rows[0])
    token_leak["feature_consumer_view"]["note"] = "gemma"
    assert mod.feature_leakage_checks([token_leak])["feature_token_leakage_count"] == 1

    bad_receipt = deepcopy(artifact)
    bad_receipt["row_file_receipt"]["path"] = "wrong"
    with pytest.raises(ValueError, match="row_file_receipt"):
        mod.verify_row_file(rows, bad_receipt)
    bad_count = deepcopy(artifact)
    bad_count["row_file_receipt"]["row_count"] += 1
    with pytest.raises(ValueError, match="row_count"):
        mod.verify_row_file(rows, bad_count)
    bad_row_hash = deepcopy(rows)
    bad_row_hash[0]["row_hash"] = "sha256:bad"
    with pytest.raises(ValueError, match="row_hash"):
        mod.verify_row_file(bad_row_hash, artifact)
    bad_file_hash = deepcopy(artifact)
    first_cell = rows[0]["embedding_cell_id"]
    bad_file_hash["row_file_receipt"]["row_hashes"][first_cell] = "sha256:bad"
    with pytest.raises(ValueError, match="row_file_hash"):
        mod.verify_row_file(rows, bad_file_hash)
    bad_sha = deepcopy(artifact)
    bad_sha["row_file_receipt"]["sha256"] = "sha256:bad"
    with pytest.raises(ValueError, match="row_file_sha256"):
        mod.verify_row_file(rows, bad_sha)

    for field, value, error in (
        ("inference_substrate", "wrong", "inference_substrate"),
        ("verifier_is_oracle", False, "verifier_is_oracle"),
        ("models_used", [], "models_used"),
    ):
        tampered = deepcopy(artifact)
        tampered[field] = value
        tampered["paired_embedding_corpus_ready_score"] = mod.paired_embedding_corpus_ready_score(
            tampered
        )
        tampered["status"] = "partial"
        tampered["honest_verdict"] = "partial: field-specific"
        tampered["reproducibility_checksum"] = mod.reproducibility_checksum(tampered)
        with pytest.raises(ValueError, match=error):
            mod.validate_artifact(tampered)
    tampered_checksum = deepcopy(artifact)
    tampered_checksum["reproducibility_checksum"] = "sha256:bad"
    with pytest.raises(ValueError, match="reproducibility_checksum"):
        mod.validate_artifact(tampered_checksum)
    missing_field = deepcopy(artifact)
    del missing_field["status"]
    with pytest.raises(ValueError, match="status"):
        mod.validate_artifact(missing_field)
    bad_provenance = deepcopy(artifact)
    bad_provenance["field_provenance"] = {}
    with pytest.raises(ValueError, match="field_provenance"):
        mod.validate_artifact(bad_provenance)
    bad_ready_status = deepcopy(artifact)
    bad_ready_status["status"] = "blocked"
    bad_ready_status["reproducibility_checksum"] = mod.reproducibility_checksum(bad_ready_status)
    with pytest.raises(ValueError, match="status"):
        mod.validate_artifact(bad_ready_status)
    bad_ready_verdict = deepcopy(artifact)
    bad_ready_verdict["honest_verdict"] = "blocked: wrong"
    bad_ready_verdict["reproducibility_checksum"] = mod.reproducibility_checksum(bad_ready_verdict)
    with pytest.raises(ValueError, match="honest_verdict"):
        mod.validate_artifact(bad_ready_verdict)

    blocked = mod.run(
        result_path=tmp_path / "missing-fixture.json",
        row_file_path=tmp_path / "missing-fixture.rows.jsonl",
        checkpoint_dir=tmp_path / "missing-checkpoints",
        fixture_artifact_path=tmp_path / "missing-exp5840.json",
        fixture_rows_path=tmp_path / "missing-exp5840.rows.jsonl",
        model_specs=_model_specs(tmp_path),
        preconditions_checked=_ready_preconditions(tmp_path),
        embedding_backend_factory=FakeEmbeddingBackend,
        test_commands=TEST_COMMANDS,
        test_exit_codes=TEST_EXIT_CODES,
        write=False,
    )
    assert blocked["status"] == "blocked"
    assert (
        "exp5840_fixture_unavailable_or_not_ready"
        in blocked["preconditions_checked"]["blocked_reasons"]
    )
    bad_blocked_status = deepcopy(blocked)
    bad_blocked_status["status"] = "weird"
    bad_blocked_status["reproducibility_checksum"] = mod.reproducibility_checksum(
        bad_blocked_status
    )
    with pytest.raises(ValueError, match="status"):
        mod.validate_artifact(bad_blocked_status)
    bad_blocked_verdict = deepcopy(blocked)
    bad_blocked_verdict["honest_verdict"] = "ready: wrong"
    bad_blocked_verdict["reproducibility_checksum"] = mod.reproducibility_checksum(
        bad_blocked_verdict
    )
    with pytest.raises(ValueError, match="honest_verdict"):
        mod.validate_artifact(bad_blocked_verdict)
