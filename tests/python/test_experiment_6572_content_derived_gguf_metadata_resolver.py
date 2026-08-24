"""Tests for the Exp6572 terminal metadata resolver artifact."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from carnot import experiment_6572_content_derived_gguf_metadata_resolver as exp


def _blob_rows() -> list[dict]:
    rows = []
    for index, hf_id in enumerate(exp.MANDATED_HF_IDS):
        rows.append(
            {
                "row_type": "flagship_blob",
                "unit_id": hf_id,
                "repository_id": hf_id,
                "path": f"/cache/blobs/{index:064x}",
                "admitted": True,
                "content_metadata": {
                    "architecture": "qwen35moe" if index == 0 else "gemma4",
                    "quantization": "Q4_K_M",
                    "tensor_count": 10,
                    "is_language_model": True,
                    "bounded_read_receipt": {
                        "physical_bytes_read": 1024,
                        "maximum_header_bytes": exp.MAX_HEADER_BYTES,
                        "tensor_payload_bytes_read": 0,
                    },
                },
                "provenance": {
                    "valid": True,
                    "repository_id": hf_id,
                    "revision": f"revision-{index}",
                    "snapshot_filename": f"model-{index}.gguf",
                    "trusted_hash_matches_blob_key": True,
                    "full_blob_rehash_performed": False,
                },
                "rejection_reasons": [],
                "passed": True,
            }
        )
    return rows


def _negative_rows() -> list[dict]:
    return [
        {
            "row_type": "negative_fixture",
            "unit_id": fixture_id,
            "expected_admitted": False,
            "observed_admitted": False,
            "expected_reason": "closed",
            "rejection_reasons": ["closed"],
            "passed": True,
            "bounded_read_receipt": {
                "maximum_header_bytes": exp.MAX_HEADER_BYTES,
                "physical_bytes_read_upper_bound": 128,
                "tensor_payload_bytes_read": 0,
            },
        }
        for fixture_id in exp.REQUIRED_NEGATIVE_FIXTURES
    ]


def _failing_before() -> list[dict]:
    return [
        {
            "row_type": "exp6567_false_negative",
            "hf_id": hf_id,
            "path": f"/cache/blobs/{index:064x}",
            "expected_language_model_file": True,
            "observed_language_model_file": False,
            "expected_quantization_known": True,
            "observed_quantization_known": False,
            "false_negative_reproduced": True,
        }
        for index, hf_id in enumerate(exp.MANDATED_HF_IDS)
    ]


def _preconditions(*, ready: bool = True) -> dict:
    return {
        "all_required_preconditions_available": ready,
        "upstream_gate": {
            "expected_value": 1.0,
            "observed_value": 1.0 if ready else 0.0,
            "passed": ready,
        },
        "checks": {"bounded_reader": True, "cache": ready},
    }


def _protected(*, unchanged: bool = True) -> dict:
    return {
        "all_unchanged": unchanged,
        "research_roadmap_yaml_unchanged": unchanged,
        "research_conductor_py_unchanged": unchanged,
        "rows": [],
    }


def _assemble(
    *,
    failing_before_rows: list[dict] | None = None,
    blob_rows: list[dict] | None = None,
    negative_rows: list[dict] | None = None,
    preconditions: dict | None = None,
    protected: dict | None = None,
) -> dict:
    return exp.assemble_artifact(
        failing_before_rows=(
            _failing_before() if failing_before_rows is None else failing_before_rows
        ),
        blob_rows=_blob_rows() if blob_rows is None else blob_rows,
        negative_rows=_negative_rows() if negative_rows is None else negative_rows,
        preconditions=_preconditions() if preconditions is None else preconditions,
        protected=_protected() if protected is None else protected,
        duration_s=2.5,
        tests_run=[{"command": "pytest focused", "exit_code": 0}],
        run_date="20260824",
    )


# REQ-REPORT-6572-CONTENT / SCENARIO-REPORT-6572-HASH-BLOB:
# the exact Exp6567 false negatives remain visible as repair targets.
def test_failing_before_rows_reproduce_all_three_hash_blob_failures() -> None:
    payload = {
        "resolved_model_file_rows": [
            {"hf_id": hf_id, "absolute_path": f"/cache/blobs/{index:064x}"}
            for index, hf_id in enumerate(exp.MANDATED_HF_IDS)
        ],
        "preconditions_checked": {
            "model_preflight_checks": {
                hf_id: {"language_model_file": False, "quantization_known": False}
                for hf_id in exp.MANDATED_HF_IDS
            }
        },
    }

    rows = exp.build_failing_before_rows(payload)

    assert len(rows) == 3
    assert all(row["false_negative_reproduced"] for row in rows)
    assert all(Path(row["path"]).suffix == "" for row in rows)


# REQ-REPORT-6572-ATTACKS / SCENARIO-REPORT-6572-REJECT:
# every required malformed fixture executes and rejects.
def test_negative_fixture_runner_emits_every_closed_row() -> None:
    rows = exp.build_negative_fixture_rows()

    assert {row["unit_id"] for row in rows} == set(exp.REQUIRED_NEGATIVE_FIXTURES)
    assert all(row["observed_admitted"] is False for row in rows)
    assert all(row["passed"] is True for row in rows)
    assert all(row["bounded_read_receipt"]["tensor_payload_bytes_read"] == 0 for row in rows)


# REQ-REPORT-6572-ATOMIC / SCENARIO-REPORT-6572-ATOMIC:
# clean per-unit rows are the only source of readiness.
def test_ready_artifact_recomputes_from_blob_and_fixture_rows() -> None:
    artifact = _assemble()

    assert artifact["status"] == "ready"
    assert artifact["verdict_class"] is None
    assert artifact["gguf_blob_metadata_ready_score"] == 1.0
    assert artifact["aggregate_row_recomputation"]["recomputed_ready_score"] == 1.0
    assert (
        artifact["per_unit_rows"]
        == artifact["gguf_blob_metadata_rows"] + artifact["negative_fixture_rows"]
    )
    assert exp.validate_artifact(artifact) == []


# REQ-REPORT-6572-ATOMIC: a usable subset is partial, not ready.
def test_partial_artifact_for_usable_blob_subset() -> None:
    blobs = _blob_rows()
    blobs[0]["admitted"] = False
    blobs[0]["passed"] = False
    blobs[0]["rejection_reasons"] = ["content_failure"]

    artifact = _assemble(blob_rows=blobs)

    assert artifact["status"] == "partial"
    assert artifact["verdict_class"] == "partial"
    assert artifact["gguf_blob_metadata_ready_score"] == 0.0


# REQ-REPORT-6572-ATTACKS: a negative fixture that admits disqualifies evidence.
def test_failed_negative_fixture_disqualifies_artifact() -> None:
    negatives = _negative_rows()
    negatives[0]["observed_admitted"] = True
    negatives[0]["passed"] = False

    artifact = _assemble(negative_rows=negatives)

    assert artifact["status"] == "disqualified"
    assert artifact["verdict_class"] == "disqualified"
    assert artifact["gguf_blob_metadata_ready_score"] == 0.0


# REQ-REPORT-6572-ATOMIC: a missing upstream gate blocks before readiness.
def test_missing_precondition_blocks_artifact() -> None:
    artifact = _assemble(preconditions=_preconditions(ready=False))

    assert artifact["status"] == "blocked"
    assert artifact["verdict_class"] == "blocked"
    assert artifact["gate_check_summary"]["failed_checks"] == ["preconditions"]


# REQ-REPORT-6572-ATOMIC: no usable blob is blocked when tools are present.
def test_no_usable_blob_is_blocked() -> None:
    artifact = _assemble(blob_rows=[])

    assert artifact["status"] == "blocked"
    assert artifact["verdict_class"] == "blocked"


# REQ-REPORT-6572-CONTENT: missing pre-fix rows disqualify causal evidence.
def test_missing_failing_before_row_disqualifies_artifact() -> None:
    artifact = _assemble(failing_before_rows=_failing_before()[:2])

    assert artifact["status"] == "disqualified"
    assert "failing_before_rows" in artifact["gate_check_summary"]["failed_checks"]


# REQ-REPORT-6572-ATOMIC: protected orchestration mutation disqualifies output.
def test_protected_file_mutation_disqualifies_artifact() -> None:
    artifact = _assemble(protected=_protected(unchanged=False))

    assert artifact["status"] == "disqualified"
    assert artifact["verdict_class"] == "disqualified"


# REQ-REPORT-6572-ATOMIC: checksum and reducer tampering are detected.
def test_validation_detects_checksum_rows_and_required_field_mutation() -> None:
    artifact = _assemble()
    artifact["gguf_blob_metadata_ready_score"] = 0.0
    artifact["protected_files_unchanged"]["all_unchanged"] = False
    artifact["field_provenance"].pop("status")

    errors = exp.validate_artifact(artifact)

    assert "ready score does not recompute" in errors
    assert "protected files changed" in errors
    assert "field_provenance must cover required fields" in errors
    assert "reproducibility checksum mismatch" in errors


# REQ-REPORT-6572-ATOMIC: atomic replacement leaves no temporary file.
def test_atomic_write_replaces_target_and_validate_cli_reads_it(
    tmp_path: Path,
    capsys,
) -> None:
    path = tmp_path / "artifact.json"
    path.write_text('{"old": true}')
    artifact = _assemble()

    exp.atomic_write_json(path, artifact)

    assert json.loads(path.read_text()) == artifact
    assert list(tmp_path.glob("*.tmp")) == []
    assert exp.main(["--validate", "--result-path", str(path)]) == 0
    assert "validated" in capsys.readouterr().out


# REQ-REPORT-6572-ATOMIC: invalid or missing CLI targets fail loudly.
def test_validate_cli_rejects_missing_and_invalid_artifacts(tmp_path: Path, capsys) -> None:
    missing = tmp_path / "missing.json"
    assert exp.main(["--validate", "--result-path", str(missing)]) == 1
    assert "artifact not found" in capsys.readouterr().out
    bad = tmp_path / "bad.json"
    bad.write_text("{}")
    assert exp.main(["--validate", "--result-path", str(bad)]) == 1
    assert "missing required fields" in capsys.readouterr().out


# REQ-REPORT-6572-ATOMIC: canonical hashes do not depend on key order.
def test_canonical_hash_helpers_are_stable() -> None:
    assert exp.sha256_json({"a": 1, "b": 2}) == exp.sha256_json({"b": 2, "a": 1})
    row = {"value": 1}
    checksum = exp.artifact_checksum(row)
    row["reproducibility_checksum"] = checksum
    assert exp.artifact_checksum(row) == checksum


# REQ-REPORT-6572-ATOMIC: file and JSON helpers fail closed on absent input.
def test_file_and_json_helpers_handle_missing_invalid_and_non_object(tmp_path: Path) -> None:
    assert exp.sha256_file(tmp_path / "missing") == "missing"
    assert exp._load_json(tmp_path / "missing.json") == {}  # noqa: SLF001
    invalid = tmp_path / "invalid.json"
    invalid.write_text("{")
    assert exp._load_json(invalid) == {}  # noqa: SLF001
    array = tmp_path / "array.json"
    array.write_text("[]")
    assert exp._load_json(array) == {}  # noqa: SLF001
    with pytest.raises(AssertionError, match="unsupported fixture type"):
        exp._fixture_value(7, 1)  # noqa: SLF001


# REQ-REPORT-6572-CONTENT: flagship rows preserve admission and source size.
def test_flagship_row_builder_wraps_reusable_admission(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = {
        "resolved_model_file_rows": [
            {
                "hf_id": hf_id,
                "absolute_path": f"/cache/{index}",
                "sha256": f"sha256:{index:064x}",
                "byte_size": index + 1,
            }
            for index, hf_id in enumerate(exp.MANDATED_HF_IDS)
        ]
    }
    calls = []

    def fake_admission(path: str, **kwargs) -> dict:
        calls.append((path, kwargs))
        return {"admitted": True, "repository_id": kwargs["repository_id"]}

    monkeypatch.setattr(exp, "build_gguf_admission_record", fake_admission)

    rows = exp.build_flagship_blob_rows(source)

    assert len(rows) == 3
    assert all(row["passed"] for row in rows)
    assert [row["exp6567_byte_size"] for row in rows] == [1, 2, 3]
    assert len(calls) == 3


# REQ-REPORT-6572-ATOMIC: protected hashes compare both required files.
def test_protected_hash_and_memory_receipts(tmp_path: Path) -> None:
    (tmp_path / "scripts").mkdir()
    (tmp_path / "research-roadmap.yaml").write_text("roadmap")
    (tmp_path / "scripts" / "research_conductor.py").write_text("conductor")
    before = exp._protected_hashes(tmp_path)  # noqa: SLF001
    unchanged = exp._protected_unchanged(before, before)  # noqa: SLF001
    changed = dict(before)
    changed["research-roadmap.yaml"] = "sha256:changed"

    assert unchanged["all_unchanged"] is True
    assert exp._protected_unchanged(before, changed)["all_unchanged"] is False  # noqa: SLF001
    assert exp._memory_receipt()["total_kib"] > 0  # noqa: SLF001


# REQ-REPORT-6572-BOUNDED: tool support records missing and unversioned installs.
def test_llama_cpp_support_handles_missing_and_unversioned(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(exp.importlib.util, "find_spec", lambda _name: None)
    assert exp._llama_cpp_support()["installed"] is False  # noqa: SLF001

    class Spec:
        origin = "/fixture/llama_cpp.py"

    monkeypatch.setattr(exp.importlib.util, "find_spec", lambda _name: Spec())

    def missing_version(_name: str) -> str:
        raise exp.importlib.metadata.PackageNotFoundError

    monkeypatch.setattr(exp.importlib.metadata, "version", missing_version)
    receipt = exp._llama_cpp_support()  # noqa: SLF001
    assert receipt["installed"] is True
    assert receipt["version"] == "unknown"


# REQ-REPORT-6572-ATOMIC: preconditions report gate, disk, RAM, tools, and cache.
def test_collect_preconditions_records_expected_and_observed_values(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "results").mkdir()
    (tmp_path / exp.UPSTREAM_RELATIVE_PATH).write_text(
        json.dumps({"v570_evidence_contract_ready_score": 1.0})
    )
    cache_files = []
    for index in range(3):
        path = tmp_path / f"blob-{index}"
        path.write_bytes(b"x")
        cache_files.append(path)
    exp6567 = {
        "resolved_model_file_rows": [
            {
                "hf_id": hf_id,
                "absolute_path": str(cache_files[index]),
                "byte_size": 1,
                "sha256": f"sha256:{index:064x}",
            }
            for index, hf_id in enumerate(exp.MANDATED_HF_IDS)
        ]
    }
    monkeypatch.setattr(
        exp,
        "_llama_cpp_support",
        lambda: {"installed": True, "gpu_offload_supported": True},
    )

    receipt = exp.collect_preconditions(tmp_path, exp6567, {}, "20260824")

    assert receipt["upstream_gate"]["expected_value"] == 1.0
    assert receipt["upstream_gate"]["observed_value"] == 1.0
    assert receipt["all_required_preconditions_available"] is True
    assert receipt["disk"]["free_bytes"] > 0
    assert receipt["llm_load_performed"] is False


# REQ-REPORT-6572-ATOMIC: validator covers wrong shapes, classes, and read bounds.
def test_validation_rejects_wrong_shapes_class_and_unbounded_receipts() -> None:
    wrong_shape = _assemble()
    wrong_shape["negative_fixture_rows"] = {}
    wrong_shape["reproducibility_checksum"] = exp.artifact_checksum(wrong_shape)
    assert "unit row fields must be lists" in exp.validate_artifact(wrong_shape)

    wrong_class = _assemble()
    wrong_class["verdict_class"] = "positive"
    wrong_class["reproducibility_checksum"] = exp.artifact_checksum(wrong_class)
    assert "ready verdict_class must be null" in exp.validate_artifact(wrong_class)

    unbounded = _assemble()
    unbounded["bounded_read_receipts"][0]["physical_bytes_read"] = MAX = exp.MAX_HEADER_BYTES + 1
    unbounded["reproducibility_checksum"] = exp.artifact_checksum(unbounded)
    assert MAX > exp.MAX_HEADER_BYTES
    assert "bounded read receipt exceeds limit" in exp.validate_artifact(unbounded)

    tensor_read = _assemble()
    tensor_read["bounded_read_receipts"][0]["tensor_payload_bytes_read"] = 1
    tensor_read["reproducibility_checksum"] = exp.artifact_checksum(tensor_read)
    assert "tensor payload bytes were read" in exp.validate_artifact(tensor_read)


def _patch_build_inputs(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(exp, "_protected_hashes", lambda _root: {"protected": "hash"})
    monkeypatch.setattr(exp, "_load_json", lambda _path: {"source": True})
    monkeypatch.setattr(exp, "collect_preconditions", lambda *_args: _preconditions())
    monkeypatch.setattr(exp, "build_failing_before_rows", lambda _source: _failing_before())
    monkeypatch.setattr(exp, "build_flagship_blob_rows", lambda _source: _blob_rows())
    monkeypatch.setattr(exp, "build_negative_fixture_rows", _negative_rows)
    monkeypatch.setattr(exp, "_protected_unchanged", lambda *_args: _protected())


# REQ-REPORT-6572-ATOMIC: the top-level builder returns only validated rows.
def test_build_artifact_success_and_validation_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_build_inputs(monkeypatch)
    artifact = exp.build_artifact(repo_root=tmp_path, tests_run=[])
    assert artifact["status"] == "ready"

    monkeypatch.setattr(exp, "validate_artifact", lambda _artifact: ["forced failure"])
    with pytest.raises(ValueError, match="forced failure"):
        exp.build_artifact(repo_root=tmp_path, tests_run=[])
