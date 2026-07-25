"""Tests for Exp5907 ConstraintIR replay contract repair.

Spec refs: REQ-BENCH-5907, SCENARIO-BENCH-5907-CANONICAL,
SCENARIO-BENCH-5907-FRESH-PROCESS, SCENARIO-BENCH-5907-TAMPER,
SCENARIO-BENCH-5907-LEGACY.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

import pytest

from carnot import constraint_ir_replay_contract as contract
from carnot import experiment_5896_typed_constraint_ir_fixture as exp5896
from carnot import experiment_5897_sota_constraint_ir_repair_ab as exp5897
from carnot import experiment_5907_constraint_ir_replay_contract as exp5907


def _write_twin(root: Path) -> dict[str, object]:
    return exp5896.write_fixture(root=root, duration_s=0.0)


def _load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


# REQ-BENCH-5907, SCENARIO-BENCH-5907-CANONICAL
def test_shared_projection_checksum_matches_producer_and_consumer(tmp_path: Path) -> None:
    artifact = _write_twin(tmp_path)
    row_path = tmp_path / exp5896.ROW_FILE_RELATIVE_PATH
    row_sha256 = exp5896.sha256_file(row_path)

    receipt = contract.projection_receipt(artifact, row_file_sha256=row_sha256)
    producer_replay = exp5896.replay_artifact(root=tmp_path)
    consumer_replay = exp5897._upstream_gate_receipt(tmp_path)

    assert receipt["projection_schema_version"] == contract.PROJECTION_SCHEMA_VERSION
    assert artifact["canonical_projection_schema_and_version"]["projection_schema_version"] == (
        contract.PROJECTION_SCHEMA_VERSION
    )
    assert receipt["checksum"] == artifact["reproducibility_checksum"]
    assert producer_replay["canonical_projection"]["checksum"] == receipt["checksum"]
    assert consumer_replay["replay_ok"] is True
    assert consumer_replay["canonical_projection"]["checksum"] == receipt["checksum"]
    assert "protected_files_unchanged" in contract.EXCLUDED_TOP_LEVEL_FIELDS
    assert receipt["bound_fields"]["row_file_sha256"] == row_sha256

    with pytest.raises(contract.ConstraintIRReplayContractError, match="unknown projection"):
        contract.canonical_projection_bytes(
            artifact,
            row_file_sha256=row_sha256,
            projection_version="carnot.constraint_ir.replay_contract_projection.v0",
        )


# REQ-BENCH-5907, SCENARIO-BENCH-5907-FRESH-PROCESS
def test_fresh_process_replay_is_independent(tmp_path: Path) -> None:
    artifact = _write_twin(tmp_path)
    expected_checksum = str(artifact["reproducibility_checksum"])
    code = """
import json
import sys
from pathlib import Path
from carnot import experiment_5896_typed_constraint_ir_fixture as exp5896
receipt = exp5896.replay_artifact(root=Path(sys.argv[1]))
print(json.dumps(receipt, sort_keys=True))
"""
    env = dict(os.environ)
    env["PYTHONPATH"] = str(exp5896.REPO_ROOT / "python")
    proc = subprocess.run(
        [sys.executable, "-c", code, str(tmp_path)],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    assert proc.returncode == 0, proc.stderr
    receipt = json.loads(proc.stdout)
    assert receipt["ok"] is True
    assert receipt["canonical_projection"]["checksum"] == expected_checksum
    assert receipt["canonical_projection"]["row_file_sha256"] == exp5896.sha256_file(
        tmp_path / exp5896.ROW_FILE_RELATIVE_PATH
    )


# REQ-BENCH-5907, SCENARIO-BENCH-5907-TAMPER
def test_tamper_detection_rejects_every_bound_component() -> None:
    matrix = exp5907.run_tamper_detection_matrix()

    components = {case["component"] for case in matrix["cases"]}
    assert components == {
        "row_file_bytes",
        "row_file_sha256_receipt",
        "constraint_ir_schema_version",
        "projection_schema_version",
        "reproducibility_checksum",
    }
    assert matrix["all_rejected"] is True
    assert all(case["rejected"] is True for case in matrix["cases"])


# REQ-BENCH-5907, SCENARIO-BENCH-5907-LEGACY
def test_legacy_mismatch_is_preserved_but_new_contract_replays() -> None:
    mismatch = exp5907.reproduce_historical_mismatch(root=exp5896.REPO_ROOT)
    legacy = exp5907.adjudicate_legacy_exp5896(root=exp5896.REPO_ROOT)
    consumer = exp5897._upstream_gate_receipt(exp5896.REPO_ROOT)

    assert mismatch["old_replay_error"].startswith("ConstraintIRReplayError:")
    assert mismatch["artifact_projection_checksum"] != mismatch["rebuilt_projection_checksum"]
    assert mismatch["root_cause"]["differing_paths"] == [
        "$.protected_files_unchanged.files[1].sha256",
        "$.protected_files_unchanged.files[2].sha256",
    ]
    assert legacy["historical_checksum_mismatch_preserved"] is True
    assert legacy["new_contract_replay_ready"] is True
    assert legacy["legacy_mode_without_projection_field"] is True
    assert consumer["replay_ok"] is True


# REQ-BENCH-5907, SCENARIO-BENCH-5907-CANONICAL,
# SCENARIO-BENCH-5907-LEGACY
def test_exp5907_artifact_schema_and_checksum(tmp_path: Path) -> None:
    output_path = tmp_path / exp5907.RESULT_RELATIVE_PATH
    artifact = exp5907.write_contract_artifact(output_path=output_path, duration_s=0.0)
    loaded = _load_json(output_path)

    assert loaded == artifact
    assert set(exp5907.REQUIRED_ARTIFACT_FIELDS).issubset(artifact)
    assert artifact["status"] == "complete_ready"
    assert artifact["constraint_ir_replay_contract_ready_score"] == 1.0
    assert artifact["inference_substrate"] == "deterministic_artifact_replay_no_llm"
    assert artifact["verifier_is_oracle"] is True
    assert artifact["honest_verdict"].startswith("complete_ready:")
    assert artifact["historical_artifacts_unchanged"]["unchanged"] is True
    assert artifact["protected_files_unchanged"]["unchanged"] is True
    assert artifact["fresh_process_replay_receipt"]["ok"] is True
    assert artifact["tamper_detection_matrix"]["all_rejected"] is True
    assert artifact["canonical_projection_schema_and_version"]["projection_schema_version"] == (
        contract.PROJECTION_SCHEMA_VERSION
    )
    exp5907.validate_artifact(artifact)

    measured = exp5907.write_contract_artifact(output_path=tmp_path / "measured.json")
    assert measured["duration_s"] > 0.0
    exp5907.validate_artifact(measured)


# REQ-BENCH-5907, SCENARIO-BENCH-5907-TAMPER
def test_projection_and_artifact_validation_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact = _write_twin(tmp_path)
    row_sha256 = exp5896.sha256_file(tmp_path / exp5896.ROW_FILE_RELATIVE_PATH)

    assert contract.sha256_text("abc").startswith("sha256:")
    assert contract.canonical_projection_bytes(artifact, row_file_sha256=row_sha256)
    assert contract.canonical_json({"tuple": (1, 2)}) == '{"tuple":[1,2]}'
    with pytest.raises(contract.ConstraintIRReplayContractError, match="finite"):
        contract.canonical_json({"bad": float("nan")})
    with pytest.raises(contract.ConstraintIRReplayContractError, match="unsupported JSON"):
        contract.canonical_json({"bad": object()})

    bad_projection_type = dict(artifact)
    bad_projection_type["canonical_projection_schema_and_version"] = []
    with pytest.raises(
        contract.ConstraintIRReplayContractError, match="metadata must be an object"
    ):
        contract.projection_receipt(bad_projection_type, row_file_sha256=row_sha256)

    missing_projection_version = dict(artifact)
    missing_projection_version["canonical_projection_schema_and_version"] = {}
    with pytest.raises(contract.ConstraintIRReplayContractError, match="lacks projection version"):
        contract.projection_receipt(missing_projection_version, row_file_sha256=row_sha256)

    with pytest.raises(contract.ConstraintIRReplayContractError, match="must be prefixed"):
        contract.projection_receipt(artifact, row_file_sha256="not-prefixed")

    missing_receipt = dict(artifact)
    del missing_receipt["row_file_receipt"]
    with pytest.raises(contract.ConstraintIRReplayContractError, match="row_file_receipt"):
        contract.projection_receipt(missing_receipt, row_file_sha256=row_sha256)

    mismatched_receipt = json.loads(json.dumps(artifact))
    mismatched_receipt["row_file_receipt"]["sha256"] = "sha256:" + "2" * 64
    with pytest.raises(contract.ConstraintIRReplayContractError, match="row file hash"):
        contract.projection_receipt(mismatched_receipt, row_file_sha256=row_sha256)

    malformed_preconditions = json.loads(json.dumps(artifact))
    malformed_preconditions["preconditions_checked"] = "not-an-object"
    assert contract.projection_receipt(malformed_preconditions, row_file_sha256=row_sha256)[
        "checksum"
    ].startswith("sha256:")

    terminal = exp5907.write_contract_artifact(
        output_path=tmp_path / "artifact.json", duration_s=0.0
    )
    for key, value, message in [
        ("honest_verdict", "blocked: wrong prefix", "ready score"),
        ("constraint_ir_replay_contract_ready_score", 0.5, "bare"),
        ("verifier_is_oracle", False, "verifier_is_oracle"),
        ("inference_substrate", "live_llm_inference", "inference_substrate"),
    ]:
        broken = json.loads(json.dumps(terminal))
        broken[key] = value
        with pytest.raises(ValueError, match=message):
            exp5907.validate_artifact(broken)

    broken_projection = json.loads(json.dumps(terminal))
    broken_projection["canonical_projection_schema_and_version"]["projection_schema_version"] = (
        "bad"
    )
    with pytest.raises(ValueError, match="canonical projection"):
        exp5907.validate_artifact(broken_projection)

    missing = dict(terminal)
    del missing["honest_verdict"]
    with pytest.raises(ValueError, match="missing required"):
        exp5907.validate_artifact(missing)

    assert exp5907._diff_paths(1, "1") == [("$", 1, "1")]
    assert exp5907._diff_paths({"a": 1}, {"b": 1}) == [
        ("$.a", 1, "<missing>"),
        ("$.b", "<missing>", 1),
    ]
    assert exp5907._diff_paths([1], [1, 2]) == [("$.length", 1, 2)]

    def replay_accepts_tamper(*args: Any, **kwargs: Any) -> dict[str, object]:
        del args, kwargs
        return {"ok": True}

    monkeypatch.setattr(exp5907.exp5896, "replay_artifact", replay_accepts_tamper)
    matrix = exp5907.run_tamper_detection_matrix()
    assert matrix["all_rejected"] is False
    assert all(case["rejected"] is False for case in matrix["cases"])
