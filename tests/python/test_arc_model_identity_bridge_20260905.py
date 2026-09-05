"""Tests for REQ-ARC-7030 and the shared ARC model identity bridge."""

from __future__ import annotations

from dataclasses import replace
import hashlib
import inspect
import json
from pathlib import Path

import pytest

from carnot import experiment_7030_arc_gguf_model_identity_bridge as experiment
from carnot.agentic import arc_belief_shadow_live_trace as shadow
from carnot.agentic import arc_eval_provenance as provenance
from carnot.agentic.arc_eval_provenance import (
    ARC_EVAL_PROVENANCE_SCHEMA_VERSION,
    ARC_EVAL_PROVENANCE_SCHEMA_VERSION_V2,
    ArcEvalProvenanceInput,
    build_arc_eval_provenance,
    build_arc_model_identity_receipt,
    compute_arc_eval_provenance_hash,
    validate_arc_eval_provenance,
)


ROOT = Path(__file__).resolve().parents[2]
HF_ID = "unsloth/Qwen3.6-35B-A3B-GGUF"
REVISION = "a483e9e6cbd595906af30beda3187c2663a1118c"
FILENAME = "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"


def _snapshot_fixture(tmp_path: Path) -> tuple[dict[str, object], Path]:
    """SCENARIO-ARC-7030-EXP7025-SNAPSHOT-BLOB-JOIN uses the real cache shape."""

    model_root = tmp_path / "hub" / "models--unsloth--Qwen3.6-35B-A3B-GGUF"
    payload = b"GGUF-shaped Exp7025 identity fixture\n"
    digest = hashlib.sha256(payload).hexdigest()
    blob = model_root / "blobs" / digest
    blob.parent.mkdir(parents=True)
    blob.write_bytes(payload)
    requested = model_root / "snapshots" / REVISION / FILENAME
    requested.parent.mkdir(parents=True)
    requested.symlink_to(Path("../../blobs") / digest)
    spec: dict[str, object] = {
        "name": "Qwen3.6-35B-A3B",
        "hf_id": HF_ID,
        "gpu": 0,
        "model_path": str(requested),
        "model_filename": FILENAME,
        "revision": REVISION,
        "model_file_hash": "sha256:" + digest,
        "resolved_via": "cached_sota_pair",
    }
    return spec, blob


def _legacy_input(**changes: object) -> ArcEvalProvenanceInput:
    source = ArcEvalProvenanceInput(
        inference_substrate="local_gguf_cuda",
        gpu_uuid="GPU-70300000-0000-0000-0000-000000000001",
        gpu_model="NVIDIA RTX 3090",
        cuda_device=0,
        model_repository=HF_ID,
        model_filename=FILENAME,
        model_hash="sha256:" + "1" * 64,
        n_ctx=4096,
        server_binary="/opt/llama.cpp/llama-server",
        server_binary_hash="sha256:" + "2" * 64,
        server_command_hash="sha256:" + "3" * 64,
        endpoint="http://127.0.0.1:17030",
        port=17030,
        lease_id="lease-7030",
        lease_hash="sha256:" + "4" * 64,
        lease_issued_at="2026-09-05T00:00:00+00:00",
        lease_expires_at="2026-09-05T02:00:00+00:00",
        lease_checked_at="2026-09-05T01:00:00+00:00",
        request_count=1,
        completion_count=1,
        error_count=0,
        policy_hash="sha256:" + "5" * 64,
        factory_hash="sha256:" + "6" * 64,
        git_commit="7" * 40,
        solve_provenance="live_agent_self_discovery",
    )
    return replace(source, **changes)


def _rehash(record: dict[str, object]) -> None:
    record["provenance_hash"] = compute_arc_eval_provenance_hash(record)


def test_req_arc_7030_spec_exists_before_implementation() -> None:
    """REQ-ARC-7030 names the positive, negative, legacy, and wiring checks."""

    text = (ROOT / "openspec/capabilities/arc-agi/spec.md").read_text(encoding="utf-8")
    assert "REQ-ARC-7030" in text
    for scenario in (
        "SCENARIO-ARC-7030-EXP7025-SNAPSHOT-BLOB-JOIN",
        "SCENARIO-ARC-7030-IDENTITY-NEGATIVE-MATRIX",
        "SCENARIO-ARC-7030-LEGACY-VERSION-READ",
        "SCENARIO-ARC-7030-SHARED-PRODUCER-WIRING",
    ):
        assert scenario in text


def test_exp7025_snapshot_symlink_joins_to_extensionless_blob(tmp_path: Path) -> None:
    """SCENARIO-ARC-7030-EXP7025-SNAPSHOT-BLOB-JOIN preserves both path facts."""

    spec, blob = _snapshot_fixture(tmp_path)
    receipt = build_arc_model_identity_receipt(
        selected_model_spec=spec,
        observed_server_model_path=str(blob),
    )

    assert receipt == {
        "requested_model_path": spec["model_path"],
        "requested_model_filename": FILENAME,
        "requested_hf_id": HF_ID,
        "requested_revision": REVISION,
        "observed_server_model_path": str(blob),
        "resolved_model_path": str(blob),
        "model_file_hash": spec["model_file_hash"],
    }
    assert not Path(receipt["observed_server_model_path"]).suffix


@pytest.mark.parametrize(
    "case",
    [
        "wrong_hash",
        "wrong_hub",
        "wrong_revision",
        "missing_requested_file",
        "broken_symlink",
        "directory_path",
        "misleading_gguf_basename",
        "observed_blob_not_reachable",
    ],
)
def test_identity_negative_matrix_fails_closed(tmp_path: Path, case: str) -> None:
    """SCENARIO-ARC-7030-IDENTITY-NEGATIVE-MATRIX rejects every named defect."""

    spec, blob = _snapshot_fixture(tmp_path)
    observed = blob
    if case == "wrong_hash":
        spec["model_file_hash"] = "sha256:" + "0" * 64
    elif case == "wrong_hub":
        spec["hf_id"] = "unsloth/Not-Qwen-GGUF"
    elif case == "wrong_revision":
        spec["revision"] = "wrong-revision"
    elif case == "missing_requested_file":
        Path(str(spec["model_path"])).unlink()
    elif case == "broken_symlink":
        Path(str(spec["model_path"])).unlink()
        Path(str(spec["model_path"])).symlink_to("../../blobs/missing")
    elif case == "directory_path":
        Path(str(spec["model_path"])).unlink()
        Path(str(spec["model_path"])).mkdir()
    elif case == "misleading_gguf_basename":
        spec["model_filename"] = "alias.gguf"
    else:
        foreign = tmp_path / "foreign" / "blobs" / blob.name
        foreign.parent.mkdir(parents=True)
        foreign.write_bytes(blob.read_bytes())
        observed = foreign

    with pytest.raises(ValueError):
        build_arc_model_identity_receipt(
            selected_model_spec=spec,
            observed_server_model_path=str(observed),
        )


def test_current_provenance_embeds_shared_identity_and_rejects_aliases(tmp_path: Path) -> None:
    """REQ-ARC-7030 extends the strict record instead of creating a side receipt."""

    spec, blob = _snapshot_fixture(tmp_path)
    identity = build_arc_model_identity_receipt(
        selected_model_spec=spec,
        observed_server_model_path=str(blob),
    )
    record = build_arc_eval_provenance(
        replace(
            _legacy_input(
                model_hash=identity["model_file_hash"],
                model_filename=identity["requested_model_filename"],
            ),
            schema_version=ARC_EVAL_PROVENANCE_SCHEMA_VERSION_V2,
            **identity,
        )
    )

    assert record["schema_version"] == ARC_EVAL_PROVENANCE_SCHEMA_VERSION_V2
    assert all(record[key] == value for key, value in identity.items())
    assert validate_arc_eval_provenance(record).valid

    aliased = dict(record)
    aliased["server_model_alias"] = aliased.pop("observed_server_model_path")
    _rehash(aliased)
    assert not validate_arc_eval_provenance(aliased).valid


def test_complete_legacy_rows_use_the_explicit_v1_branch_unchanged() -> None:
    """SCENARIO-ARC-7030-LEGACY-VERSION-READ does not guess current fields."""

    legacy = build_arc_eval_provenance(_legacy_input())
    before = json.dumps(legacy, sort_keys=True)
    decision = validate_arc_eval_provenance(json.loads(before))

    assert legacy["schema_version"] == ARC_EVAL_PROVENANCE_SCHEMA_VERSION
    assert decision.valid
    assert "requested_model_path" not in legacy
    assert json.dumps(legacy, sort_keys=True) == before

    malformed = dict(legacy, requested_model_path="/tmp/alias.gguf")
    _rehash(malformed)
    assert not validate_arc_eval_provenance(malformed).valid


def test_policy_builder_and_both_live_consumers_use_the_shared_bridge() -> None:
    """SCENARIO-ARC-7030-SHARED-PRODUCER-WIRING proves the production call chain."""

    builder_source = inspect.getsource(provenance.build_arc_eval_provenance_for_policy)
    shadow_source = (ROOT / "python/carnot/agentic/arc_belief_shadow_live_trace.py").read_text()
    submitted_source = (ROOT / "scripts/arc_leaderboard_eval.py").read_text()

    assert "build_arc_model_identity_receipt" in builder_source
    assert "build_arc_model_identity_receipt" in shadow_source
    assert "build_arc_eval_provenance_for_policy" in submitted_source
    assert "arc_eval_provenance" in submitted_source
    assert "model_alias" not in builder_source


def test_exp7030_artifact_is_complete_and_independently_validated(tmp_path: Path) -> None:
    """REQ-ARC-7030 sets readiness only after all fixture and wiring gates pass."""

    artifact = experiment.build_artifact(ROOT, execution_date="20260905", work_dir=tmp_path)

    assert experiment.validate_artifact(artifact) == []
    assert set(experiment.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert artifact["identity_schema_version"] == ARC_EVAL_PROVENANCE_SCHEMA_VERSION_V2
    assert artifact["arc_model_identity_bridge_ready_score"] == 1
    assert all(not row["accepted"] for row in artifact["negative_fixture_rows"])
    assert all(row["accepted"] for row in artifact["positive_fixture_rows"])
    assert all(row["accepted"] for row in artifact["legacy_compatibility_rows"])
    assert artifact["verdict_class"] == "positive"
    assert artifact["honest_verdict"].startswith("complete_positive_")


def test_exp7030_validator_rejects_forged_readiness(tmp_path: Path) -> None:
    """REQ-ARC-7030 recomputes receipt evidence instead of trusting the score."""

    artifact = experiment.build_artifact(ROOT, execution_date="20260905", work_dir=tmp_path)
    artifact["negative_fixture_rows"][0]["accepted"] = True
    artifact["reproducibility_checksum"] = experiment.artifact_checksum(artifact)

    assert "negative_fixture_accepted" in experiment.validate_artifact(artifact)


def test_exp7030_missing_source_writes_a_complete_blocked_artifact(tmp_path: Path) -> None:
    """REQ-ARC-7030 records the exact source precondition instead of inventing evidence."""

    root = tmp_path / "empty-root"
    (root / "results").mkdir(parents=True)
    artifact = experiment.build_artifact(
        root,
        execution_date="20260905",
        work_dir=tmp_path / "blocked-work",
    )

    assert artifact["arc_model_identity_bridge_ready_score"] == 0
    assert artifact["verdict_class"] == "blocked"
    assert artifact["gate_check_summary"]["passed"] is False
    assert artifact["gate_check_summary"]["failed_check"]
    assert experiment.validate_artifact(artifact) == []


def test_exp7030_source_readers_fail_closed_on_malformed_evidence(tmp_path: Path) -> None:
    """REQ-ARC-7030 source discovery returns absence for malformed evidence."""

    missing = tmp_path / "missing.json"
    malformed = tmp_path / "malformed.json"
    nonobject = tmp_path / "nonobject.json"
    malformed.write_text("{", encoding="utf-8")
    nonobject.write_text("[]", encoding="utf-8")
    assert experiment._load_json(missing) == {}
    assert experiment._load_json(malformed) == {}
    assert experiment._load_json(nonobject) == {}
    assert experiment._exp7025_snapshot_path({}) is None
    assert experiment._exp7025_snapshot_path(
        {"MODEL_SPECS": [{"hf_id": experiment.HF_ID, "model_path": "/tmp/not-a-blob"}]}
    ) is None

    model_root = tmp_path / ("models--" + experiment.HF_ID.replace("/", "--"))
    blob = model_root / "blobs" / ("0" * 64)
    blob.parent.mkdir(parents=True)
    blob.write_bytes(b"not-matching-hash")
    broken = model_root / "snapshots" / "revision" / "broken.gguf"
    broken.parent.mkdir(parents=True)
    broken.symlink_to("../../blobs/missing")
    assert experiment._exp7025_snapshot_path(
        {"MODEL_SPECS": [{"hf_id": experiment.HF_ID, "model_path": str(blob)}]}
    ) is None


@pytest.mark.parametrize(
    ("mutation", "expected"),
    [
        (lambda a: a.pop("rows"), "required_fields_missing"),
        (lambda a: a.__setitem__("field_principles", {}), "field_principles"),
        (lambda a: a.__setitem__("inference_substrate", "live_llm_inference"), "inference_substrate"),
        (lambda a: a.__setitem__("duration_s", -1), "duration_s"),
        (lambda a: a.__setitem__("identity_schema_version", "unknown"), "identity_schema_version"),
        (lambda a: a.__setitem__("verifier_is_oracle", True), "verifier_is_oracle"),
        (lambda a: a.__setitem__("arc_model_identity_bridge_ready_score", True), "ready_score"),
        (lambda a: a.__setitem__("positive_fixture_rows", []), "positive_fixture"),
        (lambda a: a.__setitem__("legacy_compatibility_rows", []), "legacy_compatibility"),
        (lambda a: a.__setitem__("producer_wiring_rows", []), "producer_wiring"),
        (lambda a: a.__setitem__("consumer_wiring_rows", []), "consumer_wiring"),
        (lambda a: a.__setitem__("hash_join_rows", []), "hash_join"),
        (lambda a: a.__setitem__("hub_revision_rows", []), "hub_revision"),
        (lambda a: a.__setitem__("gate_check_summary", {}), "gate_check_summary"),
        (lambda a: a.__setitem__("verdict_class", "partial"), "positive_verdict"),
        (lambda a: a.__setitem__("reproducibility_checksum", "sha256:wrong"), "checksum"),
    ],
)
def test_exp7030_validator_rejects_malformed_artifact_fields(
    tmp_path: Path, mutation, expected: str
) -> None:
    """REQ-ARC-7030 rejects missing, malformed, and contradictory result fields."""

    artifact = experiment.build_artifact(ROOT, execution_date="20260905", work_dir=tmp_path)
    mutation(artifact)
    if expected != "checksum":
        artifact["reproducibility_checksum"] = experiment.artifact_checksum(artifact)
    assert any(expected in error for error in experiment.validate_artifact(artifact))


def test_exp7030_validator_and_writer_defensive_paths(tmp_path: Path) -> None:
    """REQ-ARC-7030 validates before publish and writes through an atomic target."""

    assert experiment.validate_artifact([]) == ["artifact_object_required"]
    artifact = experiment.build_artifact(
        ROOT,
        execution_date="20260905",
        work_dir=tmp_path / "fixture",
    )
    output = tmp_path / "result.json"
    experiment.write_artifact(output, artifact)
    assert json.loads(output.read_text(encoding="utf-8")) == artifact

    invalid = dict(artifact)
    invalid["reproducibility_checksum"] = "bad"
    with pytest.raises(ValueError, match="invalid Exp7030 artifact"):
        experiment.write_artifact(tmp_path / "invalid.json", invalid)


def test_exp7030_run_uses_temporary_fixtures_and_writes_output(tmp_path: Path) -> None:
    """REQ-ARC-7030 exposes the no-GPU experiment through its canonical runner."""

    output = tmp_path / "run.json"
    artifact = experiment.run(ROOT, execution_date="20260905", output_path=output)

    assert output.is_file()
    assert artifact["arc_model_identity_bridge_ready_score"] == 1
    assert experiment.validate_artifact(artifact) == []


def test_live_model_resolver_keeps_snapshot_request_and_revision(tmp_path: Path) -> None:
    """REQ-ARC-7030 keeps the requested receipt before llama.cpp resolves its symlink."""

    spec, _blob = _snapshot_fixture(tmp_path)
    resolved = shadow.resolve_model_spec(
        lambda **_kwargs: [spec, {"hf_id": "unused/model", "model_path": "/missing"}],
        gpu_index=0,
    )

    assert resolved is not None
    assert resolved["model_path"] == spec["model_path"]
    assert resolved["model_filename"] == FILENAME
    assert resolved["revision"] == REVISION
