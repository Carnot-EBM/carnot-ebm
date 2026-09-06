"""Tests for REQ-ARC-7052 typed raw model identity evidence."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path

import pytest

from carnot import experiment_7052_v618_typed_identity_attack_audit as experiment
from carnot.agentic import arc_eval_provenance as provenance


ROOT = Path(__file__).resolve().parents[2]
HF_ID = "frozen-owner/frozen-model-GGUF"
REVISION = "7052frozenrevision"
FILENAME = "frozen-model.gguf"
PAYLOAD = b"frozen Exp7051 GGUF report fixture bytes\n"
IDENTITY_FIELDS = ("model_path", "model", "model_alias")
ATTACKS = {
    "relative_alias",
    "broken_link",
    "wrong_snapshot",
    "same_size_different_content",
    "changed_hub",
    "changed_revision",
    "conflicting_props_fields",
    "hard_link_ambiguity",
    "symlink_swap",
    "missing_evidence",
    "artifact_checksum_change",
}


def _hash_bytes(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def _write_source_artifact(path: Path, raw_props: dict[str, object]) -> None:
    """Write a deterministic source object with the Exp7051 checksum projection."""

    body: dict[str, object] = {
        "raw_server_props": raw_props,
        "checksum_recomputation_rows": [],
        "reproducibility_checksum": "",
    }
    body["reproducibility_checksum"] = provenance.identity_source_payload_sha256(body)
    path.write_text(json.dumps(body, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def _snapshot_fixture(
    directory: Path,
) -> tuple[dict[str, str], Path, Path, dict[str, object], Path]:
    """Freeze the Exp7051 identity shape without reading its mutable result file."""

    model_root = directory / "hub" / "models--frozen-owner--frozen-model-GGUF"
    digest = hashlib.sha256(PAYLOAD).hexdigest()
    blob = model_root / "blobs" / digest
    blob.parent.mkdir(parents=True)
    blob.write_bytes(PAYLOAD)
    requested = model_root / "snapshots" / REVISION / FILENAME
    requested.parent.mkdir(parents=True)
    requested.symlink_to(Path("../../blobs") / digest)
    raw_props: dict[str, object] = {
        "model_path": str(requested),
        "model_alias": FILENAME,
    }
    source_path = directory / "frozen-exp7051-source.json"
    _write_source_artifact(source_path, raw_props)
    spec = {
        "model_path": str(requested),
        "model_filename": FILENAME,
        "hf_id": HF_ID,
        "revision": REVISION,
        "model_file_hash": _hash_bytes(PAYLOAD),
    }
    return spec, requested, blob, raw_props, source_path


def _direct_fixture(
    directory: Path,
) -> tuple[dict[str, str], Path, dict[str, object], Path]:
    """Create the accepted direct regular file branch inside a selected snapshot."""

    requested = (
        directory
        / "hub"
        / "models--frozen-owner--frozen-model-GGUF"
        / "snapshots"
        / REVISION
        / FILENAME
    )
    requested.parent.mkdir(parents=True)
    requested.write_bytes(PAYLOAD)
    raw_props: dict[str, object] = {
        "model_path": str(requested),
        "model_alias": FILENAME,
    }
    source_path = directory / "frozen-exp7051-source.json"
    _write_source_artifact(source_path, raw_props)
    return (
        {
            "model_path": str(requested),
            "model_filename": FILENAME,
            "hf_id": HF_ID,
            "revision": REVISION,
            "model_file_hash": _hash_bytes(PAYLOAD),
        },
        requested,
        raw_props,
        source_path,
    )


def _receipt(
    spec: dict[str, str],
    raw_props: dict[str, object],
    source_path: Path,
) -> dict[str, object]:
    source = provenance.capture_arc_model_identity_source_provenance(
        raw_server_props=raw_props,
        requested_model_path=spec["model_path"],
        source_kind="frozen_exp7051_report_fixture",
        source_artifact_path=source_path,
    )
    return provenance.build_typed_arc_model_identity_receipt(
        selected_model_spec=spec,
        launch_model_argument=spec["model_path"],
        raw_server_props=raw_props,
        source_provenance=source,
    )


def test_req_arc_7052_spec_exists_before_implementation() -> None:
    """REQ-ARC-7052 names the raw, attack, process, legacy, and block contracts."""

    text = (ROOT / "openspec/capabilities/arc-agi/spec.md").read_text(encoding="utf-8")
    assert "REQ-ARC-7052" in text
    for scenario in (
        "SCENARIO-ARC-7052-FROZEN-RAW-REPORT",
        "SCENARIO-ARC-7052-THREE-POSITIVE-PATHS",
        "SCENARIO-ARC-7052-ONE-FACTOR-ATTACKS",
        "SCENARIO-ARC-7052-COLD-BYTE-AGREEMENT",
        "SCENARIO-ARC-7052-LEGACY-AND-WIRING",
        "SCENARIO-ARC-7052-BLOCKED-PRECONDITION",
    ):
        assert scenario in text


def test_frozen_raw_report_stays_distinct_from_canonical_paths(tmp_path: Path) -> None:
    """SCENARIO-ARC-7052-FROZEN-RAW-REPORT preserves the exact three-field shape."""

    spec, requested, blob, raw_props, source_path = _snapshot_fixture(tmp_path)
    receipt = _receipt(spec, raw_props, source_path)
    decision = provenance.validate_typed_arc_model_identity_receipt(receipt)

    assert decision.valid, decision.errors
    assert receipt["identity_schema_version"] == provenance.ARC_MODEL_IDENTITY_SCHEMA_VERSION
    assert receipt["requested_model_path"] == str(requested)
    assert receipt["launch_model_argument"] == str(requested)
    assert receipt["observed_server_model_path"] == str(requested)
    assert receipt["observed_server_resolved_path"] == str(blob)
    assert receipt["resolved_model_path"] == str(blob)
    assert receipt["model_file_hash"] == _hash_bytes(PAYLOAD)
    rows = {row["field"]: row for row in receipt["raw_report_observations"]}
    assert tuple(rows) == IDENTITY_FIELDS
    assert rows["model_path"]["raw_value"] == str(requested)
    assert rows["model"]["present"] is False
    assert rows["model"]["raw_value"] is None
    assert rows["model_alias"]["raw_value"] == FILENAME
    assert rows["model_alias"]["raw_kind"] == "relative_path"


@pytest.mark.parametrize("path_form", ["snapshot_alias", "canonical_blob", "direct_file"])
def test_all_explicit_positive_path_forms_keep_every_obligation_supported(
    tmp_path: Path, path_form: str
) -> None:
    """SCENARIO-ARC-7052-THREE-POSITIVE-PATHS keeps all ten checks active."""

    if path_form == "direct_file":
        spec, requested, raw_props, source_path = _direct_fixture(tmp_path)
    else:
        spec, requested, blob, raw_props, source_path = _snapshot_fixture(tmp_path)
        if path_form == "canonical_blob":
            raw_props = {"model_path": str(blob), "model_alias": FILENAME}
            _write_source_artifact(source_path, raw_props)
    receipt = _receipt(spec, raw_props, source_path)
    rows = receipt["identity_obligation_rows"]

    assert receipt["path_form"] == path_form
    assert {row["obligation"] for row in rows} == set(provenance.IDENTITY_OBLIGATIONS)
    assert all(row["status"] == "supported" for row in rows)
    assert all(row["evidence_source"] for row in rows)
    assert provenance.validate_typed_arc_model_identity_receipt(receipt).valid
    assert Path(str(receipt["requested_model_path"])) == requested


def _attacked_receipt(tmp_path: Path, case: str) -> dict[str, object]:
    """Change one factor in an otherwise valid frozen fixture."""

    spec, requested, blob, raw_props, source_path = _snapshot_fixture(tmp_path)
    source = provenance.capture_arc_model_identity_source_provenance(
        raw_server_props=raw_props,
        requested_model_path=spec["model_path"],
        source_kind="frozen_exp7051_report_fixture",
        source_artifact_path=source_path,
    )
    launch = spec["model_path"]
    if case == "relative_alias":
        raw_props = {"model_path": FILENAME, "model_alias": FILENAME}
    elif case == "broken_link":
        blob.unlink()
    elif case == "wrong_snapshot":
        spec["model_path"] = spec["model_path"].replace(REVISION, "wrong-snapshot")
    elif case == "same_size_different_content":
        changed = b"X" * len(PAYLOAD)
        changed_path = blob.with_name(hashlib.sha256(changed).hexdigest())
        changed_path.write_bytes(changed)
        raw_props = {"model_path": str(changed_path), "model_alias": FILENAME}
    elif case == "changed_hub":
        spec["hf_id"] = "other-owner/frozen-model-GGUF"
    elif case == "changed_revision":
        spec["revision"] = "changed-revision"
    elif case == "conflicting_props_fields":
        other = tmp_path / "other.gguf"
        other.write_bytes(PAYLOAD)
        raw_props["model"] = str(other)
    elif case == "hard_link_ambiguity":
        os.link(blob, blob.with_name("second-link"))
    elif case == "symlink_swap":
        other_root = tmp_path / "other-hub" / "blobs"
        other_root.mkdir(parents=True)
        other_blob = other_root / blob.name
        other_blob.write_bytes(PAYLOAD)
        requested.unlink()
        requested.symlink_to(other_blob)
    elif case == "missing_evidence":
        raw_props = {"model_alias": FILENAME}
    elif case == "artifact_checksum_change":
        data = json.loads(source_path.read_text(encoding="utf-8"))
        data["post_capture_change"] = True
        source_path.write_text(json.dumps(data, sort_keys=True), encoding="utf-8")
    else:  # pragma: no cover - the parameter set is closed above
        raise AssertionError(case)
    return provenance.build_typed_arc_model_identity_receipt(
        selected_model_spec=spec,
        launch_model_argument=launch,
        raw_server_props=raw_props,
        source_provenance=source,
    )


@pytest.mark.parametrize("case", sorted(ATTACKS))
def test_one_factor_attacks_fail_closed(tmp_path: Path, case: str) -> None:
    """SCENARIO-ARC-7052-ONE-FACTOR-ATTACKS rejects every named mutation."""

    receipt = _attacked_receipt(tmp_path, case)
    decision = provenance.validate_typed_arc_model_identity_receipt(receipt)

    assert decision.valid is False
    assert decision.headline_eligible is False
    assert any(row["status"] != "supported" for row in receipt["identity_obligation_rows"])
    assert decision.errors


def test_obligation_rows_are_recomputed_instead_of_trusted(tmp_path: Path) -> None:
    """REQ-ARC-7052 rejects a forged supported status and any omitted evidence."""

    spec, _requested, _blob, raw_props, source_path = _snapshot_fixture(tmp_path)
    receipt = _receipt(spec, raw_props, source_path)
    forged = deepcopy(receipt)
    forged["identity_obligation_rows"][0]["status"] = "contradicted"
    assert not provenance.validate_typed_arc_model_identity_receipt(forged).valid
    missing = deepcopy(receipt)
    del missing["source_provenance"]
    assert not provenance.validate_typed_arc_model_identity_receipt(missing).valid


def test_complete_legacy_reader_is_explicit_and_does_not_infer_fields() -> None:
    """SCENARIO-ARC-7052-LEGACY-AND-WIRING preserves complete old rows only."""

    old = experiment.complete_legacy_v1_fixture()
    reread = provenance.read_complete_legacy_arc_eval_provenance(old)

    assert reread == old
    assert reread is not old
    assert "launch_model_argument" not in reread
    with pytest.raises(ValueError, match="complete legacy"):
        provenance.read_complete_legacy_arc_eval_provenance({**old, "port": None})
    with pytest.raises(ValueError, match="legacy schema"):
        provenance.read_complete_legacy_arc_eval_provenance(
            {"schema_version": provenance.ARC_EVAL_PROVENANCE_SCHEMA_VERSION_V3}
        )


def test_all_current_consumers_use_the_shared_typed_builder_and_validator() -> None:
    """SCENARIO-ARC-7052-LEGACY-AND-WIRING keeps all consumers on shared code."""

    rows = experiment.consumer_wiring_rows(ROOT)

    assert {row["consumer"] for row in rows} == {
        "submitted_arc_evaluator",
        "reusable_server_check",
        "belief_shadow_consumer",
    }
    assert all(row["shared_builder"] is True for row in rows)
    assert all(row["shared_validator"] is True for row in rows)
    assert all(row["local_copy_present"] is False for row in rows)


def test_fresh_worker_recomputes_byte_identical_positive_and_all_attacks(
    tmp_path: Path,
) -> None:
    """SCENARIO-ARC-7052-COLD-BYTE-AGREEMENT audits a fresh interpreter."""

    shared = tmp_path / "shared"
    parent = experiment.execute_audit_fixture(shared)
    child, receipt = experiment.run_fresh_subprocess(shared)

    assert receipt["returncode"] == 0
    assert child["positive_obligation_bytes"] == parent["positive_obligation_bytes"]
    assert {row["case"] for row in child["attack_rows"]} == ATTACKS
    assert all(row["accepted"] is False for row in child["attack_rows"])
    assert child["subprocess_rows"][0]["fresh_process"] is True


def test_artifact_is_schema_complete_and_forged_readiness_is_rejected(tmp_path: Path) -> None:
    """REQ-ARC-7052 recomputes readiness from every required evidence family."""

    artifact = experiment.build_audit_artifact_from_frozen_fixture(
        execution_date="20260906", work_dir=tmp_path
    )

    assert artifact["typed_identity_attack_audit_ready_score"] == 1
    assert artifact["verdict_class"] == "positive"
    assert artifact["honest_verdict"].startswith("complete_positive")
    assert experiment.validate_artifact(artifact) == []
    forged = deepcopy(artifact)
    forged["attack_rows"][0]["accepted"] = True
    forged["reproducibility_checksum"] = experiment.artifact_checksum(forged)
    assert "attack_rows_invalid" in experiment.validate_artifact(forged)


def test_upstream_block_is_terminal_not_partial(tmp_path: Path) -> None:
    """SCENARIO-ARC-7052-BLOCKED-PRECONDITION keeps an upstream block terminal."""

    artifact = experiment.blocked_artifact(
        execution_date="20260906",
        checks=[experiment.gate_row("model_report_evidence_ready_score", 1, 0)],
        source_hashes={},
        duration_s=0.01,
    )

    assert artifact["typed_identity_attack_audit_ready_score"] == 0
    assert artifact["verdict_class"] == "blocked"
    assert artifact["honest_verdict"].startswith("blocked")
    assert artifact["gate_check_summary"]["failed_check"] == "model_report_evidence_ready_score"
    assert artifact["gate_check_summary"]["expected_value"] == 1
    assert artifact["gate_check_summary"]["observed_value"] == 0
    assert experiment.validate_artifact(artifact) == []
