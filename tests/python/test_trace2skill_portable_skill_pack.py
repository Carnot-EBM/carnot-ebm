"""Tests for Exp 1514 trace2skill portable skill/provenance pack.

Spec: REQ-LEARN-1514, SCENARIO-LEARN-1516, SCENARIO-LEARN-1517.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from carnot.reporting import trace2skill_portable_skill_pack as mod


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in rows) + ("\n" if rows else ""),
        encoding="utf-8",
    )


def _rollback_artifact(*, passed: bool = True) -> dict[str, Any]:
    return {
        "schema": "fr11_policy_rollback_replay_audit_v1",
        "status": "complete" if passed else "blocked",
        "rollback_audit_passed": passed,
        "gated_inputs_present": True,
        "policy_updates_replayed": 1,
        "accepted_policy_updates": 1 if passed else 0,
        "rolled_back_policy_updates": 0 if passed else 1,
        "rollback_manifest_path": "rollback.jsonl",
        "blockers": [] if passed else ["rollback_failed"],
        "honest_verdict": "complete: fr11_policy_rollback_replay_audit_passed",
    }


def _reachability_artifact(*, clean: bool = True) -> dict[str, Any]:
    return {
        "schema": "trace2skill_artifact_reachability_audit_v1",
        "status": "complete",
        "artifact_reachability_audit_complete": clean,
        "gated_inputs_present": True,
        "reachable_artifact_count": 1 if clean else 0,
        "unreachable_artifact_count": 0 if clean else 1,
        "stale_artifact_count": 0,
        "ambiguous_resolver_count": 0,
        "resolver_keys": [
            "source_artifact_present",
            "paired_replay_case",
            "verifier_signal_present",
            "zero_soundness_policy_allowed",
        ],
        "source_artifact_audit": [
            {
                "status": "reachable" if clean else "unreachable",
                "path": "results/source.json",
                "referenced_as": "results/source.json",
            }
        ],
        "blockers": [] if clean else ["unreachable_source_artifact"],
        "honest_verdict": "complete: trace2skill_artifact_reachability_audit_passed",
    }


def _rollback_row(
    skill_id: str = "fr11_v10_trace2skill/case-a",
    *,
    decision: str = "keep",
    reachable: bool = True,
    stale: bool = False,
    deterministic: bool = True,
    soundness_mistakes: int = 0,
    false_accept_delta: int = 0,
    rollback_reasons: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "schema": "fr11_policy_rollback_replay_row_v1",
        "spec": ["REQ-LEARN-1513", "SCENARIO-LEARN-1514"],
        "run_date": "20260508",
        "skill_id": skill_id,
        "source_event_id": f"daily_eval:{skill_id.rsplit('/', 1)[-1]}",
        "source_case_id": skill_id.rsplit("/", 1)[-1],
        "source_kind": "daily_eval",
        "policy_action": "retrieval_boost",
        "decision": decision,
        "source_evidence_reachable": reachable,
        "source_evidence_stale": stale,
        "deterministic_validator_supported": deterministic,
        "soundness_mistakes": soundness_mistakes,
        "false_accept_delta": false_accept_delta,
        "utility_delta": 1,
        "rollback_reasons": list(rollback_reasons or []),
    }


def test_req_learn_1514_writes_in_progress_artifact_first(tmp_path: Path) -> None:
    """REQ-LEARN-1514-1/5: bootstrap artifact exposes required fields."""

    output_path = tmp_path / mod.OUTPUT_FILE
    manifest_path = tmp_path / mod.PACK_MANIFEST_FILE
    note_path = tmp_path / mod.OPS_NOTE_FILE

    artifact = mod.write_in_progress_artifact(
        output_path,
        pack_manifest_path=manifest_path,
        ops_note_path=note_path,
        project_root=tmp_path,
        run_date="20260508",
    )

    assert json.loads(output_path.read_text(encoding="utf-8")) == artifact
    assert artifact["status"] == "in_progress"
    assert artifact["portable_skill_pack_ready"] is False
    assert artifact["pack_manifest_path"] == mod.PACK_MANIFEST_FILE
    assert artifact["ops_note_path"] == mod.OPS_NOTE_FILE
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert field in artifact
    mod.validate_artifact(artifact)


def test_scenario_learn_1516_packages_only_eligible_rows(tmp_path: Path) -> None:
    """SCENARIO-LEARN-1516: rollback-passing rows become portable entries."""

    rows = [_rollback_row()]
    manifest = mod.build_pack_manifest(
        rows,
        reachability_artifact=_reachability_artifact(),
        rollback_manifest_path=tmp_path / mod.ROLLBACK_MANIFEST_FILE,
        project_root=tmp_path,
        run_date="20260508",
    )
    artifact = mod.build_artifact(
        rollback_artifact=_rollback_artifact(),
        reachability_artifact=_reachability_artifact(),
        pack_manifest=manifest,
        pack_manifest_path=tmp_path / mod.PACK_MANIFEST_FILE,
        ops_note_path=tmp_path / mod.OPS_NOTE_FILE,
        manifest_exists=True,
        gated_inputs_present=True,
        gate_blockers=[],
        project_root=tmp_path,
        run_date="20260508",
        tests_run=["focused pytest"],
    )

    entry = manifest["entries"][0]
    assert manifest["schema"] == mod.PACK_SCHEMA
    assert manifest["packaged_entry_count"] == 1
    assert manifest["rejected_entry_count"] == 0
    assert entry["skill_id"] == "fr11_v10_trace2skill/case-a"
    assert entry["source_artifact"] == mod.ROLLBACK_MANIFEST_FILE
    assert entry["reachable_source_artifacts"] == ["results/source.json"]
    assert entry["resolver_key"] == "daily_eval:case-a"
    assert entry["resolver_checks"] == _reachability_artifact()["resolver_keys"]
    assert entry["created_date"] == "20260508"
    assert entry["promotion_status"] == "packaged_rollback_passed"
    assert entry["verifier_evidence"]["rollback_decision"] == "keep"
    assert entry["verifier_evidence"]["deterministic_validator_supported"] is True
    assert artifact["portable_skill_pack_ready"] is True
    assert artifact["rollback_passing_entries"] == 1
    assert artifact["packaged_skill_entries"] == 1
    assert artifact["rejected_skill_entries"] == 0
    assert artifact["provenance_fields_present"] is True
    assert artifact["resolver_keys_present"] is True
    assert artifact["tests_run"] == ["focused pytest"]
    mod.validate_pack_manifest(manifest)
    _write_json(tmp_path / mod.PACK_MANIFEST_FILE, manifest)
    mod.validate_artifact(artifact, pack_manifest_path=tmp_path / mod.PACK_MANIFEST_FILE)


def test_scenario_learn_1517_rejects_unsupported_rows_without_promotion(
    tmp_path: Path,
) -> None:
    """SCENARIO-LEARN-1517: unsupported rows are rejected, not promoted."""

    rows = [
        _rollback_row("fr11_v10_trace2skill/rolled-back", decision="rollback"),
        _rollback_row("fr11_v10_trace2skill/unreachable", reachable=False),
        _rollback_row("fr11_v10_trace2skill/stale", stale=True),
        _rollback_row("fr11_v10_trace2skill/no-validator", deterministic=False),
        _rollback_row("fr11_v10_trace2skill/false-accept", false_accept_delta=1),
        _rollback_row("fr11_v10_trace2skill/soundness", soundness_mistakes=1),
        dict(_rollback_row("fr11_v10_trace2skill/missing-skill"), skill_id=""),
    ]

    manifest = mod.build_pack_manifest(
        rows,
        reachability_artifact=_reachability_artifact(),
        rollback_manifest_path=tmp_path / mod.ROLLBACK_MANIFEST_FILE,
        project_root=tmp_path,
        run_date="20260508",
    )
    artifact = mod.build_artifact(
        rollback_artifact=_rollback_artifact(),
        reachability_artifact=_reachability_artifact(),
        pack_manifest=manifest,
        pack_manifest_path=tmp_path / mod.PACK_MANIFEST_FILE,
        ops_note_path=tmp_path / mod.OPS_NOTE_FILE,
        manifest_exists=True,
        gated_inputs_present=True,
        gate_blockers=[],
        project_root=tmp_path,
        run_date="20260508",
    )
    rejected = {entry["skill_id"]: entry for entry in manifest["rejected_entries"]}

    assert manifest["entries"] == []
    assert set(rejected) == {row["skill_id"] for row in rows}
    assert rejected["fr11_v10_trace2skill/rolled-back"]["promotion_status"] == (
        "rejected_not_promoted"
    )
    assert (
        "rollback_decision_not_keep"
        in rejected["fr11_v10_trace2skill/rolled-back"]["rejection_reasons"]
    )
    assert (
        "source_evidence_unreachable"
        in rejected["fr11_v10_trace2skill/unreachable"]["rejection_reasons"]
    )
    assert "source_evidence_stale" in rejected["fr11_v10_trace2skill/stale"]["rejection_reasons"]
    assert (
        "missing_deterministic_validator_support"
        in rejected["fr11_v10_trace2skill/no-validator"]["rejection_reasons"]
    )
    assert (
        "false_accept_delta_positive"
        in rejected["fr11_v10_trace2skill/false-accept"]["rejection_reasons"]
    )
    assert "soundness_mistake" in rejected["fr11_v10_trace2skill/soundness"]["rejection_reasons"]
    assert artifact["portable_skill_pack_ready"] is True
    assert artifact["packaged_skill_entries"] == 0
    assert "missing_skill_id" in rejected[""]["rejection_reasons"]
    assert artifact["rejected_skill_entries"] == 7
    assert artifact["honest_verdict"] == mod.EMPTY_VERDICT


def test_req_learn_1514_run_writes_manifest_note_and_terminal_artifact(
    tmp_path: Path,
) -> None:
    """REQ-LEARN-1514-2/4/5: runner writes pack manifest, note, and artifact."""

    rollback_artifact_path = tmp_path / "exp1513.json"
    reachability_artifact_path = tmp_path / "exp1498.json"
    rollback_manifest_path = tmp_path / mod.ROLLBACK_MANIFEST_FILE
    pack_manifest_path = tmp_path / mod.PACK_MANIFEST_FILE
    ops_note_path = tmp_path / mod.OPS_NOTE_FILE
    output_path = tmp_path / mod.OUTPUT_FILE
    _write_json(rollback_artifact_path, _rollback_artifact())
    _write_json(reachability_artifact_path, _reachability_artifact())
    _write_jsonl(rollback_manifest_path, [_rollback_row()])

    artifact = mod.run(
        rollback_artifact_path=rollback_artifact_path,
        reachability_artifact_path=reachability_artifact_path,
        rollback_manifest_path=rollback_manifest_path,
        output_path=output_path,
        pack_manifest_path=pack_manifest_path,
        ops_note_path=ops_note_path,
        project_root=tmp_path,
        run_date="20260508",
        tests_run=["focused pytest"],
    )
    pack = json.loads(pack_manifest_path.read_text(encoding="utf-8"))
    note = ops_note_path.read_text(encoding="utf-8")

    assert json.loads(output_path.read_text(encoding="utf-8")) == artifact
    assert pack["entries"][0]["promotion_status"] == "packaged_rollback_passed"
    assert "Packaged entries: 1" in note
    assert "Rejected entries: 0" in note
    assert artifact["status"] == "complete"
    assert artifact["pack_manifest_path"] == mod.PACK_MANIFEST_FILE
    assert artifact["ops_note_path"] == mod.OPS_NOTE_FILE
    mod.validate_artifact(
        artifact,
        pack_manifest_path=pack_manifest_path,
        ops_note_path=ops_note_path,
    )


def test_req_learn_1514_gate_failures_are_terminal(tmp_path: Path) -> None:
    """REQ-LEARN-1514-2/5: absent or failed gates prevent packaging."""

    rollback_artifact_path = tmp_path / "exp1513.json"
    reachability_artifact_path = tmp_path / "exp1498.json"
    rollback_manifest_path = tmp_path / "missing.jsonl"
    _write_json(rollback_artifact_path, _rollback_artifact(passed=False))
    _write_json(reachability_artifact_path, _reachability_artifact(clean=False))

    artifact = mod.run(
        rollback_artifact_path=rollback_artifact_path,
        reachability_artifact_path=reachability_artifact_path,
        rollback_manifest_path=rollback_manifest_path,
        output_path=tmp_path / mod.OUTPUT_FILE,
        pack_manifest_path=tmp_path / mod.PACK_MANIFEST_FILE,
        ops_note_path=tmp_path / mod.OPS_NOTE_FILE,
        project_root=tmp_path,
        run_date="20260508",
    )

    assert artifact["status"] == "blocked"
    assert artifact["gated_inputs_present"] is False
    assert artifact["portable_skill_pack_ready"] is False
    assert artifact["packaged_skill_entries"] == 0
    assert "exp1513_rollback_audit_not_passed" in artifact["blockers"]
    assert "missing_rollback_manifest" in artifact["blockers"]
    assert "exp1498_reachability_not_clean" in artifact["blockers"]
    assert not (tmp_path / mod.PACK_MANIFEST_FILE).exists()

    missing = mod.run(
        rollback_artifact_path=tmp_path / "missing-exp1513.json",
        reachability_artifact_path=tmp_path / "missing-exp1498.json",
        rollback_manifest_path=rollback_manifest_path,
        output_path=tmp_path / "missing" / mod.OUTPUT_FILE,
        pack_manifest_path=tmp_path / "missing" / mod.PACK_MANIFEST_FILE,
        ops_note_path=tmp_path / "missing" / mod.OPS_NOTE_FILE,
        project_root=tmp_path,
        run_date="20260508",
    )
    assert "missing_exp1513_rollback_artifact" in missing["blockers"]
    assert "missing_exp1498_reachability_artifact" in missing["blockers"]


def test_req_learn_1514_validation_rejects_bad_contracts(tmp_path: Path) -> None:
    """REQ-LEARN-1514-4/5: validators enforce no false promotion."""

    manifest = mod.build_pack_manifest(
        [_rollback_row()],
        reachability_artifact=_reachability_artifact(),
        rollback_manifest_path=tmp_path / mod.ROLLBACK_MANIFEST_FILE,
        project_root=tmp_path,
        run_date="20260508",
    )
    artifact = mod.build_artifact(
        rollback_artifact=_rollback_artifact(),
        reachability_artifact=_reachability_artifact(),
        pack_manifest=manifest,
        pack_manifest_path=tmp_path / mod.PACK_MANIFEST_FILE,
        ops_note_path=tmp_path / mod.OPS_NOTE_FILE,
        manifest_exists=True,
        gated_inputs_present=True,
        gate_blockers=[],
        project_root=tmp_path,
    )

    with pytest.raises(AssertionError):
        mod.validate_pack_manifest(dict(manifest, entries=[{"skill_id": "missing-fields"}]))
    with pytest.raises(AssertionError):
        mod.validate_pack_manifest(
            dict(
                manifest,
                rejected_entries=[
                    {
                        "skill_id": "bad",
                        "promotion_status": "packaged_rollback_passed",
                        "rejection_reasons": ["should_not_promote"],
                    }
                ],
            )
        )
    with pytest.raises(AssertionError):
        mod.validate_artifact(dict(artifact, honest_verdict="blocked_without_prefix"))
    with pytest.raises(AssertionError):
        mod.validate_artifact(dict(artifact, packaged_skill_entries=2))
    with pytest.raises(AssertionError):
        mod.validate_artifact(dict(artifact, portable_skill_pack_ready=True, blockers=["bad"]))
    with pytest.raises(AssertionError):
        mod.validate_artifact(
            dict(
                artifact,
                packaged_skill_entries=0,
                rejected_skill_entries=0,
                portable_skill_pack_ready=False,
            )
        )


def test_req_learn_1514_defensive_edges_are_deterministic(tmp_path: Path) -> None:
    """REQ-LEARN-1514-2/4/5: malformed helper inputs fail closed."""

    outside = tmp_path / "outside" / mod.PACK_MANIFEST_FILE
    assert mod._display_path(outside, project_root=tmp_path / "root") == mod.PACK_MANIFEST_FILE

    array_json = tmp_path / "array.json"
    array_json.write_text("[]", encoding="utf-8")
    with pytest.raises(AssertionError):
        mod._load_json(array_json)

    blank_jsonl = tmp_path / "blank.jsonl"
    blank_jsonl.write_text("\n{}\n", encoding="utf-8")
    assert mod._load_jsonl(blank_jsonl) == [{}]
    bad_jsonl = tmp_path / "bad.jsonl"
    bad_jsonl.write_text("[]\n", encoding="utf-8")
    with pytest.raises(AssertionError):
        mod._load_jsonl(bad_jsonl)

    sparse_reachability = {"source_artifact_audit": "bad", "resolver_keys": "bad"}
    no_resolver_manifest = mod.build_pack_manifest(
        [_rollback_row("fr11_v10_trace2skill/no-resolver")],
        reachability_artifact=sparse_reachability,
        rollback_manifest_path=tmp_path / mod.ROLLBACK_MANIFEST_FILE,
        project_root=tmp_path,
    )
    blocked = mod.build_artifact(
        rollback_artifact=_rollback_artifact(),
        reachability_artifact=_reachability_artifact(),
        pack_manifest=no_resolver_manifest,
        pack_manifest_path=tmp_path / mod.PACK_MANIFEST_FILE,
        ops_note_path=tmp_path / mod.OPS_NOTE_FILE,
        manifest_exists=True,
        gated_inputs_present=True,
        gate_blockers=[],
        project_root=tmp_path,
    )
    assert blocked["blockers"] == ["resolver_keys_missing"]

    quarantined_manifest = mod.build_pack_manifest(
        [
            dict(
                _rollback_row("fr11_v10_trace2skill/quarantined"),
                exp1512_quarantined=True,
                rollback_reasons=["manual_rollback"],
            )
        ],
        reachability_artifact=_reachability_artifact(),
        rollback_manifest_path=tmp_path / mod.ROLLBACK_MANIFEST_FILE,
        project_root=tmp_path,
    )
    rejection = quarantined_manifest["rejected_entries"][0]
    assert "exp1512_quarantined" in rejection["rejection_reasons"]
    assert "rollback_reason:manual_rollback" in rejection["rejection_reasons"]
    note = mod.write_ops_note(
        quarantined_manifest,
        dict(blocked, pack_manifest_path=mod.PACK_MANIFEST_FILE),
        tmp_path / mod.OPS_NOTE_FILE,
    )
    assert "Rejected rows remain unpromoted" in note

    weird = mod.build_artifact(
        rollback_artifact=_rollback_artifact(),
        reachability_artifact=_reachability_artifact(),
        pack_manifest={"entries": "bad", "rejected_entries": "bad", "resolver_keys": []},
        pack_manifest_path=tmp_path / mod.PACK_MANIFEST_FILE,
        ops_note_path=tmp_path / mod.OPS_NOTE_FILE,
        manifest_exists=True,
        gated_inputs_present=True,
        gate_blockers=[],
        project_root=tmp_path,
    )
    assert weird["honest_verdict"] == mod.EMPTY_VERDICT

    with pytest.raises(AssertionError):
        mod.build_artifact(
            rollback_artifact=_rollback_artifact(),
            reachability_artifact=_reachability_artifact(),
            pack_manifest=mod._empty_pack_manifest("20260508"),
            pack_manifest_path=tmp_path / mod.PACK_MANIFEST_FILE,
            ops_note_path=tmp_path / mod.OPS_NOTE_FILE,
            manifest_exists=False,
            gated_inputs_present=True,
            gate_blockers=[],
            project_root=tmp_path,
        )

    valid_manifest = mod.build_pack_manifest(
        [_rollback_row("fr11_v10_trace2skill/valid")],
        reachability_artifact=_reachability_artifact(),
        rollback_manifest_path=tmp_path / mod.ROLLBACK_MANIFEST_FILE,
        project_root=tmp_path,
    )
    valid_entry = valid_manifest["entries"][0]
    valid_rejected = quarantined_manifest["rejected_entries"][0]

    bad_manifests = [
        dict(valid_manifest, schema="bad"),
        dict(valid_manifest, entries="bad"),
        dict(valid_manifest, rejected_entries="bad"),
        dict(valid_manifest, packaged_entry_count=2),
        dict(valid_manifest, rejected_entry_count=1),
        dict(valid_manifest, entries=[1]),
        dict(valid_manifest, entries=[dict(valid_entry, promotion_status="bad")]),
        dict(valid_manifest, entries=[dict(valid_entry, resolver_key="")]),
        dict(valid_manifest, rejected_entries=[1], rejected_entry_count=1),
        dict(
            valid_manifest,
            rejected_entries=[dict(valid_rejected, promotion_status="bad")],
            rejected_entry_count=1,
        ),
        dict(
            valid_manifest,
            rejected_entries=[dict(valid_rejected, rejection_reasons=[])],
            rejected_entry_count=1,
        ),
    ]
    for bad_manifest in bad_manifests:
        with pytest.raises(AssertionError):
            mod.validate_pack_manifest(bad_manifest)

    valid_artifact = mod.build_artifact(
        rollback_artifact=_rollback_artifact(),
        reachability_artifact=_reachability_artifact(),
        pack_manifest=valid_manifest,
        pack_manifest_path=tmp_path / mod.PACK_MANIFEST_FILE,
        ops_note_path=tmp_path / mod.OPS_NOTE_FILE,
        manifest_exists=True,
        gated_inputs_present=True,
        gate_blockers=[],
        project_root=tmp_path,
    )
    pack_file = tmp_path / mod.PACK_MANIFEST_FILE
    note_file = tmp_path / mod.OPS_NOTE_FILE
    _write_json(pack_file, valid_manifest)
    note_file.parent.mkdir(parents=True, exist_ok=True)
    note_file.write_text("note\n", encoding="utf-8")

    missing_field = dict(valid_artifact)
    del missing_field["status"]
    bad_artifacts = [
        missing_field,
        dict(valid_artifact, status="bad"),
        dict(valid_artifact, packaged_skill_entries=-1, rollback_passing_entries=-1),
        dict(valid_artifact, provenance_fields_present=False),
        dict(valid_artifact, resolver_keys_present=False),
    ]
    for bad_artifact in bad_artifacts:
        with pytest.raises(AssertionError):
            mod.validate_artifact(bad_artifact)
    with pytest.raises(AssertionError):
        mod.validate_artifact(valid_artifact, pack_manifest_path=tmp_path / "missing.json")
    with pytest.raises(AssertionError):
        mod.validate_artifact(valid_artifact, ops_note_path=tmp_path / "missing.md")
    with pytest.raises(AssertionError):
        mod.validate_artifact(dict(valid_artifact, portable_skill_pack_ready=False, blockers=[]))
    with pytest.raises(AssertionError):
        mod.validate_artifact(
            dict(
                valid_artifact,
                portable_skill_pack_ready=False,
                blockers=["blocked"],
                packaged_skill_entries=0,
                rollback_passing_entries=0,
                rejected_skill_entries=0,
            )
        )
