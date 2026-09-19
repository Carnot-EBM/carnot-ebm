"""Tests for V650 exact-contract and evidence-custody qualification.

Spec refs: REQ-REPORT-7409 and SCENARIO-REPORT-7409-*.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest
import yaml

from carnot import experiment_7409_v650_evidence_custody as custody


ROOT = Path(__file__).resolve().parents[2]


def _roadmap() -> dict[str, object]:
    return yaml.safe_load((ROOT / custody.ROADMAP_PATH).read_text(encoding="utf-8"))


def _design() -> str:
    return (ROOT / custody.DESIGN_PATH).read_text(encoding="utf-8")


def _passing_receipts(tmp_path: Path) -> list[dict[str, object]]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for index, name in enumerate((*custody.AFFECTED_CHECK_NAMES, *custody.TERMINAL_CHECK_NAMES)):
        log = tmp_path / f"{index:02d}-{name}.log"
        log.write_text(f"{name}: pass\n", encoding="utf-8")
        rows.append(
            {
                "name": name,
                "command": name,
                "command_argv": [name],
                "command_environment": {"COVERAGE_FILE": str(tmp_path / ".coverage")},
                "scope": "fixture",
                "exit_code": 0,
                "duration_s": 0.01,
                "log_path": str(log),
                "log_sha256": custody.sha256_file(log),
                "passed": True,
                "timed_out": False,
            }
        )
    return rows


def test_contract_matches_twelve_rows_and_declared_producers() -> None:
    """REQ-REPORT-7409; SCENARIO-REPORT-7409-CONTRACT."""

    result = custody.compare_contract_authorities(_design(), _roadmap())

    assert result["passed"] is True
    assert result["milestone"] == "2026.09.650"
    assert len(result["contract_rows"]) == 12
    assert [row["task_id"] for row in result["contract_rows"]] == list(custody.EXPECTED_TASK_IDS)
    assert all(row["producer_fields_declared"] for row in result["contract_rows"])
    assert result["advisory_only"] is True


def test_contract_mutations_are_rejected() -> None:
    """REQ-REPORT-7409; SCENARIO-REPORT-7409-CONTRACT."""

    rows = custody.run_contract_mutation_controls(_design(), _roadmap())

    assert [row["mutation"] for row in rows] == [
        "removed",
        "reordered",
        "stale_milestone",
        "wrong_field",
        "quarantined_input",
    ]
    assert all(row["rejected"] is True for row in rows)

    stale_markdown = _design().replace("2026.09.650", "2026.09.649", 1)
    stale_result = custody.compare_contract_authorities(stale_markdown, _roadmap())
    assert "markdown_milestone_mismatch" in stale_result["errors"]
    assert "markdown_task_count_mismatch" not in stale_result["errors"]

    malformed = custody.compare_contract_authorities("not a contract", {"tasks": "bad"})
    assert any(error.startswith("markdown_parse_error:") for error in malformed["errors"])
    assert any(error.startswith("yaml_parse_error:") for error in malformed["errors"])
    assert "markdown_task_count_mismatch" in malformed["errors"]
    assert "markdown_order_mismatch" in malformed["errors"]


def test_missing_v649_producers_keep_supporting_evidence_separate() -> None:
    """REQ-REPORT-7409; SCENARIO-REPORT-7409-MISSING."""

    rows = custody.inspect_missing_producers(ROOT)

    assert [row["experiment_id"] for row in rows] == [
        "exp7397-delayed-adapter",
        "exp7399-online-trial",
    ]
    assert all(row["terminal_state"] == "missing" for row in rows)
    assert all(row["terminal_tracked"] is False for row in rows)
    assert all(row["producer_module_state"] == "present_tracked" for row in rows)
    assert all(row["producer_entrypoint_state"] == "present_tracked" for row in rows)
    assert all(row["checkpoint_file_count"] > 0 for row in rows)
    assert all(row["raw_evidence_file_count"] > 0 for row in rows)
    assert rows[0]["recorded_terminal_sha256"] == (
        "sha256:c4e5139f7f3c600bd129e7e8425f362bac458796ae6a7db6052d6895140b2e3b"
    )
    assert rows[1]["recorded_terminal_sha256"] is None
    assert all(row["science_eligible"] is False for row in rows)
    assert all(row["recovery_performed"] is False for row in rows)
    assert all(row["history_rewrite_cause_established"] is False for row in rows)


def test_bundle_restart_recovers_each_completed_unit_once(tmp_path: Path) -> None:
    """REQ-REPORT-7409; SCENARIO-REPORT-7409-BUNDLE."""

    bundle = custody.EvidenceBundle(tmp_path / "bundle", protocol_id="fixture-v1")
    first = {"unit_id": "u1", "value": 3.5, "status": "complete"}
    second = {"unit_id": "u2", "value": -1.0, "status": "complete"}

    bundle.record_unit(first)
    bundle.record_unit(first)
    restarted = custody.EvidenceBundle(tmp_path / "bundle", protocol_id="fixture-v1")
    restarted.record_unit(second)

    assert restarted.recover_rows() == [first, second]
    manifest = restarted.finalize()
    assert manifest["completed_unit_count"] == 2
    assert custody.verify_bundle(tmp_path / "bundle", expected_protocol_id="fixture-v1") == []
    with pytest.raises(ValueError, match="unit_conflict"):
        restarted.record_unit({**first, "value": 9.0})


def test_bundle_fault_controls_reject_every_defect(tmp_path: Path) -> None:
    """REQ-REPORT-7409; SCENARIO-REPORT-7409-BUNDLE; SCENARIO-REPORT-7409-SIZE."""

    rows = custody.run_bundle_fault_controls(tmp_path / "faults")

    assert [row["fault"] for row in rows] == [
        "kill_before_rename",
        "missing_sidecar",
        "corrupted_bytes",
        "stale_completion",
        "oversized_payload",
    ]
    assert all(row["passed"] is True for row in rows)
    assert rows[0]["terminal_path_observed"] is False
    assert rows[-1]["observed_bytes"] > custody.RESULT_SIZE_LIMIT_BYTES


def test_bundle_rejects_invalid_rows_and_checkpoint_shapes(tmp_path: Path) -> None:
    """REQ-REPORT-7409; SCENARIO-REPORT-7409-BUNDLE."""

    bundle = custody.EvidenceBundle(tmp_path / "bundle", protocol_id="fixture-v1")
    with pytest.raises(ValueError, match="unit_id_required"):
        bundle.record_unit({"value": 1})
    bundle.record_unit({"unit_id": "u1", "value": 1})
    checkpoint = tmp_path / "bundle" / custody.BUNDLE_CHECKPOINT_NAME
    checkpoint.write_text("[]\n", encoding="utf-8")

    with pytest.raises(ValueError, match="checkpoint_mapping_required"):
        bundle.recover_rows()

    checkpoint.write_text("{bad", encoding="utf-8")
    with pytest.raises(ValueError, match="checkpoint_mapping_required"):
        bundle.recover_rows()

    checkpoint.write_text(
        json.dumps({"protocol_id": "fixture-v1", "completed_units": {}}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="checkpoint_units_required"):
        bundle.recover_rows()


def test_bundle_reader_rejects_duplicate_identity_and_manifest_defects(tmp_path: Path) -> None:
    """REQ-REPORT-7409; SCENARIO-REPORT-7409-BUNDLE; SCENARIO-REPORT-7409-SIZE."""

    assert custody.verify_bundle(tmp_path / "absent", expected_protocol_id="v1") == [
        "missing_manifest"
    ]
    bundle = custody.EvidenceBundle(tmp_path / "bundle", protocol_id="v1")
    reference = bundle.record_unit({"unit_id": "u1", "value": 1})
    bundle.finalize()
    with pytest.raises(ValueError, match="checkpoint_unit_duplicate"):
        bundle._read_references([reference, reference])

    row_path = bundle.directory / reference["path"]
    row_path.write_text('{"unit_id":"wrong"}\n', encoding="utf-8")
    changed = {**reference, "sha256": custody.sha256_file(row_path)}
    with pytest.raises(ValueError, match="unit_identity_mismatch"):
        bundle._read_references([changed])

    row_path.write_text('{"unit_id":"u1"}\n', encoding="utf-8")
    with row_path.open("r+b") as stream:
        stream.truncate(custody.RESULT_SIZE_LIMIT_BYTES + 1)
    manifest = json.loads(bundle.manifest_path.read_text(encoding="utf-8"))
    manifest["raw_files"][0]["sha256"] = custody.sha256_file(row_path)
    bundle.manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    errors = custody.verify_bundle(bundle.directory, expected_protocol_id="v1")
    assert "oversized_payload" in errors

    checkpoint = bundle.directory / custody.BUNDLE_CHECKPOINT_NAME
    checkpoint.write_text("{}\n", encoding="utf-8")
    errors = custody.verify_bundle(bundle.directory, expected_protocol_id="v1")
    assert "checkpoint_hash_mismatch" in errors


def test_artifact_fixture_validates_and_detects_drift(tmp_path: Path) -> None:
    """REQ-REPORT-7409; SCENARIO-REPORT-7409-ARTIFACT."""

    receipts = _passing_receipts(tmp_path / "logs")
    artifact = custody.build_artifact_for_test(ROOT, tmp_path / "bundle", receipts)

    assert custody.validate_artifact(artifact, root=ROOT) == []
    assert artifact["evidence_custody_ready_score"] == 1
    assert artifact["promotion_score"] == 0
    assert artifact["verdict_class"] == "null"
    assert artifact["MODEL_SPECS"] == []
    assert artifact["model_invoked"] is False
    assert artifact["inference_substrate"] == "aggregation_from_upstream_artifacts"
    assert artifact["inference_substrate_class"] == "aggregation"
    assert artifact["execution_venue"] == "host"
    assert all(row["science_eligible"] is False for row in artifact["missing_producer_rows"])

    changed = deepcopy(artifact)
    changed["contract_rows"][0]["title"] = "changed"
    assert "contract_rows_mismatch" in custody.validate_artifact(changed, root=ROOT)

    changed = deepcopy(artifact)
    changed["evidence_custody_ready_score"] = 0
    assert "custody_score_mismatch" in custody.validate_artifact(changed, root=ROOT)

    changed = deepcopy(artifact)
    changed["promotion_score"] = 1
    assert "promotion_score_nonzero" in custody.validate_artifact(changed, root=ROOT)


def test_artifact_validation_rejects_identity_receipt_and_hash_errors(tmp_path: Path) -> None:
    """REQ-REPORT-7409; SCENARIO-REPORT-7409-ARTIFACT."""

    artifact = custody.build_artifact_for_test(
        ROOT, tmp_path / "bundle", _passing_receipts(tmp_path / "logs")
    )

    changed = deepcopy(artifact)
    changed.pop("schema")
    assert custody.validate_artifact(changed, root=ROOT) == ["missing_required_field:schema"]

    changed = deepcopy(artifact)
    changed["invocation_counts"]["generation_calls_attempted"] = 1
    assert "current_work_receipt_invalid" in custody.validate_artifact(changed, root=ROOT)

    changed = deepcopy(artifact)
    first_source = next(iter(changed["source_artifact_hashes"].values()))
    first_source["sha256"] = "sha256:" + "0" * 64
    assert "source_hash_mismatch" in custody.validate_artifact(changed, root=ROOT)

    changed = deepcopy(artifact)
    changed["source_artifact_hashes"] = []
    assert "source_hash_mismatch" in custody.validate_artifact(changed, root=ROOT)
    assert custody._source_hashes_valid({"bad": []}, ROOT) is False

    changed = deepcopy(artifact)
    changed["validation_receipts"][0]["log_sha256"] = "sha256:" + "0" * 64
    assert "validation_log_hash_mismatch" in custody.validate_artifact(changed, root=ROOT)

    changed = deepcopy(artifact)
    changed["field_principles"].pop("rows")
    assert "field_principles_invalid" in custody.validate_artifact(changed, root=ROOT)

    changed = deepcopy(artifact)
    changed["reproducibility_checksum"] = "sha256:" + "0" * 64
    assert "reproducibility_checksum_mismatch" in custody.validate_artifact(changed, root=ROOT)

    assert custody.validate_artifact([]) == ["artifact_mapping_required"]

    changed = deepcopy(artifact)
    changed["schema"] = "wrong"
    assert "identity_invalid" in custody.validate_artifact(changed, root=ROOT)

    changed = deepcopy(artifact)
    changed["contract_mutation_rows"] = []
    assert "contract_mutations_invalid" in custody.validate_artifact(changed, root=ROOT)

    changed = deepcopy(artifact)
    changed["missing_producer_rows"] = []
    assert "missing_producer_rows_mismatch" in custody.validate_artifact(changed, root=ROOT)

    changed = deepcopy(artifact)
    changed["evidence_bundle"] = None
    assert "evidence_bundle_invalid" in custody.validate_artifact(changed, root=ROOT)

    changed = deepcopy(artifact)
    changed["evidence_bundle"]["path"] = Path(changed["evidence_bundle"]["path"]).name
    assert "evidence_bundle_invalid" in custody.validate_artifact(changed, root=ROOT)

    changed = deepcopy(artifact)
    changed["validation_receipts"] = changed["validation_receipts"][1:]
    assert "affected_validation_invalid" in custody.validate_artifact(changed, root=ROOT)

    changed = deepcopy(artifact)
    changed["validation_receipts"] = [
        row
        for row in changed["validation_receipts"]
        if row["name"] not in custody.TERMINAL_CHECK_NAMES
    ]
    assert "terminal_validation_invalid" in custody.validate_artifact(changed, root=ROOT)

    changed = deepcopy(artifact)
    changed["validation_receipts"][0].pop("log_path")
    assert "validation_log_hash_mismatch" in custody.validate_artifact(changed, root=ROOT)


def test_gate_summary_names_exact_unavailable_producers(tmp_path: Path) -> None:
    """REQ-REPORT-7409; SCENARIO-REPORT-7409-MISSING; SCENARIO-REPORT-7409-ARTIFACT."""

    artifact = custody.build_artifact_for_test(
        ROOT, tmp_path / "bundle", _passing_receipts(tmp_path / "logs")
    )

    blocked = artifact["gate_check_summary"]["blocked_external_inputs"]
    assert [row["upstream"] for row in blocked] == [
        "exp7397-delayed-adapter",
        "exp7399-online-trial",
    ]
    assert all(row["observed"] is None for row in blocked)
    assert artifact["honest_verdict"].startswith("complete_")


def test_validation_plan_is_scoped_and_preserves_coverage_file(tmp_path: Path) -> None:
    """REQ-REPORT-7409; SCENARIO-REPORT-7409-ARTIFACT."""

    commands = custody.build_validation_plan(ROOT, tmp_path / "private")

    assert custody.validate_validation_plan(ROOT, commands) == []
    assert [command.name for command in commands] == list(custody.AFFECTED_CHECK_NAMES)
    assert all("full_python_suite" not in command.argv for command in commands)
    report = next(row for row in commands if row.name == "changed_module_coverage_report")
    assert dict(report.command_environment)["COVERAGE_FILE"].endswith("/.coverage")


def test_parser_and_cold_main_reject_bad_inputs(tmp_path: Path) -> None:
    """REQ-REPORT-7409; SCENARIO-REPORT-7409-ARTIFACT."""

    with pytest.raises(SystemExit):
        custody.parse_args(["--date", "20260918"])
    with pytest.raises(SystemExit):
        custody.parse_args([])

    missing = tmp_path / "missing.json"
    assert custody.main(["--date", custody.RUN_DATE, "--validate", str(missing)]) == 1

    malformed = tmp_path / "malformed.json"
    malformed.write_text("[]\n", encoding="utf-8")
    assert custody.main(["--date", custody.RUN_DATE, "--validate", str(malformed)]) == 1


def test_file_size_inventory_and_deferred_gate_are_reviewable(tmp_path: Path) -> None:
    """REQ-REPORT-7409; SCENARIO-REPORT-7409-SIZE."""

    small = tmp_path / "small.json"
    small.write_text(json.dumps({"ok": True}), encoding="utf-8")

    inventory = custody.file_size_inventory([small, tmp_path / "absent"])
    proposal = custody.deferred_size_gate()

    assert inventory[0]["state"] == "present"
    assert inventory[0]["size_bytes"] == small.stat().st_size
    assert inventory[1]["state"] == "missing"
    assert proposal["status"] == "deferred_outer_loop"
    assert proposal["conductor_protected"] is False
    assert proposal["proposed_limit_bytes"] == custody.RESULT_SIZE_LIMIT_BYTES


def test_small_helpers_cover_progress_yaml_and_path_states(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """REQ-REPORT-7409; SCENARIO-REPORT-7409-CONTRACT; SCENARIO-REPORT-7409-MISSING."""

    assert "+00:00" in custody.utc_now()
    custody.progress(0.0, "fixture", "done", units=1)
    assert "phase=fixture" in capsys.readouterr().out
    assert custody.load_yaml(tmp_path / "missing.yaml") == {}

    custody._path_state.cache_clear()
    monkeypatch.setattr(custody, "_tracked_state", lambda _root, _path: (False, []))
    assert custody._path_state(tmp_path, Path("missing"))[0] == "missing"
    present = tmp_path / "present"
    present.write_text("x", encoding="utf-8")
    custody._path_state.cache_clear()
    assert custody._path_state(tmp_path, Path("present"))[0] == "present_untracked"

    custody._path_state.cache_clear()
    monkeypatch.setattr(custody, "_tracked_state", lambda _root, _path: (False, ["abc"]))
    assert custody._path_state(tmp_path, Path("stripped"))[0] == "stripped"
    custody._path_state.cache_clear()
    monkeypatch.setattr(custody, "_tracked_state", lambda _root, _path: (None, []))
    assert custody._path_state(tmp_path, Path("unknown"))[0] == "unknown"
    assert custody._directory_manifest(str(tmp_path), "absent") == ()
