"""Tests for REQ-REPORT-7233 and SCENARIO-REPORT-7233-*.

All writers use a private repository copy. The tests never replace an active
roadmap, raw research record, checkpoint, or terminal artifact.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from carnot import experiment_7233_v637_contract as mod


ROOT = Path(__file__).resolve().parents[2]


def _fake_fetch(url: str) -> dict[str, Any]:
    """Return the configured title and version for one primary record."""

    source = next(row for row in mod.SOURCES if row["url"] == url)
    body = (
        f"<html><head><title>{source['title']}</title></head>"
        f"<body>[{source['planning_version']}] {source['planning_version_date']}</body></html>"
    )
    return {
        "ok": True,
        "status_code": 200,
        "url": url,
        "headers": {},
        "body": body,
        "error": None,
    }


def _private_root(tmp_path: Path, *, active_milestone: str = mod.MILESTONE) -> Path:
    """Copy authorities and required inputs into an isolated repository tree."""

    for relative in mod.SOURCE_PATHS:
        source = ROOT / relative
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.read_bytes())
    active = yaml.safe_load((ROOT / mod.ACTIVE_ROADMAP_PATH).read_text(encoding="utf-8"))
    active["milestone"] = active_milestone
    for task in active["tasks"]:
        task["milestone"] = active_milestone
    target = tmp_path / mod.ACTIVE_ROADMAP_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(yaml.safe_dump(active, sort_keys=False), encoding="utf-8")
    return tmp_path


def _passing_validation_hook(
    _root: Path, _checkpoint: Path, artifact: dict[str, Any], _started: float
) -> list[dict[str, Any]]:
    """Stand in for subprocesses while preserving their complete result contract."""

    rows = [
        {
            "name": name,
            "command": f"fixture {name}",
            "exit_code": 0,
            "stdout": "ok",
            "stderr": "",
            "timed_out": False,
            "duration_s": 0.01,
            "passed": True,
        }
        for name in mod.VALIDATION_COMMAND_NAMES
    ]
    artifact["validation_required"] = True
    artifact["validation_command_rows"] = rows
    return rows


def _fast_classifier_sidecar(root: Path, raw_dir: Path) -> dict[str, str]:
    """Use compact valid evidence when a test targets the artifact lifecycle."""

    reports = [
        {
            "case": f"valid_{index}",
            "critical_flag_count": 0,
        }
        for index in range(4)
    ]
    reports.extend(
        {
            "case": f"contradiction_{index}",
            "critical_flag_count": 1,
        }
        for index in range(3)
    )
    historical = [
        {
            "quarantined": index > 0,
            "accepted_for_evidence": index == 0,
        }
        for index in range(5)
    ]
    sidecar = {
        "isolated_fixture_reports": reports,
        "historical_source_rows": historical,
        "unresolved_classifier_problem": {"status": "unresolved"},
    }
    path = raw_dir / mod.CLASSIFIER_SIDECAR_NAME
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(sidecar), encoding="utf-8")
    return {"path": str(path.relative_to(root)), "sha256": mod.sha256(path)}


def test_req_report_7233_spec_precedes_implementation() -> None:
    """REQ-REPORT-7233 names each required field and focused scenario."""

    text = (ROOT / mod.SPEC_PATH).read_text(encoding="utf-8")
    section = text[text.index("### REQ-REPORT-7233") :]
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert f"`{field}`" in section
    for scenario in (
        "PARITY",
        "PREFLIGHT",
        "GATES",
        "QUARANTINE",
        "WRAPPERS",
        "CLASSIFIER",
        "SOURCES",
        "ARTIFACT",
        "VALIDATION",
    ):
        assert f"SCENARIO-REPORT-7233-{scenario}" in section


def test_scenario_report_7233_parity_matches_actual_authorities() -> None:
    """SCENARIO-REPORT-7233-PARITY recovers thirteen exact task rows."""

    markdown = (ROOT / mod.DESIGN_PATH).read_text(encoding="utf-8")
    roadmap = yaml.safe_load((ROOT / mod.ACTIVE_ROADMAP_PATH).read_text(encoding="utf-8"))
    result = mod.evaluate_contract(markdown, roadmap)
    assert result["passed"] is True
    assert result["expected_id_order"] == list(mod.EXPECTED_ID_ORDER)
    assert len(result["contract_rows"]) == 13
    assert all(row["passed"] for row in result["contract_rows"])
    assert len(result["gate_producer_rows"]) == 5
    assert result["receipt_dependency_rows"][0]["observed_consumers"] == []

    changed = deepcopy(roadmap)
    changed["tasks"][4]["title"] = "changed title"
    assert mod.evaluate_contract(markdown, changed)["passed"] is False


def test_scenario_report_7233_preflight_selects_only_v637(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7233-PREFLIGHT prefers active, then matching staged YAML."""

    root = _private_root(tmp_path)
    authority, roadmap, content, rows = mod.select_yaml_authority(root)
    assert authority == mod.ACTIVE_ROADMAP_PATH
    assert roadmap and content and all("observed_value" in row for row in rows)

    stale = yaml.safe_load((root / mod.ACTIVE_ROADMAP_PATH).read_text(encoding="utf-8"))
    stale["milestone"] = "2026.09.636"
    for task in stale["tasks"]:
        task["milestone"] = "2026.09.636"
    (root / mod.ACTIVE_ROADMAP_PATH).write_text(
        yaml.safe_dump(stale, sort_keys=False), encoding="utf-8"
    )
    (root / mod.NEXT_ROADMAP_PATH).write_bytes((ROOT / mod.ACTIVE_ROADMAP_PATH).read_bytes())
    authority, roadmap, content, _rows = mod.select_yaml_authority(root)
    assert authority == mod.NEXT_ROADMAP_PATH and roadmap and content

    (root / mod.NEXT_ROADMAP_PATH).unlink()
    authority, roadmap, content, _rows = mod.select_yaml_authority(root)
    assert (authority, roadmap, content) == (None, None, None)


def test_scenarios_report_7233_gates_quarantine_and_wrappers(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7233-GATES, QUARANTINE, and WRAPPERS fail closed."""

    roadmap = yaml.safe_load((ROOT / mod.ACTIVE_ROADMAP_PATH).read_text(encoding="utf-8"))
    rows = mod.run_gate_replays(roadmap, tmp_path)
    assert len(rows) == 20
    assert {row["case"] for row in rows} == {
        "passing",
        "failed",
        "absent_field",
        "absent_file",
    }
    for edge in range(5):
        edge_rows = [row for row in rows if row["edge_index"] == edge]
        assert [row["passed"] for row in edge_rows] == [True, False, False, False]
    assert mod.gate_replays_complete(rows)
    changed = deepcopy(rows)
    changed[0]["passed"] = False
    assert mod.gate_replays_complete(changed) is False
    reordered = deepcopy(rows)
    reordered[0], reordered[1] = reordered[1], reordered[0]
    assert mod.gate_replays_complete(reordered) is False
    assert mod.gate_replays_complete(None) is False

    malformed = deepcopy(roadmap)
    malformed["tasks"][4]["gated_on"][0]["upstream"] = "not-an-experiment"
    with pytest.raises(ValueError, match="invalid upstream task id"):
        mod.run_gate_replays(malformed, tmp_path / "malformed")

    wrapper = {"principle": "explain", "value": 1}
    ordinary = {"value": 1, "domain": "ordinary mapping"}
    assert mod.unwrap_principle_value(wrapper) == 1
    assert mod.unwrap_principle_value(ordinary) is ordinary
    assert mod.authenticate_value({"ready": wrapper}, "ready", 1)["accepted"] is True
    rejected = mod.authenticate_value({"ready": wrapper, "flagged_adversarial": True}, "ready", 1)
    assert rejected["accepted"] is False and rejected["quarantined"] is True


def test_scenario_report_7233_classifier_sidecar_isolates_evidence(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7233-CLASSIFIER keeps fixtures outside current provenance."""

    root = _private_root(tmp_path)
    metadata = mod.build_classifier_sidecar(root, root / mod.RAW_DIR)
    path = root / metadata["path"]
    sidecar = json.loads(path.read_text(encoding="utf-8"))
    assert metadata["sha256"] == mod.sha256(path)
    assert len(sidecar["isolated_fixture_reports"]) == 7
    valid = sidecar["isolated_fixture_reports"][:4]
    assert [row["case"] for row in valid] == [
        "aggregation",
        "bounded_generation",
        "full_generation",
        "load_only",
    ]
    assert all(row["critical_flag_count"] == 0 for row in valid)
    assert all(row["current_task_fixture"] is False for row in valid)
    contradictory = sidecar["isolated_fixture_reports"][4:]
    assert all(row["critical_flag_count"] > 0 for row in contradictory)
    assert any("INFERENCE_PROVENANCE_CONTRADICTION" in row["flag_kinds"] for row in contradictory)
    assert len(sidecar["historical_source_rows"]) == 5
    quarantines = sidecar["historical_source_rows"][1:]
    assert all(row["quarantined"] is True for row in quarantines)
    assert all(row["accepted_for_evidence"] is False for row in quarantines)
    assert sidecar["unresolved_classifier_problem"]["status"] == "unresolved"
    assert mod._critical_flags({"flags": None}) == []


def test_req_report_7233_precondition_failures_are_explicit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-REPORT-7233 turns missing or malformed prerequisites into evidence."""

    root = _private_root(tmp_path)
    first_source = mod.HISTORICAL_SOURCES[0]
    monkeypatch.setattr(mod, "HISTORICAL_SOURCES", (first_source,))
    (root / mod.SPEC_PATH).unlink()
    (root / mod.REFERENCE_PATH).unlink()
    rows, _authority, _roadmap, _yaml_bytes = mod._preconditions(
        root,
        root / mod.DEFAULT_OUTPUT_PATH,
        root / mod.RAW_DIR,
        root / mod.CHECKPOINT_PATH,
    )
    assert next(row for row in rows if row["check"] == "driving_requirement")["available"] is False
    assert (
        next(row for row in rows if row["check"] == "planning_source_table")["available"] is False
    )

    historical_path = root / first_source[0]
    historical_path.write_text("[]", encoding="utf-8")
    rows, *_rest = mod._preconditions(
        root,
        root / mod.DEFAULT_OUTPUT_PATH,
        root / mod.RAW_DIR,
        root / mod.CHECKPOINT_PATH,
    )
    assert rows[-1]["available"] is False
    monkeypatch.setattr(mod, "verify_artifact", lambda *_args, **_kwargs: {"flags": []})
    with pytest.raises(ValueError, match="historical source must be a mapping"):
        mod.build_classifier_sidecar(root, root / mod.RAW_DIR)

    historical_path.unlink()
    rows, *_rest = mod._preconditions(
        root,
        root / mod.DEFAULT_OUTPUT_PATH,
        root / mod.RAW_DIR,
        root / mod.CHECKPOINT_PATH,
    )
    assert rows[-1]["available"] is False
    assert mod._failed_precondition({"preconditions_checked": None}) is None


def test_scenario_report_7233_sources_keep_versions_and_limits() -> None:
    """SCENARIO-REPORT-7233-SOURCES records three bounded no-delta checks."""

    rows = mod.collect_source_method_rows(_fake_fetch)
    assert len(rows) == 3
    assert all(row["retrieval_date"] == "2026-09-12" for row in rows)
    assert all(row["title_verified"] is True for row in rows)
    assert all(row["access_outcome"] == "http_200" for row in rows)
    assert all(row["execution_time_version_changed"] is False for row in rows)
    assert all(row["recheck_outcome"] == "no_delta" for row in rows)

    def changed(url: str) -> dict[str, Any]:
        receipt = _fake_fetch(url)
        receipt["body"] = receipt["body"].replace("[v3]", "[v4]").replace("[v1]", "[v2]")
        return receipt

    changed_rows = mod.collect_source_method_rows(changed)
    assert all(row["execution_time_version_changed"] is True for row in changed_rows)
    assert all(row["recheck_outcome"] == "version_changed_review_required" for row in changed_rows)

    def unavailable(_url: str) -> dict[str, Any]:
        return {"ok": False, "status_code": None, "body": "", "error": "offline"}

    limited = mod.collect_source_method_rows(unavailable)
    assert all(row["access_outcome"] == "unavailable_cached_primary_evidence" for row in limited)
    assert all(row["recheck_outcome"] == "access_failed" for row in limited)
    assert all(row["title_verified"] is None for row in limited)

    def raises(_url: str) -> dict[str, Any]:
        raise RuntimeError("fetcher failed")

    assert all(
        row["access_error"] == "fetcher failed" for row in mod.collect_source_method_rows(raises)
    )


def test_req_report_7233_archive_and_historical_inputs_are_observed() -> None:
    """REQ-REPORT-7233 records the archive transition without rewriting it."""

    row = mod.archive_lag_row(ROOT)
    assert row["planning_refresh_archive_latest_milestone"] == "2026.09.635"
    assert row["archive_latest_milestone_at_execution"] == "2026.09.636"
    assert row["v636_terminal_artifacts_observed"] is True
    assert row["research_complete_rewritten"] is False
    assert all((ROOT / path).is_file() for path, _field, _expected in mod.HISTORICAL_SOURCES)


def test_scenario_report_7233_artifact_builds_and_rejects_forgery(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-REPORT-7233-ARTIFACT recomputes a private complete receipt."""

    root = _private_root(tmp_path)
    monkeypatch.setattr(mod, "_run_validation_commands", _passing_validation_hook)
    monkeypatch.setattr(mod, "build_classifier_sidecar", _fast_classifier_sidecar)
    output = root / mod.DEFAULT_OUTPUT_PATH
    raw_dir = root / mod.RAW_DIR
    checkpoint = root / mod.CHECKPOINT_PATH
    artifact = mod.build_artifact(
        root,
        mod.RUN_DATE,
        output_path=output,
        raw_dir=raw_dir,
        checkpoint_path=checkpoint,
        fetcher=_fake_fetch,
        run_commands=True,
    )
    assert artifact["source_contract_complete_score"] == 1
    assert artifact["verdict_class"] == "positive"
    assert artifact["MODEL_SPECS"] == [] and artifact["model_invoked"] is False
    assert artifact["model_invocation_count"] == 0
    assert len(artifact["gate_replay_rows"]) == 20
    assert output.is_file() and checkpoint.is_file()
    assert (raw_dir / mod.RAW_MARKDOWN_NAME).read_bytes() == (root / mod.DESIGN_PATH).read_bytes()
    assert (raw_dir / mod.RAW_YAML_NAME).read_bytes() == (
        root / mod.ACTIVE_ROADMAP_PATH
    ).read_bytes()
    assert mod.validate_artifact(artifact, root=root) == []

    mutations = (
        lambda value: value.__setitem__("status", "running"),
        lambda value: value.__setitem__("schema", "wrong"),
        lambda value: value.pop("run_date"),
        lambda value: value["field_principles"].__setitem__("status", "changed"),
        lambda value: value.__setitem__("run_date", "20260911"),
        lambda value: value.__setitem__("started_at_utc", "invalid"),
        lambda value: value.__setitem__("completed_at_utc", "2026-09-12T00:00:00"),
        lambda value: value.__setitem__("execution_venue", "elsewhere"),
        lambda value: value.__setitem__("execution_host", "other-host"),
        lambda value: value.__setitem__("duration_s", 0),
        lambda value: value.__setitem__("random_seed", 0),
        lambda value: value.__setitem__("verifier_is_oracle", True),
        lambda value: value.__setitem__("rows", []),
        lambda value: value.__setitem__("sample_size_budget", {}),
        lambda value: value.__setitem__("source_artifact_hashes", {}),
        lambda value: value.__setitem__("raw_source_rows", []),
        lambda value: value.__setitem__("validation_command_rows", [{"name": "wrong"}]),
        lambda value: value.__setitem__("source_contract_complete_score", 0),
        lambda value: value["contract_rows"][0].__setitem__("passed", False),
        lambda value: value.__setitem__("model_invoked", True),
        lambda value: value.__setitem__("model_invocation_count", 1),
        lambda value: value.__setitem__("classifier_receipt_sha256", "sha256:forged"),
        lambda value: value.__setitem__("reproducibility_checksum", "sha256:forged"),
    )
    for mutation in mutations:
        changed = deepcopy(artifact)
        mutation(changed)
        assert mod.validate_artifact(changed, root=root)
    classifier_path = root / artifact["classifier_receipt_path"]
    classifier_bytes = classifier_path.read_bytes()
    classifier_path.unlink()
    assert "classifier_receipt_invalid" in mod.validate_artifact(artifact, root=root)
    classifier_path.write_bytes(classifier_bytes)
    assert mod.validate_artifact([], root=root) == ["artifact_mapping_required"]

    no_commands = mod.build_artifact(
        root,
        mod.RUN_DATE,
        output_path=root / "results/no-commands.json",
        raw_dir=root / "results/raw/no-commands",
        checkpoint_path=root / "results/checkpoints/no-commands.json",
        fetcher=_fake_fetch,
        run_commands=False,
    )
    assert no_commands["source_contract_complete_score"] == 0
    assert no_commands["verdict_class"] == "disqualified"
    assert mod.validate_artifact(no_commands, root=root) == []


def test_req_report_7233_blocked_receipt_is_diagnosed(tmp_path: Path) -> None:
    """REQ-REPORT-7233 writes blocked_no_run when no roadmap authority matches."""

    root = _private_root(tmp_path, active_milestone="2026.09.636")
    artifact = mod.build_artifact(
        root,
        mod.RUN_DATE,
        output_path=root / mod.DEFAULT_OUTPUT_PATH,
        raw_dir=root / mod.RAW_DIR,
        checkpoint_path=root / mod.CHECKPOINT_PATH,
        fetcher=_fake_fetch,
        run_commands=False,
    )
    assert artifact["verdict_class"] == "blocked"
    assert artifact["inference_substrate"] == "blocked_no_run"
    assert artifact["inference_substrate_class"] == "blocked_no_run"
    assert artifact["gate_check_summary"]["artifact_field"] == "milestone"
    assert mod.validate_artifact(artifact, root=root) == []


def test_scenario_report_7233_validation_commands_are_scoped() -> None:
    """SCENARIO-REPORT-7233-VALIDATION uses explicit changed paths."""

    names = [
        name
        for name, _command in mod.validation_commands(
            ROOT, ROOT / mod.CHECKPOINT_PATH, mod.ACTIVE_ROADMAP_PATH
        )
    ]
    assert names == list(mod.VALIDATION_COMMAND_NAMES)
    assert mod.date_argument("20260912") == "20260912"
    with pytest.raises(argparse.ArgumentTypeError):
        mod.date_argument("20260911")
    with pytest.raises(ValueError):
        mod.date_argument("not-a-date")


def test_scenario_report_7233_validation_receipts_stream(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-REPORT-7233-VALIDATION checkpoints each subprocess result."""

    artifact = mod.base_artifact(
        mod.RUN_DATE, tmp_path / "checkpoint.json", "2026-09-12T00:00:00+00:00"
    )
    artifact["yaml_authority_path"] = str(mod.ACTIVE_ROADMAP_PATH)
    monkeypatch.setattr(
        mod,
        "validation_commands",
        lambda *_args: (("artifact", ["python", "-c", "pass"]),),
    )
    monkeypatch.setattr(
        mod,
        "_run_streaming_command",
        lambda *_args, **_kwargs: {
            "exit_code": 0,
            "stdout": "ok",
            "stderr": "",
            "timed_out": False,
            "duration_s": 0.01,
        },
    )
    rows = mod._run_validation_commands(tmp_path, tmp_path / "checkpoint.json", artifact, 0.0)
    assert rows[0]["passed"] is True
    assert artifact["validation_required"] is True
