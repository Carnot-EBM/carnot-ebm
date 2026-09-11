"""Tests for REQ-REPORT-7219 and SCENARIO-REPORT-7219-*.

The writer tests use a private repository fixture. They cannot replace the
project's active roadmap or terminal research receipt.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from carnot import experiment_7219_v636_source_contract as mod


ROOT = Path(__file__).resolve().parents[2]


def _fake_fetch(url: str) -> dict[str, Any]:
    """Return a small primary-page response with the configured exact title."""

    source = next(item for item in mod.SOURCES if item["url"] == url)
    title = source["title"]
    body = f"<html><head><title>{title}</title></head><body>{title}</body></html>"
    return {
        "ok": True,
        "status_code": 200,
        "url": url,
        "headers": {},
        "body": body,
        "error": None,
    }


def _private_root(tmp_path: Path, *, active_milestone: str = mod.MILESTONE) -> Path:
    """Copy exact authorities and make non-authority prerequisites nonempty."""

    for relative in mod.SOURCE_PATHS:
        source = ROOT / relative
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if source.is_file():
            target.write_bytes(source.read_bytes())
        else:
            target.write_text("fixture\n", encoding="utf-8")
    active = yaml.safe_load((ROOT / mod.ACTIVE_ROADMAP_PATH).read_text(encoding="utf-8"))
    active["milestone"] = active_milestone
    for task in active["tasks"]:
        task["milestone"] = active_milestone
    target = tmp_path / mod.ACTIVE_ROADMAP_PATH
    target.write_text(yaml.safe_dump(active, sort_keys=False), encoding="utf-8")
    return tmp_path


def test_req_report_7219_spec_precedes_implementation() -> None:
    """REQ-REPORT-7219 names each required field and focused scenario."""

    text = (ROOT / mod.SPEC_PATH).read_text(encoding="utf-8")
    section = text[text.index("### REQ-REPORT-7219") :]
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert f"`{field}`" in section
    for scenario in (
        "PARITY",
        "PREFLIGHT",
        "GATES",
        "QUARANTINE",
        "WRAPPERS",
        "DURATION",
        "SOURCES",
        "ARTIFACT",
    ):
        assert f"SCENARIO-REPORT-7219-{scenario}" in section


def test_scenario_report_7219_parity_matches_actual_authorities() -> None:
    """SCENARIO-REPORT-7219-PARITY independently recovers all exact rows."""

    markdown = (ROOT / mod.DESIGN_PATH).read_text(encoding="utf-8")
    roadmap = yaml.safe_load((ROOT / mod.ACTIVE_ROADMAP_PATH).read_text(encoding="utf-8"))
    result = mod.evaluate_contract(markdown, roadmap)
    assert result["passed"] is True
    assert result["expected_id_order"] == list(mod.EXPECTED_ID_ORDER)
    assert len(result["contract_rows"]) == 14
    assert all(row["passed"] for row in result["contract_rows"])
    assert len(result["gate_producer_rows"]) == 7
    assert all(row["passed"] for row in result["gate_producer_rows"])
    assert result["receipt_dependency_rows"][0]["observed_consumers"] == []

    changed = deepcopy(roadmap)
    changed["tasks"][1]["title"] = "readable mismatch"
    assert mod.evaluate_contract(markdown, changed)["passed"] is False


def test_scenario_report_7219_preflight_selects_only_v636(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7219-PREFLIGHT prefers active, then matching staged YAML."""

    root = _private_root(tmp_path)
    authority, roadmap, content, rows = mod.select_yaml_authority(root)
    assert authority == mod.ACTIVE_ROADMAP_PATH
    assert roadmap and content and all("observed_value" in row for row in rows)

    stale = yaml.safe_load((root / mod.ACTIVE_ROADMAP_PATH).read_text(encoding="utf-8"))
    stale["milestone"] = "2026.09.635"
    for task in stale["tasks"]:
        task["milestone"] = "2026.09.635"
    (root / mod.ACTIVE_ROADMAP_PATH).write_text(
        yaml.safe_dump(stale, sort_keys=False), encoding="utf-8"
    )
    (root / mod.NEXT_ROADMAP_PATH).write_bytes((ROOT / mod.ACTIVE_ROADMAP_PATH).read_bytes())
    authority, roadmap, content, _rows = mod.select_yaml_authority(root)
    assert authority == mod.NEXT_ROADMAP_PATH and roadmap and content

    (root / mod.NEXT_ROADMAP_PATH).unlink()
    authority, roadmap, content, _rows = mod.select_yaml_authority(root)
    assert (authority, roadmap, content) == (None, None, None)


def test_scenarios_report_7219_gate_edges_quarantine_and_wrappers(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7219-GATES, QUARANTINE, and WRAPPERS fail closed."""

    roadmap = yaml.safe_load((ROOT / mod.ACTIVE_ROADMAP_PATH).read_text(encoding="utf-8"))
    rows = mod.run_gate_validation_fixtures(roadmap, tmp_path)
    assert len(rows) == 35
    assert {row["case"] for row in rows} == {
        "passing",
        "failed",
        "missing_field",
        "missing_file",
        "quarantined_passing",
    }
    for edge in range(7):
        edge_rows = [row for row in rows if row["edge_index"] == edge]
        assert [row["conductor_passed"] for row in edge_rows] == [
            True,
            False,
            False,
            False,
            True,
        ]
        assert edge_rows[-1]["experiment_precondition_passed"] is False
        assert edge_rows[-1]["field_gate_authenticates_artifact"] is False
    assert mod.activation_complete(
        [{"case": "roadmap_schema_parse", "passed": True, "validation_input": True}, *rows]
    )

    wrapper = {"principle": "explain", "value": 1}
    domain = {"value": 1, "domain": "ordinary mapping"}
    assert mod.unwrap_principle_value(wrapper) == 1
    assert mod.unwrap_principle_value(domain) is domain
    assert mod.upstream_precondition({"ready": wrapper}, "ready", 1)["passed"] is True
    assert (
        mod.upstream_precondition({"ready": wrapper, "flagged_adversarial": True}, "ready", 1)[
            "passed"
        ]
        is False
    )


def test_scenario_report_7219_duration_uses_real_classifier() -> None:
    """SCENARIO-REPORT-7219-DURATION retains the Exp7208 counterexample."""

    rows = mod.duration_classifier_rows(ROOT)
    by_case = {row["case"]: row for row in rows}
    aggregation = by_case["v636_aggregation_pair"]
    assert aggregation["class_floor_s"] == aggregation["name_floor_s"]
    assert aggregation["class_flags"] == []
    negative = by_case["exp7208_negative_counterexample"]
    assert negative["flagged_adversarial"] is True
    assert negative["name_floor_s"] > negative["duration_s"]
    assert negative["name_floor_reason"] == "live_model"


def test_scenario_report_7219_sources_keep_titles_and_access_limits() -> None:
    """SCENARIO-REPORT-7219-SOURCES records five bounded execution-time checks."""

    rows = mod.collect_source_method_rows(_fake_fetch)
    assert len(rows) == 5
    assert all(row["retrieval_date"] == "2026-09-11" for row in rows)
    assert all(row["title_verified"] is True for row in rows)
    assert all(row["access_outcome"] == "http_200" for row in rows)
    assert all(row["post_planning_delta"] is False for row in rows)

    def unavailable(_url: str) -> dict[str, Any]:
        return {"ok": False, "status_code": None, "body": "", "error": "offline"}

    limited = mod.collect_source_method_rows(unavailable)
    assert all(row["access_outcome"] == "unavailable_cached_primary_evidence" for row in limited)
    assert all(row["title_verified"] is None for row in limited)

    def error_page(_url: str) -> dict[str, Any]:
        return {
            "ok": False,
            "status_code": 403,
            "body": "<title>Access denied</title>",
            "error": "forbidden",
        }

    assert all(row["title_verified"] is None for row in mod.collect_source_method_rows(error_page))

    def raises(_url: str) -> dict[str, Any]:
        raise RuntimeError("fetcher failed")

    raised = mod.collect_source_method_rows(raises)
    assert all(row["access_error"] == "fetcher failed" for row in raised)


def test_req_report_7219_uses_actual_v635_paths_and_archive_boundary() -> None:
    """REQ-REPORT-7219 uses Exp7218's real producer paths and no V634 invention."""

    assert ROOT.joinpath(mod.V635_RECEIPT_PATH).is_file()
    assert ROOT.joinpath(mod.V635_CAPSTONE_PATH).is_file()
    assert ROOT.joinpath(mod.V635_CAPSTONE_MODULE_PATH).is_file()
    assert ROOT.joinpath(mod.V635_CAPSTONE_WRAPPER_PATH).is_file()
    assert all("experiment_7204_v634_capstone.py" not in str(path) for path in mod.SOURCE_PATHS)
    archive = mod.archive_lag_row(ROOT)
    assert archive["planning_refresh_archive_latest_milestone"] == "2026.09.634"
    assert archive["archive_latest_milestone"] == "2026.09.635"
    assert archive["active_evidence_milestone"] == "2026.09.635"
    assert archive["archive_lag_observed_at_execution"] is False
    assert archive["research_complete_rewritten"] is False


def test_scenario_report_7219_artifact_builds_and_rejects_forgery(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7219-ARTIFACT validates a private complete receipt."""

    root = _private_root(tmp_path)
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
        run_commands=False,
    )
    assert artifact["source_contract_complete_score"] == 1
    assert artifact["verdict_class"] == "positive"
    assert artifact["MODEL_SPECS"] == [] and artifact["model_invoked"] is False
    assert len(artifact["activation_validation_rows"]) == 36
    assert output.is_file() and checkpoint.is_file()
    assert (raw_dir / mod.RAW_MARKDOWN_NAME).read_bytes() == (root / mod.DESIGN_PATH).read_bytes()
    assert (raw_dir / mod.RAW_YAML_NAME).read_bytes() == (
        root / mod.ACTIVE_ROADMAP_PATH
    ).read_bytes()
    assert mod.validate_artifact(artifact) == []

    for mutation in (
        lambda value: value.__setitem__("status", "running"),
        lambda value: value.pop("run_date"),
        lambda value: value["field_principles"].__setitem__("status", "changed"),
        lambda value: value.__setitem__("run_date", "20260910"),
        lambda value: value.__setitem__("started_at_utc", "invalid"),
        lambda value: value.__setitem__("completed_at_utc", "2026-09-11T00:00:00"),
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
        lambda value: value.__setitem__("reproducibility_checksum", "sha256:forged"),
    ):
        changed = deepcopy(artifact)
        mutation(changed)
        assert mod.validate_artifact(changed)
    assert mod.validate_artifact([]) == ["artifact_mapping_required"]

    changed = deepcopy(artifact)
    changed["validation_required"] = True
    assert mod._score_from_artifact(changed) == 0
    changed["validation_command_rows"] = [{"name": "roadmap_schema", "passed": False}]
    assert mod._score_from_artifact(changed) == 0
    changed["contract_rows"] = None
    assert mod._score_from_artifact(changed) == 0

    changed = deepcopy(artifact)
    changed["markdown_milestone"] = "stale"
    assert mod._failure_summary(changed)["failed_check"] == "markdown_milestone"
    changed = deepcopy(artifact)
    changed["activation_validation_rows"] = []
    assert mod._failure_summary(changed)["failed_check"] == "activation_validation"
    changed = deepcopy(artifact)
    changed["validation_command_rows"] = [{"name": "fixture", "passed": False, "exit_code": 2}]
    assert mod._failure_summary(changed)["failed_check"] == "validation:fixture"
    assert mod._failure_summary(artifact)["failed_check"] == "stored_source_contract_evidence"


def test_req_report_7219_blocked_receipt_is_diagnosed(tmp_path: Path) -> None:
    """REQ-REPORT-7219 writes blocked_no_run for a missing roadmap authority."""

    root = _private_root(tmp_path, active_milestone="2026.09.635")
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
    assert artifact["inference_substrate_class"] == "blocked_no_run"
    assert artifact["gate_check_summary"]["field"] == "milestone"
    assert mod.validate_artifact(artifact) == []


def test_req_report_7219_validation_commands_are_scoped() -> None:
    """REQ-REPORT-7219 keeps validators on the changed module and test."""

    names = [
        name
        for name, _command in mod.validation_commands(
            ROOT, ROOT / mod.DEFAULT_OUTPUT_PATH, mod.ACTIVE_ROADMAP_PATH
        )
    ]
    assert names == list(mod.VALIDATION_COMMAND_NAMES)
    assert mod.date_argument("20260911") == "20260911"
    with pytest.raises(argparse.ArgumentTypeError):
        mod.date_argument("20260910")

    stored = json.loads((ROOT / mod.V635_RECEIPT_PATH).read_text(encoding="utf-8"))
    intake = mod.authenticate_upstream(stored, "source_contract_complete_score", 1)
    assert intake["accepted_for_evidence"] is False
    assert intake["known_failed_value"] is True


def test_req_report_7219_defensive_preconditions_and_activation(tmp_path: Path) -> None:
    """REQ-REPORT-7219 retains diagnosed malformed and missing local inputs."""

    assert mod._failed_precondition({"preconditions_checked": None}) is None
    assert mod.activation_complete(None) is False
    rows = [{"case": "roadmap_schema_parse", "passed": False, "validation_input": True}]
    rows.extend({"edge_index": index // 5, "validation_input": True} for index in range(35))
    assert mod.activation_complete(rows) is False

    roadmap = yaml.safe_load((ROOT / mod.ACTIVE_ROADMAP_PATH).read_text(encoding="utf-8"))
    valid = mod.run_gate_validation_fixtures(roadmap, tmp_path / "gates")
    activation = [
        {"case": "roadmap_schema_parse", "passed": True, "validation_input": True},
        *valid,
    ]
    changed = deepcopy(activation)
    changed[1]["conductor_passed"] = False
    assert mod.activation_complete(changed) is False
    changed = deepcopy(activation)
    changed[5]["experiment_precondition_passed"] = True
    assert mod.activation_complete(changed) is False
    assert mod.activation_rows({"tasks": []})[0]["passed"] is False

    root = _private_root(tmp_path / "inputs")
    (root / mod.SPEC_PATH).unlink()
    (root / mod.REFERENCE_PATH).unlink()
    preconditions, _authority, _roadmap, _content = mod._preconditions(
        root,
        root / mod.DEFAULT_OUTPUT_PATH,
        root / mod.RAW_DIR,
        root / mod.CHECKPOINT_PATH,
    )
    by_check = {row["check"]: row for row in preconditions}
    assert by_check["driving_requirement"]["available"] is False
    assert by_check["planning_source_table"]["available"] is False
    intake = mod.upstream_intake_rows(tmp_path / "missing-history")
    assert all(row["error"] for row in intake)


def test_req_report_7219_streams_validation_receipts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-REPORT-7219 checkpoints each truthful subprocess result."""

    artifact = mod._base_artifact(
        mod.RUN_DATE, tmp_path / "checkpoint.json", "2026-09-11T00:00:00+00:00"
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


def test_req_report_7219_build_runs_validation_hook(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-REPORT-7219 calls scoped validation when command execution is enabled."""

    root = _private_root(tmp_path)

    def validation_hook(
        _root: Path, _checkpoint: Path, artifact: dict[str, Any], _started: float
    ) -> list[dict[str, Any]]:
        artifact["validation_required"] = False
        return []

    monkeypatch.setattr(mod, "_run_validation_commands", validation_hook)
    artifact = mod.build_artifact(
        root,
        mod.RUN_DATE,
        output_path=root / mod.DEFAULT_OUTPUT_PATH,
        raw_dir=root / mod.RAW_DIR,
        checkpoint_path=root / mod.CHECKPOINT_PATH,
        fetcher=_fake_fetch,
        run_commands=True,
    )
    assert artifact["source_contract_complete_score"] == 1
