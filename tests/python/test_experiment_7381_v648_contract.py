"""Tests for the V648 advisory source and exact-contract receipt.

Spec refs: REQ-REPORT-7381 and SCENARIO-REPORT-7381-*.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from carnot import experiment_7381_v648_contract as mod
from carnot.experiment_7358_v646_validation_contract import (
    build_command_plan,
    validate_command_plan,
)
from carnot.reporting.experiment_7303_validation_scope import REQUIRED_CHECK_NAMES


ROOT = Path(__file__).resolve().parents[2]


def _receipt(name: str, passed: bool = True) -> dict[str, Any]:
    """Create one real-shaped command receipt without starting a child process."""

    row: dict[str, Any] = {
        "name": name,
        "command": f"check {name}",
        "command_argv": ["check", name],
        "command_environment": {"COVERAGE_FILE": f"/tmp/.coverage-{name}"},
        "scope": "test_fixture",
        "exit_code": 0 if passed else 1,
        "duration_s": 0.01,
        "log_path": f"/tmp/{name}.log",
        "log_sha256": "sha256:" + "1" * 64,
        "passed": passed,
        "timed_out": False,
        "output_tail": "ok" if passed else "failed",
    }
    if name == "worktree_imports":
        row["resolved_imports"] = {
            "carnot.experiment_7381_v648_contract": str((ROOT / mod.MODULE_PATH).resolve())
        }
    return row


def _validation(**kwargs: Any) -> dict[str, Any]:
    """Return the fixed scoped plan while checking the task-owned file set."""

    assert kwargs["test_paths"] == [mod.TEST_PATH.as_posix()]
    assert kwargs["changed_modules"] == [mod.MODULE_PATH.as_posix()]
    assert kwargs["static_paths"] == [mod.WRAPPER_PATH.as_posix()]
    names = [*mod.ROADMAP_CHECK_NAMES, *REQUIRED_CHECK_NAMES]
    return {
        "validation_receipts": [_receipt(name) for name in names],
        "required_checks_passed": True,
        "plan_errors": [],
        "repository_health": {
            "status": "historical_observations_retained",
            "affects_required_checks": False,
            "observations": list(kwargs["historical_observations"]),
        },
    }


def _terminal(_root: Path, candidate: Path, _raw_dir: Path) -> list[dict[str, Any]]:
    """Cold-reload the measured candidate before terminal checks pass."""

    value = json.loads(candidate.read_text(encoding="utf-8"))
    assert value["status"] == "complete"
    assert mod.validate_artifact(value, root=ROOT, require_terminal=False) == []
    return [_receipt(name) for name in mod.TERMINAL_CHECK_NAMES]


def _fetch(url: str) -> dict[str, Any]:
    """Return deterministic bytes for the single bounded source request."""

    assert url == mod.SOURCE_DELTA_URL
    return {"ok": True, "status_code": 200, "body": "SHIP revision bytes", "error": None}


def test_scenario_report_7381_contract_selects_and_matches_authorities(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7381-CONTRACT checks exact selection and task parity."""

    active = yaml.safe_load((ROOT / mod.ACTIVE_ROADMAP_PATH).read_text(encoding="utf-8"))
    staged = deepcopy(active)
    (tmp_path / mod.ACTIVE_ROADMAP_PATH).write_text(yaml.safe_dump(active), encoding="utf-8")
    (tmp_path / mod.NEXT_ROADMAP_PATH).write_text(yaml.safe_dump(staged), encoding="utf-8")
    path, document, _content, candidates = mod.select_yaml_authority(tmp_path)
    assert path == mod.NEXT_ROADMAP_PATH
    assert document == staged
    assert [row["available"] for row in candidates] == [True, True]

    staged["milestone"] = "2026.09.647"
    (tmp_path / mod.NEXT_ROADMAP_PATH).write_text(yaml.safe_dump(staged), encoding="utf-8")
    path, document, _content, candidates = mod.select_yaml_authority(tmp_path)
    assert path == mod.ACTIVE_ROADMAP_PATH
    assert document == active
    assert [row["available"] for row in candidates] == [False, True]

    exact = mod.evaluate_contract((ROOT / mod.DESIGN_PATH).read_text(encoding="utf-8"), active)
    assert exact["passed"] is True
    assert exact["active_exact_contract_count"] == 1
    assert [row["unit_id"] for row in exact["contract_rows"]] == list(mod.EXPECTED_ID_ORDER)
    assert all(row["passed"] for row in exact["contract_rows"])
    assert exact["gate_declaration_result"]["passed"] is True

    duplicated = (ROOT / mod.DESIGN_PATH).read_text(encoding="utf-8") + "\n## Exact Task Contract\n"
    with pytest.raises(ValueError, match="exactly one active"):
        mod.parse_active_markdown_contract(duplicated)

    (tmp_path / mod.NEXT_ROADMAP_PATH).write_text("[]\n", encoding="utf-8")
    (tmp_path / mod.ACTIVE_ROADMAP_PATH).unlink()
    path, document, content, candidates = mod.select_yaml_authority(tmp_path)
    assert (path, document, content) == (None, None, None)
    assert "ValueError" in candidates[0]["observed_value"]
    assert "FileNotFoundError" in candidates[1]["observed_value"]


def test_scenario_report_7381_mutations_reject_every_named_defect(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7381-MUTATIONS rejects task, producer, and prior defects."""

    roadmap = yaml.safe_load((ROOT / mod.ACTIVE_ROADMAP_PATH).read_text(encoding="utf-8"))
    markdown = (ROOT / mod.DESIGN_PATH).read_text(encoding="utf-8")
    rows = mod.run_contract_mutations(markdown, roadmap, tmp_path)
    assert [row["mutation"] for row in rows] == list(mod.CONTRACT_MUTATION_NAMES)
    assert all(row["baseline_passed"] and row["rejected"] for row in rows)
    assert {row["rejecting_check"] for row in rows} == {
        "exact_contract",
        "roadmap_schema",
        "gate_declaration",
        "retirement_contract",
    }

    malformed = deepcopy(roadmap)
    malformed["tasks"][0]["prior_failures"][0].pop("addressed_by")
    result = mod.validate_retirement_declarations(malformed)
    assert result["passed"] is False
    assert result["rows"][0]["missing_fields"] == ["addressed_by"]
    assert mod.validate_retirement_declarations({"tasks": ["bad"]})["passed"] is False

    retired = deepcopy(roadmap)
    retired["tasks"][4]["gated_on"][0]["upstream"] = "exp2091-retired"
    assert mod.validate_gate_declarations(retired)["passed"] is False
    assert mod.validate_gate_declarations({"tasks": ["bad"]})["passed"] is False

    manifest = tmp_path / "manifest.yaml"
    manifest.write_text("retired:\n  - not-a-mapping\n", encoding="utf-8")
    assert mod._retired_ids(manifest) == set()

    freshness = mod.check_fresh_ids(ROOT, roadmap)
    assert freshness["passed"] is True
    assert [row["experiment_number"] for row in freshness["rows"]] == list(range(7381, 7395))
    assert mod.check_fresh_ids(tmp_path, roadmap)["passed"] is True


def test_scenario_report_7381_sources_keep_access_separate_from_methods() -> None:
    """SCENARIO-REPORT-7381-SOURCES preserves one bounded access result."""

    source_rows, access_rows = mod.collect_source_method_rows(_fetch)
    assert len(access_rows) == 1
    assert access_rows[0]["access_outcome"] == "http_success"
    assert mod.source_rows_complete(source_rows, access_rows) is True
    assert {row["method_family"] for row in source_rows} == {
        "ship",
        "calarena",
        "calvert",
        "online_calibration",
        "thermodynamic_learning",
    }
    assert all(row["frozen_roster_changed"] is False for row in source_rows)
    assert all(row["full_citation_census_claimed"] is False for row in source_rows)

    def unavailable(_url: str) -> dict[str, Any]:
        return {"ok": False, "status_code": 503, "body": "", "error": "unavailable"}

    rows, access = mod.collect_source_method_rows(unavailable)
    assert access[0]["access_outcome"] == "access_failed"
    assert rows[0]["access_status"] == "access_failed"
    assert mod.source_rows_complete(rows, access) is True

    def raises(_url: str) -> dict[str, Any]:
        raise TimeoutError("bounded timeout")

    _rows, access = mod.collect_source_method_rows(raises)
    assert access[0]["access_outcome"] == "access_failed"

    assert mod._access_outcome({"ok": False, "status_code": 429}) == "http_429"
    assert mod._access_outcome({"ok": False, "body": "browser challenge"}) == "browser_challenge"


def test_scenario_report_7381_history_authenticates_disqualified_receipts() -> None:
    """SCENARIO-REPORT-7381-HISTORY preserves exact V647 hashes and classes."""

    history = mod.collect_historical_boundary(ROOT)
    assert history["design_hash_matches"] is True
    assert history["planning_snapshot_ledger_terminal_milestone"] == "2026.09.646"
    assert history["current_ledger_terminal_milestone"] == "2026.09.647"
    assert [row["experiment_number"] for row in history["producer_rows"]] == [7372, 7376, 7378]
    assert all(row["verdict_class"] == "disqualified" for row in history["producer_rows"])
    assert all(row["flagged_adversarial"] is True for row in history["producer_rows"])
    assert all(row["accepted_as_current_readiness"] is False for row in history["producer_rows"])


def test_scenario_report_7381_validation_plan_is_exp7358_scoped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-REPORT-7381-VALIDATION derives the fixed affected commands."""

    commands = build_command_plan(ROOT, mod.V648_MANIFEST, tmp_path / "private")
    assert [command.name for command in commands] == list(REQUIRED_CHECK_NAMES)
    assert validate_command_plan(ROOT, mod.V648_MANIFEST, commands) == []
    assert "full_python_suite" not in mod.ALL_REQUIRED_CHECK_NAMES

    calls: list[str] = []

    def roadmap_runner(_root: Path, specs: Any, **_kwargs: Any) -> list[dict[str, Any]]:
        calls.extend(spec.name for spec in specs)
        return [_receipt(spec.name) for spec in specs]

    def affected_runner(_root: Path, planned: Any, **_kwargs: Any) -> list[dict[str, Any]]:
        calls.extend(row.spec.name for row in planned)
        return [_receipt(row.spec.name) for row in planned]

    result = mod.run_affected_validation(
        ROOT,
        mod.ACTIVE_ROADMAP_PATH,
        tmp_path / "raw",
        roadmap_runner=roadmap_runner,
        affected_runner=affected_runner,
    )
    assert result["required_checks_passed"] is True
    assert calls == [*mod.ROADMAP_CHECK_NAMES, *REQUIRED_CHECK_NAMES]

    monkeypatch.setattr(mod, "validate_command_plan", lambda *_args: ["plan drift"])
    rejected = mod.run_affected_validation(
        ROOT,
        mod.ACTIVE_ROADMAP_PATH,
        tmp_path / "rejected",
        historical_observations=[{"name": "old failure"}],
        roadmap_runner=roadmap_runner,
        affected_runner=affected_runner,
    )
    assert rejected["required_checks_passed"] is False
    assert rejected["validation_receipts"] == []
    assert rejected["repository_health"]["affects_required_checks"] is True

    terminal_names: list[str] = []

    def terminal_runner(_root: Path, specs: Any, **_kwargs: Any) -> list[dict[str, Any]]:
        terminal_names.extend(spec.name for spec in specs)
        return [_receipt(spec.name) for spec in specs]

    monkeypatch.setattr(mod, "run_commands", terminal_runner)
    candidate = tmp_path / "candidate.json"
    candidate.write_text("{}", encoding="utf-8")
    assert [row["name"] for row in mod.run_terminal_validation(ROOT, candidate, tmp_path)] == list(
        mod.TERMINAL_CHECK_NAMES
    )
    assert terminal_names == list(mod.TERMINAL_CHECK_NAMES)


def test_scenario_report_7381_artifact_replays_and_rejects_tampering(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7381-ARTIFACT recomputes the terminal advisory receipt."""

    artifact = mod.build_artifact(
        ROOT,
        mod.RUN_DATE,
        output_path=tmp_path / "result.json",
        raw_dir=tmp_path / "raw",
        checkpoint_path=tmp_path / "checkpoint.json",
        fetcher=_fetch,
        validation_runner=_validation,
        terminal_runner=_terminal,
    )
    assert artifact["status"] == "complete"
    assert artifact["verdict_class"] == "circular_positive"
    assert artifact["contract_complete_score"] == 1
    assert artifact["science_value_score"] == artifact["promotion_score"] == 0
    assert artifact["MODEL_SPECS"] == []
    assert artifact["model_invoked"] is False
    assert set(artifact["invocation_counts"].values()) == {0}
    assert artifact["inference_substrate"] == "aggregation_from_upstream_artifacts"
    assert artifact["inference_substrate_class"] == "aggregation"
    assert artifact["execution_venue"] == "host"
    assert artifact["study_mapping_ingested"] is True
    assert artifact["source_access_rows"][0]["access_outcome"] == "http_success"
    assert [row["name"] for row in artifact["validation_receipts"]] == list(
        mod.ALL_REQUIRED_CHECK_NAMES
    )
    assert mod.independent_reduce(artifact, require_terminal=True) == []
    assert mod.validate_artifact(artifact, root=ROOT, require_terminal=True) == []
    assert json.loads((tmp_path / "result.json").read_text(encoding="utf-8")) == artifact

    for error, field, value in (
        ("contract_rows_invalid", "contract_rows", []),
        ("mutation_rows_invalid", "mutation_rows", []),
        ("source_method_rows_invalid", "source_method_rows", []),
        ("history_invalid", "historical_boundary", {}),
        ("gate_controls_invalid", "gate_control_rows", []),
        ("fresh_ids_invalid", "fresh_id_rows", []),
        ("raw_authorities_invalid", "raw_authority_rows", []),
        ("rows_invalid", "rows", []),
        ("phase_spans_invalid", "phase_spans", [{"start_elapsed_s": 2.0, "end_elapsed_s": 1.0}]),
        ("validation_receipts_invalid", "validation_receipts", []),
    ):
        changed = deepcopy(artifact)
        changed[field] = value
        assert error in mod.independent_reduce(changed, require_terminal=True)

    changed = deepcopy(artifact)
    changed["reproducibility_checksum"] = "sha256:" + "0" * 64
    assert "reproducibility_checksum_invalid" in mod.validate_artifact(changed, root=ROOT)
    changed = deepcopy(artifact)
    changed["MODEL_SPECS"] = ["unexpected"]
    assert "model_contract_invalid" in mod.validate_artifact(changed, root=ROOT)
    changed = deepcopy(artifact)
    changed["execution_venue"] = "host_cpu"
    assert "substrate_invalid" in mod.validate_artifact(changed, root=ROOT)
    changed = deepcopy(artifact)
    changed["milestone"] = "2026.09.000"
    assert "identity_invalid" in mod.validate_artifact(changed, root=ROOT)
    changed = deepcopy(artifact)
    changed["status"] = "running"
    assert "lifecycle_invalid" in mod.validate_artifact(changed, root=ROOT)
    changed = deepcopy(artifact)
    changed["field_principles"] = {}
    assert "field_principles_invalid" in mod.validate_artifact(changed, root=ROOT)
    changed = deepcopy(artifact)
    changed["source_artifact_hashes"][next(iter(changed["source_artifact_hashes"]))] = (
        "sha256:" + "0" * 64
    )
    assert "source_hash_mismatch" in mod.validate_artifact(changed, root=ROOT)
    changed = deepcopy(artifact)
    changed["honest_verdict"] = "complete_changed"
    assert "honest_verdict_invalid" in mod.validate_artifact(changed, root=ROOT)
    changed = deepcopy(artifact)
    del changed["schema"]
    assert mod.validate_artifact(changed, root=ROOT) == ["missing_required_field:schema"]
    assert mod.validate_artifact([], root=ROOT) == ["artifact_mapping_required"]


def test_req_report_7381_blocks_missing_input_and_rejects_wrong_date(tmp_path: Path) -> None:
    """REQ-REPORT-7381 maps external absence to blocked, not partial."""

    artifact = mod.build_artifact(
        tmp_path,
        mod.RUN_DATE,
        output_path=tmp_path / "blocked.json",
        raw_dir=tmp_path / "raw",
        checkpoint_path=tmp_path / "checkpoint.json",
        fetcher=_fetch,
        validation_runner=_validation,
        terminal_runner=_terminal,
    )
    assert artifact["status"] == "blocked"
    assert artifact["verdict_class"] == "blocked"
    assert artifact["contract_complete_score"] == 0
    assert artifact["gate_check_summary"]["passed"] is False
    assert artifact["validation_receipts"] == []
    assert mod.validate_artifact(artifact, root=tmp_path) == []

    disqualified: dict[str, Any] = {
        "preconditions_checked": [],
        "acceptance_gate_results": [
            {
                "criterion": "contract_exact",
                "expected": True,
                "observed": False,
                "passed": False,
            }
        ],
    }
    mod._terminal_state(disqualified)
    assert disqualified["verdict_class"] == "disqualified"
    assert disqualified["contract_complete_score"] == 0

    assert mod.date_argument(mod.RUN_DATE) == mod.RUN_DATE
    with pytest.raises(Exception, match="run date must be"):
        mod.date_argument("20260917")
