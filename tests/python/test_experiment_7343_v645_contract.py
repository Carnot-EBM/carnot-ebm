"""Tests for the V645 advisory contract receipt.

Spec refs: REQ-REPORT-7343 and SCENARIO-REPORT-7343-*.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from carnot import experiment_7343_v645_contract as mod
from carnot.reporting.experiment_7303_validation_scope import REQUIRED_CHECK_NAMES


ROOT = Path(__file__).resolve().parents[2]


def _receipt(name: str, passed: bool = True) -> dict[str, Any]:
    """Create a complete subprocess receipt for an injected bounded check."""

    return {
        "name": name,
        "command": f"check {name}",
        "command_argv": ["check", name],
        "scope": "test_fixture",
        "exit_code": 0 if passed else 1,
        "duration_s": 0.01,
        "log_path": f"/tmp/{name}.log",
        "log_sha256": "sha256:" + "1" * 64,
        "passed": passed,
        "timed_out": False,
        "output_tail": "ok" if passed else "failed",
    }


def _validation(**kwargs: Any) -> dict[str, Any]:
    """Return all current checks without starting nested test processes."""

    assert kwargs["test_paths"] == [str(mod.TEST_PATH)]
    assert kwargs["changed_modules"] == [str(mod.MODULE_PATH)]
    names = [*mod.ROADMAP_CHECK_NAMES, *REQUIRED_CHECK_NAMES]
    return {
        "validation_receipts": [_receipt(name) for name in names],
        "required_checks_passed": True,
        "repository_health": {
            "status": "degraded_open",
            "incident_open": True,
            "historical_failures": list(kwargs["historical_failures"]),
            "historical_failure_count": len(kwargs["historical_failures"]),
            "unresolved_collection_error_observation_count": 0,
            "affects_required_checks": False,
        },
    }


def _terminal(_root: Path, candidate: Path, _raw_dir: Path) -> list[dict[str, Any]]:
    """Prove the measured candidate exists before terminal checks run."""

    assert candidate.is_file()
    return [_receipt(name) for name in mod.TERMINAL_CHECK_NAMES]


def _fetch(url: str) -> dict[str, Any]:
    """Return deterministic primary-page bodies for the happy path."""

    paper_id = next(key for key in mod.SOURCE_IDS if key in url)
    return {"ok": True, "status_code": 200, "body": f"primary {paper_id}", "error": None}


def test_scenario_report_7343_authority_uses_matching_next_only(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7343-AUTHORITY selects by milestone, not file presence."""

    active = {"milestone": mod.MILESTONE, "tasks": []}
    staged = {"milestone": mod.MILESTONE, "tasks": [{"id": "staged"}]}
    (tmp_path / mod.ACTIVE_ROADMAP_PATH).write_text(yaml.safe_dump(active))
    (tmp_path / mod.NEXT_ROADMAP_PATH).write_text(yaml.safe_dump(staged))
    path, document, _content, candidates = mod.select_yaml_authority(tmp_path)
    assert path == mod.NEXT_ROADMAP_PATH
    assert document == staged
    assert [row["available"] for row in candidates] == [True, True]

    (tmp_path / mod.NEXT_ROADMAP_PATH).write_text(yaml.safe_dump({"milestone": "stale"}))
    path, document, _content, candidates = mod.select_yaml_authority(tmp_path)
    assert path == mod.ACTIVE_ROADMAP_PATH
    assert document == active
    assert [row["available"] for row in candidates] == [False, True]

    (tmp_path / mod.ACTIVE_ROADMAP_PATH).unlink()
    path, document, content, candidates = mod.select_yaml_authority(tmp_path)
    assert (path, document, content) == (None, None, None)
    assert "FileNotFoundError" in candidates[1]["observed_value"]

    (tmp_path / mod.NEXT_ROADMAP_PATH).write_text("{}")
    path, document, content, candidates = mod.select_yaml_authority(tmp_path)
    assert (path, document, content) == (None, None, None)
    assert "ValueError" in candidates[0]["observed_value"]


def test_scenario_report_7343_contract_rejects_named_mutations(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7343-CONTRACT rejects every required parity mutation."""

    roadmap = yaml.safe_load((ROOT / mod.ACTIVE_ROADMAP_PATH).read_text())
    markdown = (ROOT / mod.DESIGN_PATH).read_text()
    result = mod.evaluate_contract(markdown, roadmap)
    assert result["passed"] is True
    assert [row["unit_id"] for row in result["contract_rows"]] == list(mod.EXPECTED_ID_ORDER)
    assert all(row["passed"] for row in result["contract_rows"])
    assert result["gate_declaration_result"]["passed"] is True

    mutations = mod.run_contract_mutations(markdown, roadmap, tmp_path)
    assert [row["mutation"] for row in mutations] == [
        "missing_task",
        "reordered_id",
        "stale_milestone",
        "changed_path",
        "changed_title",
        "changed_phase",
        "misspelled_gate_field",
        "malformed_operator",
    ]
    assert all(row["rejected"] for row in mutations)

    malformed = markdown.replace(
        'verdict_class in ["positive", "circular_positive", "null"]',
        'verdict_class in "positive"',
        1,
    )
    with pytest.raises(ValueError, match="JSON list"):
        mod.parse_markdown_contract(malformed)

    changed = deepcopy(roadmap)
    changed["tasks"][3]["gated_on"][0]["upstream"] = mod.FIRST_TASK_ID
    assert mod.validate_gate_declarations(changed)["passed"] is False
    assert mod.validate_gate_declarations({"tasks": ["malformed"]})["passed"] is False


def test_scenario_report_7343_gate_controls_reject_all_six_adverse_cases(
    tmp_path: Path,
) -> None:
    """SCENARIO-REPORT-7343-GATES uses the real reader for every edge case."""

    roadmap = yaml.safe_load((ROOT / mod.ACTIVE_ROADMAP_PATH).read_text())
    rows = mod.run_gate_controls(roadmap, tmp_path)
    gate_count = sum(len(task.get("gated_on") or []) for task in roadmap["tasks"])
    assert len(rows) == gate_count * 7
    assert mod.gate_controls_complete(rows, gate_count) is True
    assert {row["case"] for row in rows} == set(mod.GATE_CASES)
    assert all(row["combined_conductor_passed"] is (row["case"] == "passing") for row in rows)
    assert all(row["experiment_precondition_passed"] is (row["case"] == "passing") for row in rows)
    quarantine = next(row for row in rows if row["case"] == "quarantined_score_one")
    assert any(
        reason.startswith("quarantined_upstream")
        for reason in quarantine["experiment_precondition_reasons"]
    )

    broken = deepcopy(rows)
    broken[-1]["combined_conductor_passed"] = True
    assert mod.gate_controls_complete(broken, gate_count) is False
    assert mod.gate_controls_complete([], gate_count) is False
    broken = deepcopy(rows)
    broken[0]["case"] = "changed"
    assert mod.gate_controls_complete(broken, gate_count) is False
    broken = deepcopy(rows)
    broken[0]["experiment_precondition_passed"] = False
    assert mod.gate_controls_complete(broken, gate_count) is False


def test_scenario_report_7343_sources_preserve_access_outcomes_and_limits() -> None:
    """SCENARIO-REPORT-7343-SOURCES keeps failures separate from contract parity."""

    calls: list[str] = []

    def fetch(url: str) -> dict[str, Any]:
        calls.append(url)
        if "2604.13283" in url:
            return {"ok": False, "status_code": 429, "body": "", "error": "HTTP 429"}
        if "2607.20792" in url:
            return {
                "ok": False,
                "status_code": 403,
                "body": "browser challenge",
                "error": None,
            }
        if "2607.29549" in url:
            raise TimeoutError("bounded timeout")
        return _fetch(url)

    source_map, access = mod.collect_source_method_map(fetch)
    assert calls == [row["url"] for row in mod.ACCESS_SPECS]
    assert [row["access_outcome"] for row in access] == [
        "http_success",
        "http_429",
        "browser_challenge",
        "access_failed",
    ]
    assert mod.source_map_complete(source_map) is True
    assert mod.source_map_complete(source_map[:-1]) is False
    assert any("neural oracle" in row["limitations"].lower() for row in source_map)
    assert any("shared executor" in row["limitations"].lower() for row in source_map)
    assert {task for row in source_map for task in row["task_mapping"]} == {
        "exp7349-prospective-learning",
        "exp7351-acquisition-prototype",
        "exp7354-arc-transfer",
    }


def test_scenario_report_7343_validation_builds_all_declared_scopes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SCENARIO-REPORT-7343-VALIDATION includes seven structural checks."""

    commands = mod.roadmap_command_specs(ROOT, mod.ACTIVE_ROADMAP_PATH)
    assert [command.name for command in commands] == list(mod.ROADMAP_CHECK_NAMES)
    captured: list[str] = []

    def roadmap_runner(_root: Path, specs: Any, **_kwargs: Any) -> list[dict[str, Any]]:
        captured.extend(spec.name for spec in specs)
        return [_receipt(spec.name) for spec in specs]

    def scoped_runner(_root: Path, **kwargs: Any) -> dict[str, Any]:
        assert kwargs["basetemp"].parent.is_dir()
        return {
            "validation_receipts": [_receipt(name) for name in REQUIRED_CHECK_NAMES],
            "required_checks_passed": True,
            "repository_health": {"historical_failures": kwargs["historical_failures"]},
        }

    result = mod.run_affected_validation(
        ROOT,
        mod.ACTIVE_ROADMAP_PATH,
        tmp_path,
        roadmap_runner=roadmap_runner,
        scoped_runner=scoped_runner,
    )
    assert captured == list(mod.ROADMAP_CHECK_NAMES)
    assert result["required_checks_passed"] is True

    def failed_runner(_root: Path, specs: Any, **_kwargs: Any) -> list[dict[str, Any]]:
        rows = [_receipt(spec.name) for spec in specs]
        rows[0] = _receipt(specs[0].name, False)
        return rows

    failed = mod.run_affected_validation(
        ROOT,
        mod.ACTIVE_ROADMAP_PATH,
        tmp_path,
        roadmap_runner=failed_runner,
        scoped_runner=scoped_runner,
    )
    assert failed["required_checks_passed"] is False

    terminal_names: list[str] = []

    def terminal_runner(_root: Path, specs: Any, **_kwargs: Any) -> list[dict[str, Any]]:
        terminal_names.extend(spec.name for spec in specs)
        return [_receipt(spec.name) for spec in specs]

    monkeypatch.setattr(mod, "run_commands", terminal_runner)
    candidate = tmp_path / "candidate.json"
    candidate.write_text("{}")
    receipts = mod.run_terminal_validation(ROOT, candidate, tmp_path)
    assert terminal_names == list(mod.TERMINAL_CHECK_NAMES)
    assert [row["name"] for row in receipts] == list(mod.TERMINAL_CHECK_NAMES)
    assert mod._historical_failures(tmp_path) == []

    preconditions, _authority, _roadmap, _content = mod._preconditions(
        ROOT,
        Path("/dev/null/result.json"),
        tmp_path / "precondition-raw",
        tmp_path / "precondition-checkpoint.json",
    )
    result_dir = next(row for row in preconditions if row["check"] == "result_directory")
    assert result_dir["available"] is False


def test_scenario_report_7343_artifact_recomputes_terminal_state(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7343-ARTIFACT validates a complete measured receipt."""

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
    assert artifact["contract_complete_score"] == 1
    assert artifact["verdict_class"] == "circular_positive"
    assert artifact["MODEL_SPECS"] == []
    assert artifact["model_invoked"] is False
    assert set(artifact["invocation_counts"].values()) == {0}
    assert artifact["rows"] == artifact["contract_rows"]
    assert artifact["repository_health"]["affects_required_checks"] is False
    assert mod.independent_reduce(artifact) == []
    assert mod.validate_artifact(artifact, root=ROOT) == []
    assert json.loads((tmp_path / "result.json").read_text()) == artifact
    assert json.loads((tmp_path / "checkpoint.json").read_text())["status"] == "running"

    changed = deepcopy(artifact)
    changed["contract_rows"][0]["passed"] = False
    assert "contract_rows_invalid" in mod.independent_reduce(changed)
    changed = deepcopy(artifact)
    changed["source_method_map"] = []
    assert "source_method_map_invalid" in mod.independent_reduce(changed)
    changed = deepcopy(artifact)
    changed["gate_control_rows"] = []
    assert "gate_controls_invalid" in mod.independent_reduce(changed)
    changed = deepcopy(artifact)
    changed["raw_authority_rows"] = []
    assert "raw_authorities_invalid" in mod.independent_reduce(changed)
    changed = deepcopy(artifact)
    changed["contract_mutation_rows"] = []
    assert "contract_mutations_invalid" in mod.independent_reduce(changed)
    changed = deepcopy(artifact)
    changed["phase_spans"] = [{"start_elapsed_s": 2, "end_elapsed_s": 1}]
    assert "phase_spans_invalid" in mod.independent_reduce(changed)
    changed = deepcopy(artifact)
    changed["rows"] = []
    assert "rows_invalid" in mod.independent_reduce(changed)
    changed = deepcopy(artifact)
    changed["reproducibility_checksum"] = "sha256:" + "0" * 64
    assert "reproducibility_checksum_invalid" in mod.validate_artifact(changed, root=ROOT)
    changed = deepcopy(artifact)
    changed["MODEL_SPECS"] = ["unexpected"]
    assert "model_contract_invalid" in mod.validate_artifact(changed, root=ROOT)
    assert mod.validate_artifact([], root=ROOT) == ["artifact_mapping_required"]
    changed = deepcopy(artifact)
    del changed["schema"]
    assert mod.validate_artifact(changed, root=ROOT) == ["missing_required_field:schema"]
    validation_mutations = {
        "identity_invalid": ("milestone", "2026.09.000"),
        "lifecycle_invalid": ("status", "running"),
        "substrate_invalid": ("execution_venue", "remote"),
        "field_principles_invalid": ("field_principles", {}),
        "flagged_adversarial_invalid": ("flagged_adversarial", True),
        "honest_verdict_invalid": ("honest_verdict", "complete_changed"),
    }
    for error, (field, value) in validation_mutations.items():
        changed = deepcopy(artifact)
        changed[field] = value
        assert error in mod.validate_artifact(changed, root=ROOT)
    changed = deepcopy(artifact)
    changed["validation_receipts"].pop()
    assert "validation_receipts_invalid" in mod.validate_artifact(changed, root=ROOT)
    changed = deepcopy(artifact)
    first_source = next(iter(changed["source_artifact_hashes"]))
    changed["source_artifact_hashes"][first_source] = "sha256:" + "0" * 64
    assert "source_hash_mismatch" in mod.validate_artifact(changed, root=ROOT)


def test_req_report_7343_blocked_disqualified_and_date_paths(tmp_path: Path) -> None:
    """REQ-REPORT-7343 keeps missing inputs distinct from readable mismatches."""

    blocked = mod.build_artifact(
        tmp_path,
        mod.RUN_DATE,
        output_path=tmp_path / "blocked.json",
        raw_dir=tmp_path / "raw",
        checkpoint_path=tmp_path / "checkpoint.json",
        fetcher=_fetch,
        validation_runner=_validation,
        terminal_runner=_terminal,
    )
    assert blocked["status"] == "blocked"
    assert blocked["verdict_class"] == "blocked"
    assert blocked["contract_complete_score"] == 0
    assert blocked["gate_check_summary"]["passed"] is False
    assert mod.validate_artifact(blocked, root=tmp_path) == []

    state: dict[str, Any] = {
        "preconditions_checked": [],
        "acceptance_gate_results": [
            {
                "criterion": "contract_exact",
                "expected": True,
                "observed": False,
                "passed": False,
                "principle": "Two files describe the same executable work.",
            }
        ],
    }
    mod.apply_terminal_state(state)
    assert state["verdict_class"] == "disqualified"
    assert state["honest_verdict"].startswith("complete_disqualified")

    assert mod.date_argument(mod.RUN_DATE) == mod.RUN_DATE
    with pytest.raises(Exception, match="run date must be"):
        mod.date_argument("20260915")
