"""Tests for the V647 advisory source and exact-contract receipt.

Spec refs: REQ-REPORT-7369 and SCENARIO-REPORT-7369-*.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from carnot import experiment_7369_v647_contract as mod
from carnot.experiment_7358_v646_validation_contract import (
    build_command_plan,
    validate_command_plan,
)
from carnot.reporting.experiment_7303_validation_scope import REQUIRED_CHECK_NAMES


ROOT = Path(__file__).resolve().parents[2]


def _receipt(name: str, passed: bool = True) -> dict[str, Any]:
    """Create one command receipt without starting a nested validation run."""

    row = {
        "name": name,
        "command": f"check {name}",
        "command_argv": ["check", name],
        "command_environment": {},
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
            "carnot.experiment_7369_v647_contract": str((ROOT / mod.MODULE_PATH).resolve())
        }
    return row


def _validation(**kwargs: Any) -> dict[str, Any]:
    """Return the exact V647 plan while asserting its affected file scope."""

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
    """Require a reloadable measured candidate before terminal checks pass."""

    value = json.loads(candidate.read_text(encoding="utf-8"))
    assert value["status"] == "complete"
    assert mod.validate_artifact(value, root=ROOT, require_terminal=False) == []
    return [_receipt(name) for name in mod.TERMINAL_CHECK_NAMES]


def _fetch(url: str) -> dict[str, Any]:
    """Return deterministic bytes while preserving Semantic Scholar rate limits."""

    if "semanticscholar" in url:
        return {"ok": False, "status_code": 429, "body": "", "error": "HTTP 429"}
    return {"ok": True, "status_code": 200, "body": f"primary:{url}", "error": None}


def test_scenario_report_7369_contract_selects_and_matches_exact_authorities(
    tmp_path: Path,
) -> None:
    """SCENARIO-REPORT-7369-CONTRACT checks exact selection and all task fields."""

    active = yaml.safe_load((ROOT / mod.ACTIVE_ROADMAP_PATH).read_text(encoding="utf-8"))
    staged = deepcopy(active)
    (tmp_path / mod.ACTIVE_ROADMAP_PATH).write_text(yaml.safe_dump(active), encoding="utf-8")
    (tmp_path / mod.NEXT_ROADMAP_PATH).write_text(yaml.safe_dump(staged), encoding="utf-8")
    path, document, _content, candidates = mod.select_yaml_authority(tmp_path)
    assert path == mod.NEXT_ROADMAP_PATH
    assert document == staged
    assert [row["available"] for row in candidates] == [True, True]

    staged["milestone"] = "2026.09.646"
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
    assert mod.validate_gate_declarations({"tasks": ["bad"]})["passed"] is False


def test_scenario_report_7369_mutations_and_fresh_ids_fail_closed(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7369-MUTATIONS rejects every named private mutation."""

    roadmap = yaml.safe_load((ROOT / mod.ACTIVE_ROADMAP_PATH).read_text(encoding="utf-8"))
    markdown = (ROOT / mod.DESIGN_PATH).read_text(encoding="utf-8")
    rows = mod.run_contract_mutations(markdown, roadmap, tmp_path)
    assert [row["mutation"] for row in rows] == list(mod.CONTRACT_MUTATION_NAMES)
    assert all(row["baseline_passed"] and row["rejected"] for row in rows)
    assert {row["rejecting_check"] for row in rows} == {
        "exact_contract",
        "roadmap_schema",
        "retirement_contract",
    }

    malformed = deepcopy(roadmap)
    malformed["tasks"][0]["prior_failures"][0].pop("retire_if_same_verdict")
    assert mod.validate_retirement_declarations(malformed)["passed"] is False
    assert mod.validate_retirement_declarations({"tasks": ["bad"]})["passed"] is False

    freshness = mod.check_fresh_ids(ROOT, roadmap)
    assert freshness["passed"] is True
    assert [row["experiment_number"] for row in freshness["rows"]] == list(range(7369, 7381))
    duplicate = deepcopy(roadmap)
    duplicate["tasks"][1]["id"] = duplicate["tasks"][0]["id"]
    assert mod.check_fresh_ids(ROOT, duplicate)["passed"] is False
    assert mod.check_fresh_ids(tmp_path, roadmap)["passed"] is True


def test_scenario_report_7369_sources_preserve_access_and_defer_boundaries() -> None:
    """SCENARIO-REPORT-7369-SOURCES keeps access failures out of method claims."""

    source_rows, access_rows = mod.collect_source_method_rows(_fetch)
    assert [row["access_id"] for row in access_rows] == list(mod.ACCESS_IDS)
    assert [row["access_outcome"] for row in access_rows[-2:]] == ["http_429", "http_429"]
    assert mod.source_rows_complete(source_rows, access_rows) is True
    assert mod.source_rows_complete(source_rows[:-1], access_rows) is False
    assert {row["method_family"] for row in source_rows} == {
        "parameterized_2sat",
        "semantic_realizability",
        "verifier_authority",
        "kan_deferral",
        "ebt_deferral",
        "ising_hardware_cost",
    }
    assert all(row["local_science_claimed"] is False for row in source_rows)
    assert all(row["frozen_roster_changed"] is False for row in source_rows)
    assert all(row["runtime_dependency_added"] is False for row in source_rows)

    outcomes = iter(("challenge", "failure", "exception"))

    def failing_fetch(_url: str) -> dict[str, Any]:
        kind = next(outcomes, "success")
        if kind == "challenge":
            return {"ok": False, "status_code": 403, "body": "browser challenge", "error": None}
        if kind == "failure":
            return {"ok": False, "status_code": 503, "body": "", "error": "unavailable"}
        if kind == "exception":
            raise TimeoutError("bounded timeout")
        return {"ok": True, "status_code": 200, "body": "ok", "error": None}

    _rows, failed_access = mod.collect_source_method_rows(failing_fetch)
    assert [row["access_outcome"] for row in failed_access[:3]] == [
        "browser_challenge",
        "access_failed",
        "access_failed",
    ]


def test_scenario_report_7369_validation_plan_is_derived_and_scoped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-REPORT-7369-VALIDATION derives exact names from Exp7358."""

    private = tmp_path / "private"
    commands = build_command_plan(ROOT, mod.V647_MANIFEST, private)
    assert [command.name for command in commands] == list(REQUIRED_CHECK_NAMES)
    assert mod.ALL_REQUIRED_CHECK_NAMES == (
        *mod.ROADMAP_CHECK_NAMES,
        *REQUIRED_CHECK_NAMES,
        *mod.TERMINAL_CHECK_NAMES,
    )
    assert "full_python_suite" not in mod.ALL_REQUIRED_CHECK_NAMES
    assert validate_command_plan(ROOT, mod.V647_MANIFEST, commands) == []

    broad = deepcopy(commands)
    broad[1] = type(commands[1])(
        "full_python_suite",
        (str(ROOT / ".venv/bin/pytest"), "tests/python", "-q"),
        "repository",
    )
    assert "unexpected_command:full_python_suite" in validate_command_plan(
        ROOT, mod.V647_MANIFEST, broad
    )

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

    def command_runner(_root: Path, specs: Any, **_kwargs: Any) -> list[dict[str, Any]]:
        terminal_names.extend(spec.name for spec in specs)
        return [_receipt(spec.name) for spec in specs]

    monkeypatch.setattr(mod, "run_commands", command_runner)
    candidate = tmp_path / "candidate.json"
    candidate.write_text("{}", encoding="utf-8")
    assert [row["name"] for row in mod.run_terminal_validation(ROOT, candidate, tmp_path)] == list(
        mod.TERMINAL_CHECK_NAMES
    )
    assert terminal_names == list(mod.TERMINAL_CHECK_NAMES)


def test_scenario_report_7369_artifact_replays_and_rejects_tampering(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7369-ARTIFACT recomputes the exact terminal receipt."""

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
    assert artifact["inference_substrate_class"] == "aggregation"
    assert artifact["study_mapping_ingested"] is True
    assert artifact["semantic_scholar_access_outcomes"] == ["http_429", "http_429"]
    assert [row["access_outcome"] for row in artifact["dated_delta_access_outcomes"]] == [
        "http_429",
        "http_429",
    ]
    assert all(
        row["inferred_citation_count"] is None for row in artifact["dated_delta_access_outcomes"]
    )
    assert [row["name"] for row in artifact["validation_receipts"]] == list(
        mod.ALL_REQUIRED_CHECK_NAMES
    )
    assert mod.independent_reduce(artifact, require_terminal=True) == []
    assert mod.validate_artifact(artifact, root=ROOT, require_terminal=True) == []
    assert json.loads((tmp_path / "result.json").read_text(encoding="utf-8")) == artifact
    assert (
        json.loads((tmp_path / "checkpoint.json").read_text(encoding="utf-8"))["status"]
        == "running"
    )

    mutations = {
        "contract_rows_invalid": ("contract_rows", []),
        "mutation_rows_invalid": ("mutation_rows", []),
        "source_method_rows_invalid": ("source_method_rows", []),
        "access_rows_invalid": ("source_access_rows", []),
        "gate_controls_invalid": ("gate_control_rows", []),
        "fresh_ids_invalid": ("fresh_id_rows", []),
        "raw_authorities_invalid": ("raw_authority_rows", []),
        "rows_invalid": ("rows", []),
        "phase_spans_invalid": (
            "phase_spans",
            [{"start_elapsed_s": 2.0, "end_elapsed_s": 1.0}],
        ),
        "validation_receipts_invalid": ("validation_receipts", []),
    }
    for error, (field, value) in mutations.items():
        changed = deepcopy(artifact)
        changed[field] = value
        assert error in mod.independent_reduce(changed, require_terminal=True)

    changed = deepcopy(artifact)
    changed["reproducibility_checksum"] = "sha256:" + "0" * 64
    assert "reproducibility_checksum_invalid" in mod.validate_artifact(changed, root=ROOT)
    changed = deepcopy(artifact)
    changed["MODEL_SPECS"] = ["unexpected"]
    assert "model_contract_invalid" in mod.validate_artifact(changed, root=ROOT)
    for error, field, value in (
        ("identity_invalid", "milestone", "2026.09.000"),
        ("lifecycle_invalid", "status", "running"),
        ("substrate_invalid", "execution_venue", "remote"),
        ("field_principles_invalid", "field_principles", {}),
        ("honest_verdict_invalid", "honest_verdict", "complete_changed"),
    ):
        changed = deepcopy(artifact)
        changed[field] = value
        assert error in mod.validate_artifact(changed, root=ROOT)
    changed = deepcopy(artifact)
    changed["source_artifact_hashes"][next(iter(changed["source_artifact_hashes"]))] = (
        "sha256:" + "0" * 64
    )
    assert "source_hash_mismatch" in mod.validate_artifact(changed, root=ROOT)
    assert mod.validate_artifact([], root=ROOT) == ["artifact_mapping_required"]
    changed = deepcopy(artifact)
    del changed["schema"]
    assert mod.validate_artifact(changed, root=ROOT) == ["missing_required_field:schema"]

    disqualified: dict[str, Any] = {
        "preconditions_checked": [],
        "acceptance_gate_results": [
            {"criterion": "contract_exact", "expected": True, "observed": False, "passed": False}
        ],
    }
    mod._terminal_state(disqualified)
    assert disqualified["verdict_class"] == "disqualified"
    assert disqualified["contract_complete_score"] == 0


def test_req_report_7369_blocks_missing_input_and_rejects_wrong_date(tmp_path: Path) -> None:
    """REQ-REPORT-7369 distinguishes an external block from owned contract failure."""

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

    assert mod.date_argument(mod.RUN_DATE) == mod.RUN_DATE
    with pytest.raises(Exception, match="run date must be"):
        mod.date_argument("20260916")
