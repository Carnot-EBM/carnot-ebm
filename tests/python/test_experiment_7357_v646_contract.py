"""Tests for the V646 advisory contract receipt.

Spec refs: REQ-REPORT-7357 and SCENARIO-REPORT-7357-*.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from carnot import experiment_7357_v646_contract as mod
from carnot.experiment_7329_v644_contract import parse_yaml_contract
from carnot.reporting.experiment_7303_validation_scope import REQUIRED_CHECK_NAMES


ROOT = Path(__file__).resolve().parents[2]


def _receipt(name: str, passed: bool = True) -> dict[str, Any]:
    """Create a truthful-shaped receipt without starting a nested command."""

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
    """Return every affected receipt while checking the declared scope."""

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
    """Confirm that terminal checks receive a reloadable measured candidate."""

    assert json.loads(candidate.read_text())["status"] == "complete"
    return [_receipt(name) for name in mod.TERMINAL_CHECK_NAMES]


def _fetch(url: str) -> dict[str, Any]:
    """Return deterministic primary-page bytes for source-map tests."""

    paper_id = next(source_id for source_id in mod.SOURCE_IDS if source_id in url)
    return {"ok": True, "status_code": 200, "body": f"primary {paper_id}", "error": None}


def _matching_markdown(roadmap: dict[str, Any]) -> str:
    """Render a small independent Markdown fixture from explicit parsed fields."""

    parsed = parse_yaml_contract(roadmap)
    lines = [
        "# V646 test contract",
        "",
        f"**Milestone:** {mod.MILESTONE}",
        "",
        "## Exact Task Contract",
        "",
        "| Order | Task ID | Exact title | Deliverable | Phase | Substrate class | Structured gate |",
        "|---|---|---|---|---|---|---|",
    ]
    for task in parsed["tasks"]:
        gates = []
        for gate in task["gates"]:
            value = json.dumps(gate["value"])
            gates.append(f"{gate['upstream']}.{gate['artifact_field']} {gate['op']} {value}")
        lines.append(
            "| {order} | {id} | {title} | {deliverable} | {phase} | {substrate} | {gates} |".format(
                **{**task, "gates": "; ".join(gates) or "None"}
            )
        )
    return "\n".join(lines) + "\n"


def test_scenario_report_7357_authority_selects_matching_staged_only(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7357-AUTHORITY selects bytes by exact milestone."""

    active = {"milestone": mod.MILESTONE, "tasks": []}
    staged = {"milestone": mod.MILESTONE, "tasks": [{"id": "staged"}]}
    (tmp_path / mod.ACTIVE_ROADMAP_PATH).write_text(yaml.safe_dump(active))
    (tmp_path / mod.NEXT_ROADMAP_PATH).write_text(yaml.safe_dump(staged))
    path, document, _content, candidates = mod.select_yaml_authority(tmp_path)
    assert path == mod.NEXT_ROADMAP_PATH
    assert document == staged
    assert [row["available"] for row in candidates] == [True, True]

    (tmp_path / mod.NEXT_ROADMAP_PATH).write_text("milestone: stale\n")
    path, document, _content, candidates = mod.select_yaml_authority(tmp_path)
    assert path == mod.ACTIVE_ROADMAP_PATH
    assert document == active
    assert [row["available"] for row in candidates] == [False, True]

    (tmp_path / mod.ACTIVE_ROADMAP_PATH).unlink()
    (tmp_path / mod.NEXT_ROADMAP_PATH).write_text("[]\n")
    path, document, content, candidates = mod.select_yaml_authority(tmp_path)
    assert (path, document, content) == (None, None, None)
    assert "ValueError" in candidates[0]["observed_value"]
    assert "FileNotFoundError" in candidates[1]["observed_value"]


def test_scenario_report_7357_contract_rejects_stale_pair_and_mutations(
    tmp_path: Path,
) -> None:
    """SCENARIO-REPORT-7357-CONTRACT compares independent task and gate fields."""

    roadmap = yaml.safe_load((ROOT / mod.ACTIVE_ROADMAP_PATH).read_text())
    matching = _matching_markdown(roadmap)
    exact = mod.evaluate_contract(matching, roadmap)
    assert exact["passed"] is True
    assert [row["unit_id"] for row in exact["contract_rows"]] == list(mod.EXPECTED_ID_ORDER)
    assert all(row["passed"] for row in exact["contract_rows"])
    assert exact["active_exact_contract_count"] == 1
    assert exact["gate_declaration_result"]["passed"] is True

    stale = mod.evaluate_contract((ROOT / mod.DESIGN_PATH).read_text(), roadmap)
    assert stale["passed"] is False
    assert len(stale["contract_rows"]) == 12
    assert stale["markdown_milestone"] == "2026.09.645"
    assert len(stale["unexpected_markdown_rows"]) == 14
    assert set(stale["missing_markdown_ids"]) == set(mod.EXPECTED_ID_ORDER)

    mutations = mod.run_contract_mutations(matching, roadmap, tmp_path)
    assert [row["mutation"] for row in mutations] == list(mod.CONTRACT_MUTATION_NAMES)
    assert all(row["baseline_passed"] and row["rejected"] for row in mutations)
    malformed_mutations = mod.run_contract_mutations("not a contract", roadmap, tmp_path / "bad")
    assert all(not row["baseline_passed"] and row["rejected"] for row in malformed_mutations)

    duplicate = matching + "\n## Exact Task Contract\n\n| Order | Task ID |\n"
    with pytest.raises(ValueError, match="exactly one active"):
        mod.parse_active_markdown_contract(duplicate)
    assert mod.validate_gate_declarations({"tasks": ["bad"]})["passed"] is False


def test_scenario_report_7357_gates_fail_closed_with_real_reader(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7357-GATES tests all seven cases through the conductor."""

    roadmap = yaml.safe_load((ROOT / mod.ACTIVE_ROADMAP_PATH).read_text())
    fixture = {"tasks": [roadmap["tasks"][1], roadmap["tasks"][2]]}
    rows = mod.run_gate_controls(fixture, tmp_path)
    gate_count = len(fixture["tasks"][1]["gated_on"])
    assert len(rows) == gate_count * len(mod.GATE_CASES)
    assert mod.gate_controls_complete(rows, gate_count) is True
    assert all(row["combined_conductor_passed"] is (row["case"] == "passing") for row in rows)
    assert all(row["experiment_precondition_passed"] is (row["case"] == "passing") for row in rows)

    broken = deepcopy(rows)
    broken[-1]["combined_conductor_passed"] = True
    assert mod.gate_controls_complete(broken, gate_count) is False


def test_scenario_report_7357_sources_keep_access_and_causal_limits() -> None:
    """SCENARIO-REPORT-7357-SOURCES keeps five outcomes and causal boundaries."""

    def fetch(url: str) -> dict[str, Any]:
        if "2609.14857" in url:
            return {"ok": False, "status_code": 429, "body": "", "error": "HTTP 429"}
        if "2604.13283" in url:
            return {"ok": False, "status_code": 403, "body": "challenge", "error": None}
        if "2607.20792" in url:
            raise TimeoutError("bounded timeout")
        return _fetch(url)

    source_map, access = mod.collect_source_method_map(fetch)
    assert [row["access_outcome"] for row in access] == [
        "http_success",
        "http_429",
        "browser_challenge",
        "access_failed",
        "http_success",
    ]
    assert mod.source_map_complete(source_map) is True
    assert mod.source_map_complete(source_map[:-1]) is False
    assert {task for row in source_map for task in row["destination_experiment_ids"]} == {
        "exp7361-fresh-plan-capture",
        "exp7362-prospective-learning",
        "exp7365-supervisor-support",
        "exp7366-supervisor-live",
    }
    dream = next(row for row in source_map if row["source_id"] == "2609.14858")
    assert dream["observed_support_replay"] is True
    assert dream["causal_counterfactual_supported"] is False


def test_scenario_report_7357_validation_uses_exact_scopes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-REPORT-7357-VALIDATION retains seven structural check receipts."""

    commands = mod.roadmap_command_specs(ROOT, mod.ACTIVE_ROADMAP_PATH)
    assert [command.name for command in commands] == list(mod.ROADMAP_CHECK_NAMES)

    def roadmap_runner(_root: Path, specs: Any, **_kwargs: Any) -> list[dict[str, Any]]:
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

    captured: list[str] = []

    def terminal_runner(_root: Path, specs: Any, **_kwargs: Any) -> list[dict[str, Any]]:
        captured.extend(spec.name for spec in specs)
        return [_receipt(spec.name) for spec in specs]

    monkeypatch.setattr(mod, "run_commands", terminal_runner)
    candidate = tmp_path / "candidate.json"
    candidate.write_text("{}")
    assert [row["name"] for row in mod.run_terminal_validation(ROOT, candidate, tmp_path)] == list(
        mod.TERMINAL_CHECK_NAMES
    )
    assert captured == list(mod.TERMINAL_CHECK_NAMES)
    assert mod._historical_failures(tmp_path) == []

    preconditions, _authority, _roadmap, _content = mod._preconditions(
        ROOT,
        Path("/dev/null/result.json"),
        tmp_path / "precondition-raw",
        tmp_path / "precondition-checkpoint.json",
    )
    result_dir = next(row for row in preconditions if row["check"] == "result_directory")
    assert result_dir["available"] is False


def test_scenario_report_7357_artifact_recomputes_real_stale_outcome(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7357-ARTIFACT preserves the real pair's disqualification."""

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
    assert artifact["verdict_class"] == "disqualified"
    assert artifact["contract_complete_score"] == 0
    assert artifact["MODEL_SPECS"] == []
    assert artifact["model_invoked"] is False
    assert set(artifact["invocation_counts"].values()) == {0}
    assert len(artifact["contract_rows"]) == 12
    assert artifact["rows"] == artifact["contract_rows"]
    assert artifact["science_readiness_score"] == 0
    assert artifact["science_value_score"] == 0
    assert artifact["science_promotion_score"] == 0
    assert mod.independent_reduce(artifact) == []
    assert mod.validate_artifact(artifact, root=ROOT) == []
    assert json.loads((tmp_path / "result.json").read_text()) == artifact
    assert json.loads((tmp_path / "checkpoint.json").read_text())["status"] == "running"

    for error, field, value in (
        ("contract_rows_invalid", "contract_rows", []),
        ("source_method_map_invalid", "source_method_map", []),
        ("gate_controls_invalid", "gate_control_rows", []),
        ("raw_authorities_invalid", "raw_authority_rows", []),
        ("contract_mutations_invalid", "contract_mutation_rows", []),
        ("phase_spans_invalid", "phase_spans", [{"start_elapsed_s": 2, "end_elapsed_s": 1}]),
        ("rows_invalid", "rows", []),
    ):
        changed = deepcopy(artifact)
        changed[field] = value
        assert error in mod.independent_reduce(changed)

    changed = deepcopy(artifact)
    changed["reproducibility_checksum"] = "sha256:" + "0" * 64
    assert "reproducibility_checksum_invalid" in mod.validate_artifact(changed, root=ROOT)
    changed = deepcopy(artifact)
    changed["MODEL_SPECS"] = ["unexpected"]
    assert "model_contract_invalid" in mod.validate_artifact(changed, root=ROOT)
    changed = deepcopy(artifact)
    changed["field_principles"] = {}
    assert "field_principles_invalid" in mod.validate_artifact(changed, root=ROOT)
    validation_mutations = {
        "identity_invalid": ("milestone", "2026.09.000"),
        "lifecycle_invalid": ("status", "running"),
        "substrate_invalid": ("execution_venue", "remote"),
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
    source = next(iter(changed["source_artifact_hashes"]))
    changed["source_artifact_hashes"][source] = "sha256:" + "0" * 64
    assert "source_hash_mismatch" in mod.validate_artifact(changed, root=ROOT)
    assert mod.validate_artifact([], root=ROOT) == ["artifact_mapping_required"]
    changed = deepcopy(artifact)
    del changed["schema"]
    assert mod.validate_artifact(changed, root=ROOT) == ["missing_required_field:schema"]


def test_req_report_7357_blocked_positive_and_date_paths(tmp_path: Path) -> None:
    """REQ-REPORT-7357 separates missing inputs from exact accounting success."""

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
    assert blocked["gate_check_summary"]["passed"] is False
    assert mod.validate_artifact(blocked, root=tmp_path) == []

    state: dict[str, Any] = {
        "preconditions_checked": [],
        "acceptance_gate_results": [
            {
                "criterion": "contract_exact",
                "expected": True,
                "observed": True,
                "passed": True,
                "principle": "Exact pair agreement controls this accounting receipt.",
            }
        ],
    }
    mod.apply_terminal_state(state)
    assert state["verdict_class"] == "circular_positive"
    assert state["contract_complete_score"] == 1

    assert mod.date_argument(mod.RUN_DATE) == mod.RUN_DATE
    with pytest.raises(Exception, match="run date must be"):
        mod.date_argument("20260916")
