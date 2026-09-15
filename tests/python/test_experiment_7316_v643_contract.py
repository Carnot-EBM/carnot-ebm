"""Tests for the V643 advisory contract receipt.

Spec refs: REQ-REPORT-7316 and SCENARIO-REPORT-7316-*.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from carnot import experiment_7316_v643_contract as mod
from carnot.reporting.experiment_7303_validation_scope import REQUIRED_CHECK_NAMES


ROOT = Path(__file__).resolve().parents[2]


def _markdown(roadmap: dict[str, Any]) -> str:
    """Build a separate Markdown authority from visible YAML values."""

    rows = []
    for order, task in enumerate(roadmap["tasks"], 1):
        gates = task.get("gated_on") or []
        gate_text = (
            "; ".join(
                f"{gate['upstream']}.{gate['artifact_field']} {gate['op']} "
                f"{json.dumps(gate['value'])}"
                for gate in gates
            )
            or "None"
        )
        rows.append(
            f"| {order} | {task['id']} | {task['title']} | {task['deliverable']} | "
            f"{task['phase']} | {gate_text} |"
        )
    return "\n".join(
        [
            "# V643 test contract",
            "",
            "**Milestone:** 2026.09.643",
            "",
            "## Exact Task Contract",
            "",
            "| Order | Task ID | Exact title | Deliverable | Phase | Structured gate |",
            "|---|---|---|---|---|---|",
            *rows,
            "",
            "## End",
        ]
    )


def _receipt(name: str, passed: bool = True) -> dict[str, Any]:
    """Create one complete injected validation receipt."""

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


def _passing_validation(**kwargs: Any) -> dict[str, Any]:
    """Return every required receipt without starting nested pytest."""

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


def _fetcher(url: str) -> dict[str, Any]:
    """Return a named primary page without network access."""

    paper_id = next(
        key for key in ("2604.13283", "2608.27038", "2503.15551", "2507.02092") if key in url
    )
    return {"ok": True, "status_code": 200, "body": f"primary page {paper_id}", "error": None}


def test_scenario_report_7316_authority_prefers_matching_staged_then_active(
    tmp_path: Path,
) -> None:
    """SCENARIO-REPORT-7316-AUTHORITY selects once by milestone identity."""

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


def test_scenario_report_7316_contract_rejects_every_named_mutation(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7316-CONTRACT covers all seven literal dimensions."""

    roadmap = yaml.safe_load((ROOT / mod.ACTIVE_ROADMAP_PATH).read_text())
    markdown = _markdown(roadmap)
    result = mod.evaluate_contract(markdown, roadmap)
    assert result["passed"] is True
    assert [row["unit_id"] for row in result["contract_rows"]] == list(mod.EXPECTED_ID_ORDER)
    assert all(row["passed"] and row["censored"] is False for row in result["contract_rows"])

    mutations = mod.run_contract_mutations(markdown, roadmap, tmp_path)
    assert [row["mutation"] for row in mutations] == [
        "count",
        "order",
        "id",
        "title",
        "phase",
        "path",
        "gate",
    ]
    assert all(row["rejected"] is True for row in mutations)
    invalid_phase = markdown.replace(
        "results/experiment_7316_v643_contract.json | 1 |",
        "results/experiment_7316_v643_contract.json | invalid |",
        1,
    )
    assert mod.evaluate_contract(invalid_phase, roadmap)["passed"] is False
    assert mod._canonical_gates(None) == []


def test_scenario_report_7316_gate_controls_use_real_reader(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7316-GATES keeps class checks beside numeric checks."""

    roadmap = yaml.safe_load((ROOT / mod.ACTIVE_ROADMAP_PATH).read_text())
    rows = mod.run_gate_controls(roadmap, tmp_path)
    gate_count = sum(len(task.get("gated_on") or []) for task in roadmap["tasks"])
    assert len(rows) == gate_count * 5
    assert mod.gate_controls_complete(rows, gate_count) is True
    assert {row["case"] for row in rows} == {
        "passing",
        "false",
        "missing_file",
        "missing_field",
        "disqualified_score_one",
    }
    disqualified = [row for row in rows if row["case"] == "disqualified_score_one"]
    assert all(row["combined_conductor_passed"] is False for row in disqualified)
    assert all(row["experiment_precondition_passed"] is False for row in disqualified)
    assert any(row["conductor_passed"] is True for row in disqualified)

    broken = deepcopy(rows)
    broken[-1]["combined_conductor_passed"] = True
    assert mod.gate_controls_complete(broken, gate_count) is False
    assert mod.upstream_precondition(None, "ready_score", 1)["passed"] is False
    assert (
        mod.upstream_precondition(
            {"status": "complete", "ready_score": 1, "flagged_adversarial": True},
            "ready_score",
            1,
        )["reason"]
        == "quarantined_upstream_rejected_before_field_consumption"
    )
    with pytest.raises(ValueError, match="malformed upstream"):
        mod._write_gate_artifacts(tmp_path / "malformed", {"bad-id": {}})

    assert mod.gate_controls_complete([], gate_count) is False
    wrong_case = deepcopy(rows)
    wrong_case[0]["case"] = "changed"
    assert mod.gate_controls_complete(wrong_case, gate_count) is False
    wrong_single = deepcopy(rows)
    wrong_single[0]["conductor_passed"] = False
    assert mod.gate_controls_complete(wrong_single, gate_count) is False
    missing_reason = deepcopy(rows)
    missing_inequality = next(
        row for row in missing_reason if row["case"] == "missing_field" and row["operator"] == "!="
    )
    missing_inequality["conductor_reason"] = ""
    assert mod.gate_controls_complete(missing_reason, gate_count) is False
    wrong_precondition = deepcopy(rows)
    wrong_precondition[0]["experiment_precondition_passed"] = False
    assert mod.gate_controls_complete(wrong_precondition, gate_count) is False


def test_scenario_report_7316_sources_are_bounded_and_keep_failures() -> None:
    """SCENARIO-REPORT-7316-SOURCES issues four sequential advisory checks."""

    calls: list[str] = []

    def fetch(url: str) -> dict[str, Any]:
        calls.append(url)
        if "2608.27038" in url:
            raise TimeoutError("bounded timeout")
        return _fetcher(url)

    dispositions, access = mod.collect_source_dispositions(fetch)
    assert calls == [row["url"] for row in mod.ACCESS_SPECS]
    assert len(access) == len(dispositions) == 4
    assert access[1]["timeout_s"] == 20
    assert access[1]["access_status"] == "access_failed"
    assert dispositions[1]["observed_access"] == "access_failed"
    assert all(row["local_evidence_claimed"] is False for row in dispositions)
    assert mod.source_dispositions_complete(dispositions) is True
    assert mod.source_dispositions_complete(dispositions[:-1]) is False


def test_scenario_report_7316_validation_uses_exp7303_scopes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-REPORT-7316-VALIDATION rejects an affected command failure."""

    roadmap_calls: list[tuple[str, ...]] = []
    scoped_calls: list[dict[str, Any]] = []

    def roadmap_runner(_root: Path, commands: Any, **_kwargs: Any) -> list[dict[str, Any]]:
        roadmap_calls.append(tuple(command.name for command in commands))
        return [_receipt(command.name) for command in commands]

    def scoped_runner(_root: Path, **kwargs: Any) -> dict[str, Any]:
        scoped_calls.append(kwargs)
        rows = [_receipt(name) for name in REQUIRED_CHECK_NAMES]
        return {
            "validation_receipts": rows,
            "required_checks_passed": True,
            "repository_health": {"historical_failures": kwargs["historical_failures"]},
        }

    result = mod.run_affected_validation(
        ROOT,
        mod.ACTIVE_ROADMAP_PATH,
        tmp_path,
        historical_failures=[{"name": "old_full_suite", "resolved": False}],
        roadmap_runner=roadmap_runner,
        scoped_runner=scoped_runner,
    )
    assert roadmap_calls == [mod.ROADMAP_CHECK_NAMES]
    assert scoped_calls[0]["test_paths"] == [str(mod.TEST_PATH)]
    assert scoped_calls[0]["basetemp"].is_dir()
    assert scoped_calls[0]["changed_modules"] == [str(mod.MODULE_PATH)]
    assert result["required_checks_passed"] is True

    def failed_runner(_root: Path, commands: Any, **_kwargs: Any) -> list[dict[str, Any]]:
        rows = [_receipt(command.name) for command in commands]
        rows[0] = _receipt(commands[0].name, False)
        return rows

    failed = mod.run_affected_validation(
        ROOT,
        mod.ACTIVE_ROADMAP_PATH,
        tmp_path,
        roadmap_runner=failed_runner,
        scoped_runner=scoped_runner,
    )
    assert failed["required_checks_passed"] is False
    assert failed["failed_required_commands"] == [mod.ROADMAP_CHECK_NAMES[0]]

    terminal_names: list[str] = []

    def command_runner(_root: Path, commands: Any, **kwargs: Any) -> list[dict[str, Any]]:
        terminal_names.extend(command.name for command in commands)
        assert kwargs["log_dir"] == tmp_path / "terminal" / "validation/terminal"
        return [_receipt(command.name) for command in commands]

    monkeypatch.setattr(mod, "run_commands", command_runner)
    terminal = mod.run_terminal_validation(ROOT, tmp_path / "candidate.json", tmp_path / "terminal")
    assert terminal_names == list(mod.TERMINAL_CHECK_NAMES)
    assert [row["name"] for row in terminal] == list(mod.TERMINAL_CHECK_NAMES)
    assert mod._historical_failures(tmp_path) == []


def test_scenario_report_7316_artifact_is_terminal_and_recomputable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-REPORT-7316-ARTIFACT validates the complete stored receipt."""

    terminal_calls: list[Path] = []

    def terminal_runner(_root: Path, candidate: Path, _raw_dir: Path) -> list[dict[str, Any]]:
        terminal_calls.append(candidate)
        assert candidate.is_file()
        return [_receipt(name) for name in mod.TERMINAL_CHECK_NAMES]

    monkeypatch.setattr(mod, "run_terminal_validation", terminal_runner)

    artifact = mod.build_artifact(
        ROOT,
        mod.RUN_DATE,
        output_path=tmp_path / "result.json",
        raw_dir=tmp_path / "raw",
        checkpoint_path=tmp_path / "checkpoint.json",
        fetcher=_fetcher,
        validation_runner=_passing_validation,
    )
    assert artifact["status"] == "complete"
    assert artifact["contract_complete_score"] == 1
    assert artifact["verdict_class"] == "circular_positive"
    assert artifact["MODEL_SPECS"] == []
    assert artifact["model_invoked"] is False
    assert terminal_calls == [tmp_path / "raw" / mod.RAW_CANDIDATE_NAME]
    assert set(artifact["invocation_counts"].values()) == {0}
    assert mod.independent_reduce(artifact) == []
    assert mod.validate_artifact(artifact, root=ROOT) == []
    assert json.loads((tmp_path / "result.json").read_text()) == artifact
    assert json.loads((tmp_path / "checkpoint.json").read_text())["status"] == "running"

    forged = deepcopy(artifact)
    forged["contract_rows"][0]["passed"] = False
    assert "contract_row_reduction" in mod.independent_reduce(forged)
    assert "reproducibility_checksum_invalid" in mod.validate_artifact(forged, root=ROOT)
    assert mod.validate_artifact([], root=ROOT) == ["artifact_mapping_required"]
    missing = deepcopy(artifact)
    del missing["schema"]
    assert mod.validate_artifact(missing, root=ROOT) == ["missing_required_field:schema"]

    validation_mutations = {
        "identity_invalid": ("milestone", "2026.09.000"),
        "lifecycle_invalid": ("status", "running"),
        "model_contract_invalid": ("model_invoked", True),
        "substrate_invalid": ("execution_venue", "remote"),
        "rows_invalid": ("rows", []),
        "field_principles_invalid": ("field_principles", {}),
    }
    for expected_error, (field, value) in validation_mutations.items():
        changed = deepcopy(artifact)
        changed[field] = value
        assert expected_error in mod.validate_artifact(changed, root=ROOT)

    failed_receipt = deepcopy(artifact)
    failed_receipt["validation_receipts"][0]["passed"] = False
    assert "validation_receipts_invalid" in mod.validate_artifact(failed_receipt, root=ROOT)
    wrong_hash = deepcopy(artifact)
    first_source = next(iter(wrong_hash["source_artifact_hashes"]))
    wrong_hash["source_artifact_hashes"][first_source] = "sha256:" + "0" * 64
    assert "source_hash_mismatch" in mod.validate_artifact(wrong_hash, root=ROOT)
    wrong_terminal = deepcopy(artifact)
    wrong_terminal["honest_verdict"] = "complete_changed"
    assert "honest_verdict_invalid" in mod.validate_artifact(wrong_terminal, root=ROOT)

    bad_count = deepcopy(artifact)
    bad_count["contract_rows"] = []
    assert "contract_row_count" in mod.independent_reduce(bad_count)
    bad_gates = deepcopy(artifact)
    bad_gates["gate_control_rows"] = []
    assert "gate_control_reduction" in mod.independent_reduce(bad_gates)
    bad_sources = deepcopy(artifact)
    bad_sources["source_dispositions"] = []
    assert "source_disposition_reduction" in mod.independent_reduce(bad_sources)
    bad_raw = deepcopy(artifact)
    bad_raw["raw_authority_rows"] = []
    assert "raw_authority_reduction" in mod.independent_reduce(bad_raw)
    bad_span = deepcopy(artifact)
    bad_span["phase_spans"] = [{"start_elapsed_s": 2, "end_elapsed_s": 1}]
    assert "phase_span_reduction" in mod.independent_reduce(bad_span)


def test_req_report_7316_terminal_derivation_preserves_exact_failure() -> None:
    """REQ-REPORT-7316 makes blocked and disqualified causes executable."""

    blocked: dict[str, Any] = {
        "preconditions_checked": [
            {
                "upstream": "missing.json",
                "check": "source_bytes",
                "artifact_field": "bytes",
                "expected_value": "present",
                "observed_value": "missing",
                "available": False,
                "blocking": True,
            }
        ],
        "acceptance_gate_results": [],
    }
    mod.apply_terminal_state(blocked)
    assert blocked["verdict_class"] == "blocked"
    assert blocked["contract_complete_score"] == 0
    assert blocked["gate_check_summary"]["observed_value"] == "missing"

    disqualified: dict[str, Any] = {
        "preconditions_checked": [],
        "acceptance_gate_results": [
            {
                "criterion": "contract_exact",
                "expected": True,
                "observed": False,
                "passed": False,
                "principle": "A mismatch cannot pass.",
            }
        ],
    }
    mod.apply_terminal_state(disqualified)
    assert disqualified["verdict_class"] == "disqualified"
    assert disqualified["honest_verdict"].startswith("complete_disqualified")
    assert disqualified["gate_check_summary"]["failed_check"] == "contract_exact"


def test_req_report_7316_date_and_precondition_fail_closed(tmp_path: Path) -> None:
    """REQ-REPORT-7316 rejects a wrong date and missing local authorities."""

    assert mod.date_argument(mod.RUN_DATE) == mod.RUN_DATE
    with pytest.raises(Exception, match="run date must be"):
        mod.date_argument("20260914")

    preconditions, authority, _roadmap, _yaml_bytes = mod._preconditions(
        ROOT,
        Path("/dev/null/result.json"),
        tmp_path / "precondition-raw",
        tmp_path / "precondition-checkpoint.json",
    )
    assert authority == mod.ACTIVE_ROADMAP_PATH
    failed_directory = next(row for row in preconditions if row["check"] == "result_directory")
    assert failed_directory["available"] is False
    assert "FileExistsError" in failed_directory["observed_value"]

    artifact = mod.build_artifact(
        tmp_path,
        mod.RUN_DATE,
        output_path=tmp_path / "result.json",
        raw_dir=tmp_path / "raw",
        checkpoint_path=tmp_path / "checkpoint.json",
        fetcher=_fetcher,
        validation_runner=_passing_validation,
    )
    assert artifact["status"] == "blocked"
    assert artifact["honest_verdict"].startswith("blocked_")
    assert artifact["contract_complete_score"] == 0
    assert artifact["gate_check_summary"]["passed"] is False
