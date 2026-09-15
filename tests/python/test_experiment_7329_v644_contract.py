"""Tests for the V644 advisory contract receipt.

Spec refs: REQ-REPORT-7329 and SCENARIO-REPORT-7329-*.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import re
from typing import Any

import pytest
import yaml

from carnot import experiment_7329_v644_contract as mod
from carnot.reporting.experiment_7303_validation_scope import REQUIRED_CHECK_NAMES


ROOT = Path(__file__).resolve().parents[2]


def _substrate(prompt: str) -> str:
    """Read the explicit class assignment used by the selected roadmap."""

    match = re.search(r"inference_substrate_class\s*=\s*([a-z_]+)", prompt)
    assert match is not None
    return match.group(1)


def _markdown(roadmap: dict[str, Any]) -> str:
    """Build an independent Markdown authority from visible YAML values."""

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
            f"{task['phase']} | {_substrate(task['prompt'])} | {gate_text} |"
        )
    return "\n".join(
        [
            "# V644 test contract",
            "",
            "**Milestone:** 2026.09.644",
            "",
            "## Exact Task Contract",
            "",
            "| Order | Task ID | Exact title | Deliverable | Phase | Substrate class | Structured gate |",
            "|---|---|---|---|---|---|---|",
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
            "status": "healthy",
            "incident_open": False,
            "historical_failures": list(kwargs["historical_failures"]),
            "historical_failure_count": len(kwargs["historical_failures"]),
            "unresolved_collection_error_observation_count": 0,
            "affects_required_checks": False,
        },
    }


def _fetcher(url: str) -> dict[str, Any]:
    """Return a named primary page without network access."""

    paper_id = next(
        key for key in ("2604.13283", "2607.20792", "2607.17047", "2607.29549") if key in url
    )
    return {"ok": True, "status_code": 200, "body": f"primary page {paper_id}", "error": None}


def test_scenario_report_7329_authority_prefers_matching_staged_then_active(
    tmp_path: Path,
) -> None:
    """SCENARIO-REPORT-7329-AUTHORITY resolves one matching V644 source."""

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


def test_scenario_report_7329_contract_rejects_every_named_mutation(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7329-CONTRACT checks all literals and closed operators."""

    roadmap = yaml.safe_load((ROOT / mod.ACTIVE_ROADMAP_PATH).read_text())
    markdown = _markdown(roadmap)
    result = mod.evaluate_contract(markdown, roadmap)
    assert result["passed"] is True
    assert [row["unit_id"] for row in result["contract_rows"]] == list(mod.EXPECTED_ID_ORDER)
    assert all(row["passed"] and row["censored"] is False for row in result["contract_rows"])
    assert all(row["checks"]["gate_fields_declared"] for row in result["contract_rows"])

    mutations = mod.run_contract_mutations(markdown, roadmap, tmp_path)
    assert [row["mutation"] for row in mutations] == [
        "missing_task",
        "reordered_id",
        "stale_milestone",
        "changed_phase",
        "changed_path",
        "changed_title",
        "changed_field",
        "malformed_operator",
    ]
    assert all(row["rejected"] is True for row in mutations)

    malformed_list = markdown.replace(
        'verdict_class in ["positive", "circular_positive", "null"]',
        'verdict_class in "positive"',
        1,
    )
    with pytest.raises(ValueError, match="JSON list"):
        mod.parse_markdown_contract(malformed_list)
    malformed_op = markdown.replace(
        "executor_fixture_ready_score == 1", "executor_fixture_ready_score contains 1", 1
    )
    with pytest.raises(ValueError, match="malformed Markdown structured gate"):
        mod.parse_markdown_contract(malformed_op)


def test_req_report_7329_parsers_reject_malformed_shapes() -> None:
    """REQ-REPORT-7329 covers malformed parser inputs, not only value drift."""

    roadmap = yaml.safe_load((ROOT / mod.ACTIVE_ROADMAP_PATH).read_text())
    markdown = _markdown(roadmap)
    assert mod._required_block("no declaration") == ""
    assert mod._field_declared("REQUIRED ARTIFACT FIELDS:\n- value: x", None) is False
    assert mod._substrate_class("no class assignment") is None
    with pytest.raises(ValueError, match="JSON list"):
        mod._parse_gate_value("in", "[")
    with pytest.raises(ValueError, match="terminal classes"):
        mod._parse_gate_value("in", '["blocked", "invented"]')
    with pytest.raises(ValueError, match="requires a scalar"):
        mod._parse_gate_value("==", "[1]")
    with pytest.raises(ValueError, match="milestone is missing"):
        mod.parse_markdown_contract(markdown.replace("**Milestone:**", "Milestone:"))
    with pytest.raises(ValueError, match="section is missing"):
        mod.parse_markdown_contract(markdown.replace("## Exact Task Contract", "## Changed"))
    with pytest.raises(ValueError, match="malformed Markdown task row"):
        mod.parse_markdown_contract(markdown.replace("| 1 | exp7329", "| bad | exp7329", 1))
    with pytest.raises(ValueError, match="malformed Markdown task row"):
        mod.parse_markdown_contract(markdown.replace("| exp7329-contract |", "| bad |", 1))
    empty_table = re.sub(
        r"\| 1 \|[\s\S]*?(?=\n## End)",
        "",
        markdown,
    )
    with pytest.raises(ValueError, match="table is missing"):
        mod.parse_markdown_contract(empty_table)

    with pytest.raises(ValueError, match="mapping with tasks"):
        mod.parse_yaml_contract([])
    malformed = deepcopy(roadmap)
    malformed["tasks"][0] = "not-a-mapping"
    with pytest.raises(ValueError, match="prompt mapping"):
        mod.parse_yaml_contract(malformed)
    malformed = deepcopy(roadmap)
    malformed["tasks"][2]["gated_on"] = "not-a-list"
    with pytest.raises(ValueError, match="malformed gates"):
        mod.parse_yaml_contract(malformed)
    malformed = deepcopy(roadmap)
    malformed["tasks"][2]["gated_on"][1]["value"] = ["invented"]
    with pytest.raises(ValueError, match="terminal class list"):
        mod.parse_yaml_contract(malformed)
    assert mod.validate_gate_declarations({"tasks": ["not-a-mapping"]})["passed"] is False


def test_scenario_report_7329_fields_reject_undeclared_or_receipt_gates() -> None:
    """SCENARIO-REPORT-7329-FIELDS binds gates to producer declarations."""

    roadmap = yaml.safe_load((ROOT / mod.ACTIVE_ROADMAP_PATH).read_text())
    checks = mod.validate_gate_declarations(roadmap)
    assert checks["passed"] is True
    assert checks["receipt_is_not_upstream"] is True

    changed = deepcopy(roadmap)
    changed["tasks"][2]["gated_on"][0]["artifact_field"] = "invented_ready_score"
    assert mod.validate_gate_declarations(changed)["passed"] is False
    changed = deepcopy(roadmap)
    changed["tasks"][2]["gated_on"][0]["upstream"] = mod.FIRST_TASK_ID
    assert mod.validate_gate_declarations(changed)["passed"] is False


def test_scenario_report_7329_gate_controls_use_real_reader(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7329-GATES makes all adverse combined cases fail."""

    roadmap = yaml.safe_load((ROOT / mod.ACTIVE_ROADMAP_PATH).read_text())
    rows = mod.run_gate_controls(roadmap, tmp_path)
    gate_count = sum(len(task.get("gated_on") or []) for task in roadmap["tasks"])
    assert len(rows) == gate_count * 6
    assert mod.gate_controls_complete(rows, gate_count) is True
    assert {row["case"] for row in rows} == {
        "passing",
        "zero_score",
        "missing_file",
        "missing_field",
        "missing_verdict_class",
        "disqualified_score_one",
    }
    assert all(row["combined_conductor_passed"] is (row["case"] == "passing") for row in rows)
    assert all(row["experiment_precondition_passed"] is (row["case"] == "passing") for row in rows)

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
    assert mod.upstream_precondition(None, "ready_score", 1)["passed"] is False
    assert (
        mod.upstream_precondition(
            {"status": "complete", "ready_score": 1, "flagged_adversarial": True},
            "ready_score",
            1,
        )["reason"]
        == "quarantined_upstream_rejected_before_field_consumption"
    )
    assert (
        mod.upstream_precondition({"status": "blocked", "ready_score": 1}, "ready_score", 1)[
            "reason"
        ]
        == "blocked_upstream_rejected_before_field_consumption"
    )
    assert (
        mod.upstream_precondition({"status": "partial", "ready_score": 1}, "ready_score", 1)[
            "reason"
        ]
        == "partial_upstream_rejected_before_field_consumption"
    )
    with pytest.raises(ValueError, match="malformed upstream"):
        mod._write_gate_artifacts(tmp_path / "malformed", {"bad-id": {}})


def test_scenario_report_7329_sources_are_bounded_and_keep_failures() -> None:
    """SCENARIO-REPORT-7329-SOURCES retains access limits without blocking."""

    calls: list[str] = []

    def fetch(url: str) -> dict[str, Any]:
        calls.append(url)
        if "2607.20792" in url:
            return {"ok": False, "status_code": 429, "body": "challenge", "error": "HTTP 429"}
        if "2607.29549" in url:
            raise TimeoutError("bounded timeout")
        if "2607.17047" in url:
            return {"ok": False, "status_code": 403, "body": "browser challenge", "error": None}
        return _fetcher(url)

    dispositions, access = mod.collect_source_dispositions(fetch)
    assert calls == [row["url"] for row in mod.ACCESS_SPECS]
    assert len(access) == len(dispositions) == 4
    assert access[1]["access_outcome"] == "http_429"
    assert access[2]["access_outcome"] == "browser_challenge"
    assert access[3]["access_outcome"] == "access_failed"
    assert all(row["local_evidence_claimed"] is False for row in dispositions)
    assert mod.source_dispositions_complete(dispositions) is True
    assert mod.source_dispositions_complete(dispositions[:-1]) is False


def test_scenario_report_7329_validation_uses_exp7303_scopes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-REPORT-7329 uses exact roadmap and changed-file validation scopes."""

    roadmap_calls: list[tuple[str, ...]] = []
    scoped_calls: list[dict[str, Any]] = []

    def roadmap_runner(_root: Path, commands: Any, **_kwargs: Any) -> list[dict[str, Any]]:
        roadmap_calls.append(tuple(command.name for command in commands))
        return [_receipt(command.name) for command in commands]

    def scoped_runner(_root: Path, **kwargs: Any) -> dict[str, Any]:
        scoped_calls.append(kwargs)
        return {
            "validation_receipts": [_receipt(name) for name in REQUIRED_CHECK_NAMES],
            "required_checks_passed": True,
            "repository_health": {"historical_failures": kwargs["historical_failures"]},
        }

    result = mod.run_affected_validation(
        ROOT,
        mod.ACTIVE_ROADMAP_PATH,
        tmp_path,
        historical_failures=[{"name": "old", "resolved": False}],
        roadmap_runner=roadmap_runner,
        scoped_runner=scoped_runner,
    )
    assert roadmap_calls == [mod.ROADMAP_CHECK_NAMES]
    assert scoped_calls[0]["test_paths"] == [str(mod.TEST_PATH)]
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

    terminal_names: list[str] = []

    def command_runner(_root: Path, commands: Any, **_kwargs: Any) -> list[dict[str, Any]]:
        terminal_names.extend(command.name for command in commands)
        return [_receipt(command.name) for command in commands]

    monkeypatch.setattr(mod, "run_commands", command_runner)
    terminal = mod.run_terminal_validation(ROOT, tmp_path / "candidate.json", tmp_path)
    assert terminal_names == list(mod.TERMINAL_CHECK_NAMES)
    assert [row["name"] for row in terminal] == list(mod.TERMINAL_CHECK_NAMES)
    assert mod._historical_failures(tmp_path) == []


def test_scenario_report_7329_artifact_is_terminal_and_recomputable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-REPORT-7329-ARTIFACT validates the stored terminal receipt."""

    def terminal_runner(_root: Path, candidate: Path, _raw_dir: Path) -> list[dict[str, Any]]:
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
    assert set(artifact["invocation_counts"].values()) == {0}
    assert mod.independent_reduce(artifact) == []
    assert mod.validate_artifact(artifact, root=ROOT) == []
    assert json.loads((tmp_path / "result.json").read_text()) == artifact
    assert json.loads((tmp_path / "checkpoint.json").read_text())["status"] == "running"

    mutation_errors = {
        "contract_row_reduction": ("contract_rows", []),
        "gate_control_reduction": ("gate_control_rows", []),
        "source_disposition_reduction": ("source_dispositions", []),
        "raw_authority_reduction": ("raw_authority_rows", []),
        "phase_span_reduction": ("phase_spans", [{"start_elapsed_s": 2, "end_elapsed_s": 1}]),
    }
    for error, (field, value) in mutation_errors.items():
        changed = deepcopy(artifact)
        changed[field] = value
        assert error in mod.independent_reduce(changed)
    changed = deepcopy(artifact)
    changed["contract_rows"][0]["passed"] = False
    assert "contract_row_reduction" in mod.independent_reduce(changed)

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
        "reproducibility_checksum_invalid": ("duration_s", -1),
    }
    for error, (field, value) in validation_mutations.items():
        changed = deepcopy(artifact)
        changed[field] = value
        assert error in mod.validate_artifact(changed, root=ROOT)

    failed_receipt = deepcopy(artifact)
    failed_receipt["validation_receipts"].pop()
    assert "validation_receipts_invalid" in mod.validate_artifact(failed_receipt, root=ROOT)
    wrong_hash = deepcopy(artifact)
    first_source = next(iter(wrong_hash["source_artifact_hashes"]))
    wrong_hash["source_artifact_hashes"][first_source] = "sha256:" + "0" * 64
    assert "source_hash_mismatch" in mod.validate_artifact(wrong_hash, root=ROOT)
    wrong_terminal = deepcopy(artifact)
    wrong_terminal["honest_verdict"] = "complete_changed"
    assert "honest_verdict_invalid" in mod.validate_artifact(wrong_terminal, root=ROOT)


def test_req_report_7329_terminal_derivation_and_preconditions(tmp_path: Path) -> None:
    """REQ-REPORT-7329 preserves exact block and disqualification causes."""

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

    assert mod.date_argument(mod.RUN_DATE) == mod.RUN_DATE
    with pytest.raises(Exception, match="run date must be"):
        mod.date_argument("20260914")

    preconditions, authority, _roadmap, _bytes = mod._preconditions(
        ROOT, Path("/dev/null/result.json"), tmp_path / "raw", tmp_path / "checkpoint.json"
    )
    assert authority == mod.ACTIVE_ROADMAP_PATH
    failed = next(row for row in preconditions if row["check"] == "result_directory")
    assert failed["available"] is False

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
    assert artifact["contract_complete_score"] == 0
