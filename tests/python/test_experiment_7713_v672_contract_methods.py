"""V672 contract checks. REQ-REPORT-7713; REQ-HARNESS-7713."""

from copy import deepcopy
import json
from pathlib import Path

import pytest
import yaml

from carnot import experiment_7713_v672_contract_methods as subject
from carnot.reporting import v672_contract as contract


ROOT = Path(__file__).resolve().parents[2]
ROADMAP = yaml.safe_load((ROOT / "research-roadmap.yaml").read_text())


def complete_design() -> str:
    """Give the parser independent representations with exact field values."""

    tasks = [
        {key: task.get(key, [] if key == "gated_on" else None) for key in contract.FIELDS}
        for task in ROADMAP["tasks"]
    ]
    table = [
        "| Order | ID | Phase | Title | Deliverable | Substrate | Models | Gates |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for index, task in enumerate(tasks, 1):
        table.append(
            f"| {index} | `{task['id']}` | {task['phase']} | {task['title']} | "
            f"`{task['deliverable']}` | `{task['inference_substrate_class']}` | "
            f"`{json.dumps(task['MODEL_SPECS'])}` | `{json.dumps(task['gated_on'])}` |"
        )
    return (
        "## Exact Task Contract\n"
        + "\n".join(table)
        + "\n```json\n"
        + json.dumps({"milestone": contract.MILESTONE, "tasks": tasks})
        + "\n```\n"
    )


def test_complete_contract_and_mutations() -> None:
    """SCENARIO-REPORT-7713-CONTRACT: all ordered fields bind."""

    design = complete_design()
    result = contract.compare_authorities(design, ROADMAP)
    assert result["passed"] and len(result["rows"]) == 13
    assert all(row["matched"] for row in result["rows"])
    for mutation in ("delete", "reorder", "stale", "producer_field", "model"):
        changed = contract.mutate_authority(ROADMAP, mutation)
        assert not contract.compare_authorities(design, changed)["passed"], mutation


def test_missing_design_is_exact_block() -> None:
    """SCENARIO-REPORT-7713-CONTRACT: absent sections never pass."""

    design = (ROOT / contract.DESIGN_PATH).read_text()
    result = contract.compare_authorities(design, ROADMAP)
    assert not result["passed"]
    assert "design_table_missing" in result["errors"]
    assert "design_json_missing" in result["errors"]
    assert len(result["rows"]) == 13
    assert all(not row["matched"] for row in result["rows"])
    assert contract.classify(result, True, True) == (
        "complete_blocked_v672_independent_contract",
        "blocked",
    )


def test_authority_resolution(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7713-CONTRACT: matching staged bytes win."""

    (tmp_path / "research-roadmap.yaml").write_text(yaml.safe_dump(ROADMAP))
    path, _, candidates = contract.resolve_authority(tmp_path)
    assert path.name == "research-roadmap.yaml"
    assert not candidates[0]["exists"]
    (tmp_path / "research-roadmap-next.yaml").write_text(yaml.safe_dump(ROADMAP))
    path, _, _ = contract.resolve_authority(tmp_path)
    assert path.name == "research-roadmap-next.yaml"


def test_artifact_and_cold_reader(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7713-TERMINAL: missing design blocks readiness."""

    value = subject.build_artifact_for_test(ROOT)
    assert value["verdict_class"] == "blocked"
    assert value["honest_verdict"].startswith("complete_blocked_")
    assert value["contract_ready_score"] == 0
    assert value["MODEL_SPECS"] == value["planned_MODEL_SPECS"] == []
    assert value["model_invoked"] is False
    assert len(value["rows"]) == 13
    assert any(row["field"] == "independent_table" for row in value["gate_check_summary"])
    assert subject.validate_artifact(value, ROOT)
    candidate = tmp_path / "candidate.json"
    candidate.write_text(json.dumps(value))
    assert subject.cold_read(candidate, ROOT)
    changed = deepcopy(value)
    changed["contract_ready_score"] = 1
    assert not subject.validate_artifact(changed, ROOT)
    candidate.write_text("{}")
    assert not subject.cold_read(candidate, ROOT)


def test_guards_and_plans(tmp_path: Path) -> None:
    """SCENARIO-HARNESS-7713-RECEIPTS: frozen checks name real files."""

    with pytest.raises(ValueError, match="run date"):
        subject.run_experiment(ROOT, "20260925", tmp_path / "bad.json")
    path, _, _ = contract.resolve_authority(ROOT)
    names = {row.name for row in subject.validation_plan(ROOT, path, tmp_path)}
    assert {"focused_pytest", "changed_module_coverage_report", "prompt_path"} <= names
    assert (tmp_path / "basetemp").is_dir()
    assert (tmp_path / "coverage").is_dir()
    terminal = subject.terminal_plan(ROOT, tmp_path / "candidate.json")
    assert {row.name for row in terminal} >= {
        "cold_replay",
        "independent_reduction",
        "adversarial_verify",
        "verdict_row_consistency_strict",
    }


@pytest.mark.parametrize(
    "field",
    [
        "selected_roadmap_path",
        "rows",
        "contract_comparison",
        "contract_ready_score",
        "eligible",
        "gate_check_summary",
        "planned_output_hash",
        "input_exists",
        "input_hash",
    ],
)
def test_cold_reader_rejects_mutated_evidence(field: str) -> None:
    """SCENARIO-REPORT-7713-TERMINAL: changing any bound operand fails."""

    value = subject.build_artifact_for_test(ROOT)
    if field == "selected_roadmap_path":
        value[field] = "missing.yaml"
    elif field == "rows":
        value[field][0]["matched"] = True
    elif field == "contract_comparison":
        value[field]["passed"] = True
    elif field == "contract_ready_score":
        value[field] = 1
    elif field == "eligible":
        value["sample_size_budget"][field] = 1
    elif field == "gate_check_summary":
        value[field] = []
    else:
        rows = value["source_artifact_hashes"]
        if field == "planned_output_hash":
            next(row for row in rows if row["role"] == "planned_output_not_input")["sha256"] = "bad"
        elif field == "input_exists":
            next(row for row in rows if row["role"] == "current_input")["exists"] = False
        else:
            next(row for row in rows if row["role"] == "current_input")["sha256"] = "bad"
    value["reproducibility_checksum"] = subject.checksum(value)
    assert not subject.validate_artifact(value, ROOT)


def test_parser_and_classifier_error_paths(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7713-CONTRACT: malformed independent bytes fail."""

    design = complete_design()
    broken = design.replace("`[]`", "`invalid`", 1)
    assert "design_table_invalid" in contract.parse_design(broken)[3]
    broken = design.replace('"tasks":', '"missing":', 1)
    assert "design_json_invalid" in contract.parse_design(broken)[3]
    with pytest.raises(ValueError, match="unknown mutation"):
        contract.mutate_authority(ROADMAP, "unknown")
    with pytest.raises(ValueError, match="matching roadmap"):
        contract.resolve_authority(tmp_path)
    good = contract.compare_authorities(design, ROADMAP)
    assert contract.classify(good, True, False)[1] == "disqualified"
    assert contract.classify(good, True, True)[1] == "null"


def test_main_read_modes_and_span(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """SCENARIO-HARNESS-7713-RECEIPTS: fresh reader modes use disk bytes."""

    candidate = tmp_path / "candidate.json"
    candidate.write_text(json.dumps(subject.build_artifact_for_test(ROOT)))
    assert subject.main(["--cold-validate", str(candidate)]) == 0
    assert subject.main(["--independent-reduce", str(candidate)]) == 0
    candidate.write_text("broken")
    assert subject.main(["--cold-validate", str(candidate)]) == 1
    assert subject.main(["--independent-reduce", str(candidate)]) == 1
    span = subject.span("unit", 1.0, 2.0, 3)
    assert span["completed_units"] == 3
    calls: list[tuple] = []
    monkeypatch.setattr(subject, "run_experiment", lambda *args: calls.append(args))
    assert subject.main([]) == 0
    assert calls == [(subject.ROOT, subject.RUN_DATE, subject.RESULT_PATH)]


def test_failed_owned_check_disqualifies_even_with_missing_design() -> None:
    """SCENARIO-HARNESS-7713-RECEIPTS: failed work never opens readiness."""

    authority, roadmap, candidates = contract.resolve_authority(ROOT)
    comparison = contract.compare_authorities((ROOT / contract.DESIGN_PATH).read_text(), roadmap)
    value = subject.build_artifact(
        ROOT,
        authority,
        candidates,
        comparison,
        [{"name": "focused_pytest", "passed": False}],
        0.0,
        [],
    )
    assert value["verdict_class"] == "disqualified"
    assert value["contract_ready_score"] == 0
