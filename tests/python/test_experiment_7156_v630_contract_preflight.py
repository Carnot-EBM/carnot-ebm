"""Focused tests for REQ-REPORT-7156 and its V630 contract scenarios."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

import pytest
import yaml

from carnot import experiment_7156_v630_contract_preflight as mod


ROOT = Path(__file__).resolve().parents[2]

# The Markdown and YAML fixtures repeat values on purpose. Separate source
# values ensure that one parser cannot make both sides agree by construction.
MARKDOWN_ROWS = (
    (
        "exp7156-v630-exact-contract-preflight",
        "V630 exact contract preflight",
        "results/experiment_7156_v630_contract_preflight.json",
    ),
    (
        "exp7157-v630-qwen38-registry-runtime",
        "Qwen3.8 registry runtime",
        "results/experiment_7157_v630_qwen38_runtime.json",
    ),
    (
        "exp7158-v630-entity-evidence-fixture",
        "Entity evidence fixture",
        "results/experiment_7158_v630_entity_evidence_fixture.json",
    ),
    (
        "exp7159-v630-evidence-generation",
        "Evidence generation",
        "results/experiment_7159_v630_evidence_generation.json",
    ),
    (
        "exp7160-v630-verifier-pilot",
        "Verifier pilot",
        "results/experiment_7160_v630_verifier_pilot.json",
    ),
    (
        "exp7161-v630-independent-audit",
        "Independent verifier audit",
        "results/experiment_7161_v630_independent_audit.json",
    ),
    (
        "exp7162-v630-noise-fixture",
        "Noise fixture",
        "results/experiment_7162_v630_noise_fixture.json",
    ),
    ("exp7163-v630-memory-pilot", "Memory pilot", "results/experiment_7163_v630_memory_pilot.json"),
    ("exp7164-v630-memory-audit", "Memory audit", "results/experiment_7164_v630_memory_audit.json"),
    (
        "exp7165-v630-arc-feasibility",
        "ARC feasibility",
        "results/experiment_7165_v630_arc_feasibility.json",
    ),
    (
        "exp7166-v630-fixed-magnetization-parity",
        "Fixed magnetization parity",
        "results/experiment_7166_v630_fixed_magnetization_parity.json",
    ),
    (
        "exp7167-v630-hardware-continuity",
        "Hardware continuity",
        "results/experiment_7167_v630_hardware_continuity.json",
    ),
    (
        "exp7168-v630-independent-capstone",
        "V630 independent capstone",
        "results/experiment_7168_v630_capstone.json",
    ),
)

YAML_ROWS = (
    (
        "exp7156-v630-exact-contract-preflight",
        "V630 exact contract preflight",
        "results/experiment_7156_v630_contract_preflight.json",
    ),
    (
        "exp7157-v630-qwen38-registry-runtime",
        "Qwen3.8 registry runtime",
        "results/experiment_7157_v630_qwen38_runtime.json",
    ),
    (
        "exp7158-v630-entity-evidence-fixture",
        "Entity evidence fixture",
        "results/experiment_7158_v630_entity_evidence_fixture.json",
    ),
    (
        "exp7159-v630-evidence-generation",
        "Evidence generation",
        "results/experiment_7159_v630_evidence_generation.json",
    ),
    (
        "exp7160-v630-verifier-pilot",
        "Verifier pilot",
        "results/experiment_7160_v630_verifier_pilot.json",
    ),
    (
        "exp7161-v630-independent-audit",
        "Independent verifier audit",
        "results/experiment_7161_v630_independent_audit.json",
    ),
    (
        "exp7162-v630-noise-fixture",
        "Noise fixture",
        "results/experiment_7162_v630_noise_fixture.json",
    ),
    ("exp7163-v630-memory-pilot", "Memory pilot", "results/experiment_7163_v630_memory_pilot.json"),
    ("exp7164-v630-memory-audit", "Memory audit", "results/experiment_7164_v630_memory_audit.json"),
    (
        "exp7165-v630-arc-feasibility",
        "ARC feasibility",
        "results/experiment_7165_v630_arc_feasibility.json",
    ),
    (
        "exp7166-v630-fixed-magnetization-parity",
        "Fixed magnetization parity",
        "results/experiment_7166_v630_fixed_magnetization_parity.json",
    ),
    (
        "exp7167-v630-hardware-continuity",
        "Hardware continuity",
        "results/experiment_7167_v630_hardware_continuity.json",
    ),
    (
        "exp7168-v630-independent-capstone",
        "V630 independent capstone",
        "results/experiment_7168_v630_capstone.json",
    ),
)


def _markdown() -> str:
    """Build a Markdown source without consulting the YAML task objects."""

    rows = []
    for order, (task_id, title, deliverable) in enumerate(MARKDOWN_ROWS, 1):
        gate = "`exp7158.fixture_ready_score == 1`" if order == 4 else "none"
        rows.append(f"| {order} | `{task_id}` | {title} | `{deliverable}` | {gate} |")
    return "\n".join(
        (
            "# V630 fixture",
            "",
            "**Milestone:** `2026.09.630`",
            "",
            "## Exact task contract",
            "",
            "| Order | Task ID | Exact title | Deliverable | Structured gate |",
            "|---:|---|---|---|---|",
            *rows,
        )
    )


def _prompt(number: int, *, model_task: bool = False, producer: bool = False) -> str:
    """Create a prompt that carries every REQ-REPORT-7156 duty."""

    model_line = (
        '- MODEL_SPECS: principle: "Use unsloth/Qwen3.8-27B-GGUF for the headline model."\n'
        if model_task
        else ""
    )
    producer_line = (
        '- fixture_ready_score: principle: "This field gates evidence generation."\n'
        if producer
        else ""
    )
    substrate = (
        "model_bounded_generation"
        if model_task
        else "cpu_exact_solver_or_simulator"
        if number == 7158
        else "aggregation"
    )
    return (
        "Work at {project_root}. The execution date is {date}.\n"
        "CONCRETE STEPS:\n"
        "0. Print and flush a phase-start line.\n"
        "1. At every numbered phase boundary, print a flushed progress line. Print flushed start "
        "and end lines immediately before and after every model load, generation, benchmark, or "
        "subprocess that can take minutes. Print a flushed heartbeat at least every 300 seconds "
        "inside long loops. Keep every stdout gap below 600 seconds.\n"
        "REQUIRED ARTIFACT FIELDS:\n"
        '- field_principles: principle: "Explain every field."\n'
        '- status: principle: "Record a terminal state."\n'
        '- preconditions_checked: principle: "Record inputs."\n'
        '- run_date: principle: "Record the date."\n'
        '- inference_substrate: principle: "Record the evidence source."\n'
        f'- inference_substrate_class: principle: "Use {substrate}, or blocked_no_run."\n'
        '- execution_venue: principle: "Record the host."\n'
        '- duration_s: principle: "Record wall time."\n'
        '- source_artifact_hashes: principle: "Bind inputs."\n'
        '- rows: principle: "Preserve every unit."\n'
        + model_line
        + producer_line
        + '- random_seed: principle: "Make ordering stable."\n'
        '- reproducibility_checksum: principle: "Detect drift."\n'
        '- gate_check_summary: principle: "Name failed check, expected value, and observed value."\n'
        '- verifier_is_oracle: principle: "Use false."\n'
        '- verdict_class: principle: "Use the closed enum positive | circular_positive | null | blocked | disqualified | partial."\n'
        '- honest_verdict: principle: "Record the terminal reason."\n'
        f"Run command: cd {{project_root}} && .venv/bin/python scripts/experiments/experiment_{number}_v630_fixture.py --date {{date}}\n"
        + mod.PROMPT_FINAL_LINE
    )


def _roadmap() -> dict[str, Any]:
    """Build YAML objects from the independent YAML row declaration."""

    tasks: list[dict[str, Any]] = []
    for order, (task_id, title, deliverable) in enumerate(YAML_ROWS, 1):
        number = 7155 + order
        task: dict[str, Any] = {
            "id": task_id,
            "title": title,
            "track": "infrastructure",
            "priority": "high",
            "requires_gpu": number == 7157,
            "max_turns": 50,
            "estimated_wall_time_min": 30,
            "per_unit_rows": True,
            "milestone": "2026.09.630",
            "deliverable": deliverable,
            "prior_failures": [],
            "prompt": _prompt(number, model_task=number == 7157, producer=number == 7158),
        }
        if number == 7156:
            task["model"] = "opus"
            task["prior_failures"] = [
                {
                    "experiment_id": "exp7151-v629-active-contract-preflight",
                    "verdict": "complete_disqualified_v629_markdown_yaml_contract_mismatch",
                    "addressed_by": "V630 checks the active files without repairing them.",
                    "retire_if_same_verdict": True,
                }
            ]
        elif number % 2:
            task["agent_type"] = "codex"
            task["model"] = "gpt-5.6-sol"
        if number == 7159:
            task["gated_on"] = [
                {
                    "upstream": YAML_ROWS[2][0],
                    "artifact_field": "fixture_ready_score",
                    "op": "==",
                    "value": 1,
                }
            ]
        tasks.append(task)
    return {
        "milestone": "2026.09.630",
        "milestone_title": "fixture",
        "milestone_doc": "fixture.md",
        "tasks": tasks,
    }


def _write_inputs(root: Path, roadmap: dict[str, Any] | None = None) -> Path:
    """Write isolated sources for artifact lifecycle tests."""

    required = {
        mod.DESIGN_PATH: _markdown(),
        mod.ACTIVE_ROADMAP_PATH: yaml.safe_dump(roadmap or _roadmap(), sort_keys=False),
        mod.EXCLUSION_PATH: "retired: []\n",
        mod.PRIOR_ARTIFACT_PATH: "{}\n",
        mod.SPEC_PATH: "REQ-REPORT-7156\n",
        mod.ROADMAP_SCHEMA_PATH: "# schema\n",
        mod.PRIOR_LINT_PATH: "# lint\n",
        mod.GATE_LINT_PATH: "# lint\n",
        mod.EXCLUSION_LINT_PATH: "# lint\n",
        mod.HARNESS_LINT_PATH: "# lint\n",
        mod.ADVERSARIAL_PATH: "# lint\n",
        mod.ROW_LINT_PATH: "# lint\n",
        mod.SPEC_COVERAGE_PATH: "# lint\n",
        mod.ROOT_CLUTTER_PATH: "# lint\n",
        mod.MODULE_PATH: "# module\n",
        mod.WRAPPER_PATH: "# wrapper\n",
        mod.TEST_PATH: "# test\n",
    }
    for relative, content in required.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    return root / mod.DEFAULT_OUTPUT_PATH


def test_req_report_7156_spec_defines_fields_and_scenarios() -> None:
    """REQ-REPORT-7156 names the result fields and focused scenarios."""

    text = (ROOT / mod.SPEC_PATH).read_text(encoding="utf-8")
    section = text[text.index("### REQ-REPORT-7156") :]
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert field in section
    for name in ("PARITY", "GATES", "DISCIPLINE", "PREFLIGHT", "ACTIVE", "COMMANDS", "ARTIFACT"):
        assert f"SCENARIO-REPORT-7156-{name}" in section


def test_scenario_report_7156_parity_uses_independent_rows() -> None:
    """SCENARIO-REPORT-7156-PARITY accepts separate matching sources."""

    result = mod.evaluate_contract(_markdown(), _roadmap(), set())
    assert result["passed"] is True
    assert len(result["task_contract_rows"]) == 13
    assert result["expected_id_order"] == [row[0] for row in MARKDOWN_ROWS]
    assert result["markdown_task_rows"] is not result["yaml_task_rows"]
    assert result["gate_contract_rows"][3]["expected"][0]["upstream"] == MARKDOWN_ROWS[2][0]


@pytest.mark.parametrize(
    "case",
    ("missing", "extra", "reordered", "duplicated", "renamed", "title", "deliverable", "milestone"),
)
def test_req_report_7156_row_mutations_fail(case: str) -> None:
    """REQ-REPORT-7156 rejects count, identity, order, and content drift."""

    roadmap = _roadmap()
    if case == "missing":
        roadmap["tasks"].pop(5)
    elif case == "extra":
        roadmap["tasks"].append(deepcopy(roadmap["tasks"][-1]))
    elif case == "reordered":
        roadmap["tasks"][5], roadmap["tasks"][6] = roadmap["tasks"][6], roadmap["tasks"][5]
    elif case == "duplicated":
        roadmap["tasks"][5] = deepcopy(roadmap["tasks"][4])
    elif case == "renamed":
        roadmap["tasks"][5]["id"] = "exp7161-v630-renamed"
    elif case == "title":
        roadmap["tasks"][0]["title"] += " changed"
    elif case == "deliverable":
        roadmap["tasks"][0]["deliverable"] = "results/wrong.json"
    else:
        roadmap["milestone"] = "2026.09.631"
    assert mod.evaluate_contract(_markdown(), roadmap, set())["passed"] is False


def test_scenario_report_7156_gates_reject_bad_producers() -> None:
    """SCENARIO-REPORT-7156-GATES rejects missing, later, or nested fields."""

    mutations: tuple[Callable[[dict[str, Any]], None], ...] = (
        lambda value: value["tasks"][3]["gated_on"][0].__setitem__(
            "artifact_field", "nested.value"
        ),
        lambda value: value["tasks"][3]["gated_on"][0].__setitem__(
            "artifact_field", "missing_field"
        ),
        lambda value: value["tasks"][3]["gated_on"][0].__setitem__("upstream", YAML_ROWS[-1][0]),
        lambda value: value["tasks"][3]["gated_on"][0].__setitem__("op", "!="),
    )
    for mutate in mutations:
        roadmap = _roadmap()
        mutate(roadmap)
        assert mod.evaluate_contract(_markdown(), roadmap, set())["passed"] is False


@pytest.mark.parametrize(
    "mutate",
    (
        lambda value: value["tasks"][0]["prior_failures"][0].__setitem__("verdict", ""),
        lambda value: value["tasks"][0]["prior_failures"][0].__setitem__(
            "retire_if_same_verdict", False
        ),
        lambda value: value["tasks"][1].__setitem__("agent_type", "gemini"),
        lambda value: value["tasks"][1].__setitem__("per_unit_rows", False),
        lambda value: value["tasks"][1].__setitem__(
            "prompt", value["tasks"][1]["prompt"].replace("Qwen3.8-27B", "Qwen3.6-35B")
        ),
        lambda value: value["tasks"][1].__setitem__(
            "prompt",
            value["tasks"][1]["prompt"].replace("at least every 300 seconds", "periodically"),
        ),
        lambda value: value["tasks"][1].__setitem__(
            "prompt", value["tasks"][1]["prompt"].replace("{project_root}", "/tmp/project")
        ),
        lambda value: value["tasks"][1].__setitem__(
            "prompt", value["tasks"][1]["prompt"].replace(mod.PROMPT_FINAL_LINE, "Do not push.")
        ),
        lambda value: value["tasks"][1].__setitem__(
            "prompt",
            value["tasks"][1]["prompt"].replace("model_bounded_generation", "invented_class"),
        ),
        lambda value: value["tasks"][2].__setitem__("operator_override", "two\nlines"),
    ),
)
def test_scenario_report_7156_discipline_mutations_fail(
    mutate: Callable[[dict[str, Any]], None],
) -> None:
    """SCENARIO-REPORT-7156-DISCIPLINE rejects metadata and prompt drift."""

    roadmap = _roadmap()
    mutate(roadmap)
    assert mod.evaluate_contract(_markdown(), roadmap, set())["passed"] is False


def test_scenario_report_7156_active_and_blocked_artifacts(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7156-ACTIVE and PREFLIGHT preserve terminal causes."""

    roadmap = _roadmap()
    roadmap["tasks"] = roadmap["tasks"][:3]
    output = _write_inputs(tmp_path, roadmap)
    disqualified = mod.build_artifact(tmp_path, "20260909", output_path=output, run_commands=False)
    assert disqualified["status"] == "complete"
    assert disqualified["verdict_class"] == "disqualified"
    assert disqualified["observed_task_count"] == 3
    assert disqualified["gate_check_summary"] == {
        "failed_check": "yaml_task_count",
        "expected_value": 13,
        "observed_value": 3,
        "passed": False,
    }
    assert mod.validate_artifact(disqualified) == []

    (tmp_path / mod.ACTIVE_ROADMAP_PATH).unlink()
    blocked = mod.build_artifact(tmp_path, "20260909", output_path=output, run_commands=False)
    assert blocked["status"] == "complete"
    assert blocked["verdict_class"] == "blocked"
    assert blocked["inference_substrate_class"] == "blocked_no_run"
    assert blocked["gate_check_summary"]["failed_check"] == "active_yaml_readable"
    assert mod.validate_artifact(blocked) == []


def test_scenario_report_7156_commands_record_exit_codes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """SCENARIO-REPORT-7156-COMMANDS records every required subprocess."""

    output = _write_inputs(tmp_path)
    artifact = mod.build_artifact(tmp_path, "20260909", output_path=output, run_commands=False)
    calls: list[list[str]] = []

    def fake_run(argv: list[str], **_kwargs: object) -> SimpleNamespace:
        calls.append(argv)
        if len(calls) == 6:
            checkpoint = json.loads(output.read_text(encoding="utf-8"))
            assert mod.validate_artifact(checkpoint) == []
        return SimpleNamespace(returncode=0, stdout="clean", stderr="")

    monkeypatch.setattr(mod.subprocess, "run", fake_run)
    rows = mod.run_validation_commands(tmp_path, output, artifact)
    assert len(rows) == len(mod.VALIDATION_COMMAND_NAMES) == 10
    assert [row["name"] for row in rows] == list(mod.VALIDATION_COMMAND_NAMES)
    assert all(row["exit_code"] == 0 and row["passed"] for row in rows)
    assert len(calls) == 10
    output_text = capsys.readouterr().out
    assert "subprocess start" in output_text
    assert "subprocess end" in output_text


@pytest.mark.parametrize(
    ("field", "value", "error"),
    (
        ("execution_venue", "gpu", "execution_venue_invalid"),
        ("v630_task_contract_conforms_score", 0, "v630_task_contract_conforms_score_invalid"),
        ("verdict_class", "disqualified", "verdict_class_not_derived"),
        ("honest_verdict", "complete_positive_forged", "honest_verdict_not_derived"),
        ("reproducibility_checksum", "sha256:forged", "reproducibility_checksum_invalid"),
    ),
)
def test_scenario_report_7156_artifact_validator_rejects_forgery(
    tmp_path: Path, field: str, value: object, error: str
) -> None:
    """SCENARIO-REPORT-7156-ARTIFACT rejects forged derived state."""

    output = _write_inputs(tmp_path)
    artifact = mod.build_artifact(tmp_path, "20260909", output_path=output, run_commands=False)
    artifact[field] = value
    assert error in mod.validate_artifact(artifact)


def test_req_report_7156_principles_and_cli(tmp_path: Path) -> None:
    """REQ-REPORT-7156 preserves exact principles and validates CLI dates."""

    output = _write_inputs(tmp_path)
    artifact = mod.build_artifact(tmp_path, "20260909", output_path=output, run_commands=False)
    for field, principle in mod.REQUIRED_FIELD_PRINCIPLES.items():
        assert artifact["field_principles"][field] == principle
    assert json.loads(output.read_text(encoding="utf-8")) == artifact
    with pytest.raises(SystemExit):
        mod.main(["--date", "20260230", "--output", str(tmp_path / "bad.json")])


def test_req_report_7156_helper_failure_paths(tmp_path: Path) -> None:
    """REQ-REPORT-7156 rejects empty YAML and missing prompt declarations."""

    empty = tmp_path / "empty.yaml"
    empty.write_text("[]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="YAML mapping required"):
        mod._load_yaml(empty)
    assert mod._required_block("no declarations") == ""
    assert mod._closed_verdict_enum("no verdict field") is False
    assert mod._substrate_class("no substrate field") is None
    assert (
        mod._substrate_class("inference_substrate_class (aggregation or blocked_no_run)")
        == "aggregation"
    )


def test_scenario_report_7156_precondition_and_parse_failures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-REPORT-7156-PREFLIGHT distinguishes missing and malformed inputs."""

    output = _write_inputs(tmp_path)
    (tmp_path / mod.DESIGN_PATH).write_text("# readable but no contract\n", encoding="utf-8")
    malformed = mod.build_artifact(tmp_path, "20260909", output_path=output, run_commands=False)
    assert malformed["verdict_class"] == "disqualified"
    assert malformed["gate_check_summary"]["failed_check"] == "contract_parse"

    _write_inputs(tmp_path)
    (tmp_path / mod.DESIGN_PATH).unlink()
    missing = mod.build_artifact(tmp_path, "20260909", output_path=output, run_commands=False)
    assert missing["verdict_class"] == "blocked"
    assert missing["gate_check_summary"]["failed_check"] == "markdown_readable"

    _write_inputs(tmp_path)

    def unwritable(*_args: object, **_kwargs: object) -> object:
        raise OSError("read-only destination")

    monkeypatch.setattr(mod.tempfile, "NamedTemporaryFile", unwritable)
    rows, roadmap, exclusion = mod._preconditions(tmp_path, output)
    assert roadmap is not None and exclusion is not None
    assert rows[-1]["available"] is False
    assert "read-only destination" in rows[-1]["observed_value"]


def test_scenario_report_7156_failure_summary_paths() -> None:
    """SCENARIO-REPORT-7156-ACTIVE names each mismatch level exactly."""

    assert (
        mod._failure_summary(
            {"markdown_task_count": 14, "yaml_task_rows": [], "task_contract_rows": []}
        )["failed_check"]
        == "markdown_task_count"
    )
    assert (
        mod._failure_summary(
            {"markdown_task_count": 13, "yaml_task_rows": [], "task_contract_rows": []}
        )["failed_check"]
        == "yaml_task_count"
    )
    assert (
        mod._failure_summary(
            {
                "markdown_task_count": 13,
                "yaml_task_rows": [{}] * 13,
                "task_contract_rows": [{"order": 7, "passed": False}],
            }
        )["failed_check"]
        == "task_contract_order_7"
    )


def test_scenario_report_7156_timeout_and_build_command_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-REPORT-7156-COMMANDS records timeouts and build integration."""

    output = _write_inputs(tmp_path)
    artifact = mod.build_artifact(tmp_path, "20260909", output_path=output, run_commands=False)

    def timeout(argv: list[str], **_kwargs: object) -> object:
        raise mod.subprocess.TimeoutExpired(argv, 300, output="partial", stderr=None)

    monkeypatch.setattr(mod.subprocess, "run", timeout)
    rows = mod.run_validation_commands(tmp_path, output, artifact)
    assert all(row["exit_code"] == 124 for row in rows)
    assert all("timed out" in row["stderr"] for row in rows)

    monkeypatch.setattr(mod, "run_validation_commands", lambda *_args: [])
    integrated = mod.build_artifact(tmp_path, "20260909", output_path=output, run_commands=True)
    assert integrated["status"] == "complete"


def test_scenario_report_7156_score_rejects_each_evidence_break(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7156-ARTIFACT recomputes every score dependency."""

    output = _write_inputs(tmp_path)
    artifact = mod.build_artifact(tmp_path, "20260909", output_path=output, run_commands=False)
    assert mod._score_from_artifact(artifact) == 1
    mutations: tuple[Callable[[dict[str, Any]], None], ...] = (
        lambda value: value.__setitem__("markdown_task_rows", None),
        lambda value: value["yaml_task_rows"][0].__setitem__("id", "exp7156-v630-wrong"),
        lambda value: value["markdown_task_rows"][0].__setitem__("id", "exp7156-v630-wrong"),
        lambda value: value["task_contract_rows"][0].__setitem__("passed", False),
        lambda value: value.__setitem__("model_compliance_rows", "bad"),
        lambda value: value["progress_contract_rows"][0].__setitem__("passed", False),
    )
    for mutate in mutations:
        changed = deepcopy(artifact)
        mutate(changed)
        assert mod._score_from_artifact(changed) == 0

    malformed_pair = deepcopy(artifact)
    malformed_pair["markdown_task_rows"][0]["id"] = "exp7156-v630-wrong"
    malformed_pair["yaml_task_rows"][0]["id"] = "exp7156-v630-wrong"
    assert mod._score_from_artifact(malformed_pair) == 0


def test_scenario_report_7156_validator_diagnostic_paths(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7156-ARTIFACT reports every malformed artifact class."""

    output = _write_inputs(tmp_path)
    artifact = mod.build_artifact(tmp_path, "20260909", output_path=output, run_commands=False)
    assert mod.validate_artifact("bad") == ["artifact_mapping_required"]
    missing = deepcopy(artifact)
    missing.pop("run_date")
    assert mod.validate_artifact(missing)[0] == "missing_required_field:run_date"

    mutations: tuple[tuple[str, Callable[[dict[str, Any]], None]], ...] = (
        ("field_principles_invalid", lambda value: value.__setitem__("field_principles", {})),
        ("status_invalid", lambda value: value.__setitem__("status", "running")),
        (
            "inference_substrate_invalid",
            lambda value: value.__setitem__("inference_substrate", "model"),
        ),
        ("expected_task_count_invalid", lambda value: value.__setitem__("expected_task_count", 14)),
        ("random_seed_invalid", lambda value: value.__setitem__("random_seed", 1)),
        ("verifier_is_oracle_invalid", lambda value: value.__setitem__("verifier_is_oracle", True)),
        ("run_date_invalid", lambda value: value.__setitem__("run_date", "bad")),
        ("duration_s_invalid", lambda value: value.__setitem__("duration_s", True)),
        (
            "source_artifact_hashes_invalid",
            lambda value: value.__setitem__("source_artifact_hashes", []),
        ),
        ("observed_task_count_invalid", lambda value: value.__setitem__("observed_task_count", 0)),
        ("observed_id_order_invalid", lambda value: value.__setitem__("observed_id_order", [])),
        ("expected_id_order_invalid", lambda value: value.__setitem__("expected_id_order", [])),
        ("rows_not_task_contract_rows", lambda value: value.__setitem__("rows", [])),
        (
            "preconditions_checked_invalid",
            lambda value: value.__setitem__("preconditions_checked", "bad"),
        ),
        (
            "inference_substrate_class_invalid",
            lambda value: value.__setitem__("inference_substrate_class", "bad"),
        ),
        ("gate_check_summary_invalid", lambda value: value.__setitem__("gate_check_summary", {})),
        (
            "validation_command_rows_invalid",
            lambda value: value.__setitem__(
                "validation_command_rows", [{"name": "bad", "exit_code": "zero"}]
            ),
        ),
    )
    for error, mutate in mutations:
        changed = deepcopy(artifact)
        mutate(changed)
        assert error in mod.validate_artifact(changed)

    blocked = deepcopy(artifact)
    blocked["preconditions_checked"][0]["available"] = False
    blocked["inference_substrate_class"] = "blocked_no_run"
    blocked["v630_task_contract_conforms_score"] = 0
    blocked["verdict_class"] = "blocked"
    blocked["honest_verdict"] = mod.BLOCKED_VERDICT
    assert "gate_check_summary_invalid" in mod.validate_artifact(blocked)

    wrong_summary = deepcopy(artifact)
    wrong_summary["gate_check_summary"]["passed"] = False
    assert "gate_check_summary_invalid" in mod.validate_artifact(wrong_summary)

    forged_summary = deepcopy(artifact)
    forged_summary["gate_check_summary"] = {
        "failed_check": "forged_check",
        "expected_value": "forged_expected",
        "observed_value": "forged_observed",
        "passed": True,
    }
    forged_summary["reproducibility_checksum"] = mod.reproducibility_checksum(forged_summary)
    assert "gate_check_summary_invalid" in mod.validate_artifact(forged_summary)


def test_req_report_7156_main_success_and_error_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-REPORT-7156 resolves output paths and returns validator state."""

    assert mod._date_argument("20260909") == "20260909"
    seen: list[Path] = []

    def fake_build(
        _root: Path, _date: str, *, output_path: Path, run_commands: bool
    ) -> dict[str, Any]:
        seen.append(output_path)
        assert run_commands is False
        return {}

    monkeypatch.setattr(mod, "build_artifact", fake_build)
    monkeypatch.setattr(mod, "validate_artifact", lambda _artifact: ["invalid"])
    assert (
        mod.main(["--date", "20260909", "--output", "relative.json", "--skip-validation-commands"])
        == 1
    )
    assert seen[-1] == mod.REPO_ROOT / "relative.json"
    monkeypatch.setattr(mod, "validate_artifact", lambda _artifact: [])
    assert (
        mod.main(
            [
                "--date",
                "20260909",
                "--output",
                str(tmp_path / "absolute.json"),
                "--skip-validation-commands",
            ]
        )
        == 0
    )
    assert seen[-1] == tmp_path / "absolute.json"
