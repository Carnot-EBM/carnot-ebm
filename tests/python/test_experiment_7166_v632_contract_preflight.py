"""Focused tests for REQ-REPORT-7166 and its V632 contract scenarios."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

import pytest
import yaml

from carnot import experiment_7166_v632_contract_preflight as mod


ROOT = Path(__file__).resolve().parents[2]

# These rows intentionally repeat the document values. The Markdown fixture
# does not read or transform any object used by the YAML fixture.
MARKDOWN_ROWS = (
    ("exp7166-v632-exact-contract-preflight", "V632 exact Markdown and YAML task-contract preflight", "results/experiment_7166_v632_contract_preflight.json"),
    ("exp7167-qwen38-claim-evidence-trace-capture", "Qwen3.8 claim/evidence structured trace capture", "results/experiment_7167_v632_claim_evidence_trace_capture.json"),
    ("exp7168-claim-evidence-energy-comparison", "Frozen claim/evidence structural-energy comparison", "results/experiment_7168_v632_claim_evidence_energy_comparison.json"),
    ("exp7169-claim-evidence-causal-audit", "Independent claim/evidence causal and leakage audit", "results/experiment_7169_v632_claim_evidence_causal_audit.json"),
    ("exp7170-supersession-aware-constraint-stream", "Immutable supersession-aware chronological constraint stream", "results/experiment_7170_v632_supersession_constraint_stream.json"),
    ("exp7171-procedural-graph-constraint-memory-csl", "Held-out-safe procedural-graph constraint-memory self-learning", "results/experiment_7171_v632_procedural_graph_memory_csl.json"),
    ("exp7172-budgeted-stale-memory-allocation-ab", "Budgeted stale-constraint memory-allocation comparison", "results/experiment_7172_v632_stale_memory_allocation_ab.json"),
    ("exp7173-cold-retention-rollback-audit", "Fresh-process self-learning retention and rollback audit", "results/experiment_7173_v632_cold_retention_rollback_audit.json"),
    ("exp7174-live-arc-adapter-path-withheld-ab", "Live ARC adapter-knowledge path-withheld generalization A/B", "results/experiment_7174_v632_live_arc_path_withheld_ab.json"),
    ("exp7175-fixed-magnetization-pair-swap-benchmark", "Fixed-magnetization pair-swap sampler exact benchmark", "results/experiment_7175_v632_fixed_magnetization_sampler.json"),
    ("exp7176-rust-fixed-magnetization-parity", "Rust fixed-magnetization sampler parity and throughput", "results/experiment_7176_v632_rust_fixed_magnetization_parity.json"),
    ("exp7177-sampler-hardware-placement-audit", "Cross-substrate fixed-magnetization sampler placement audit", "results/experiment_7177_v632_sampler_hardware_placement.json"),
    ("exp7178-v632-independent-capstone", "V632 independent evidence capstone and next handoff", "results/experiment_7178_v632_capstone.json"),
)

YAML_ROWS = (
    ("exp7166-v632-exact-contract-preflight", "V632 exact Markdown and YAML task-contract preflight", "results/experiment_7166_v632_contract_preflight.json"),
    ("exp7167-qwen38-claim-evidence-trace-capture", "Qwen3.8 claim/evidence structured trace capture", "results/experiment_7167_v632_claim_evidence_trace_capture.json"),
    ("exp7168-claim-evidence-energy-comparison", "Frozen claim/evidence structural-energy comparison", "results/experiment_7168_v632_claim_evidence_energy_comparison.json"),
    ("exp7169-claim-evidence-causal-audit", "Independent claim/evidence causal and leakage audit", "results/experiment_7169_v632_claim_evidence_causal_audit.json"),
    ("exp7170-supersession-aware-constraint-stream", "Immutable supersession-aware chronological constraint stream", "results/experiment_7170_v632_supersession_constraint_stream.json"),
    ("exp7171-procedural-graph-constraint-memory-csl", "Held-out-safe procedural-graph constraint-memory self-learning", "results/experiment_7171_v632_procedural_graph_memory_csl.json"),
    ("exp7172-budgeted-stale-memory-allocation-ab", "Budgeted stale-constraint memory-allocation comparison", "results/experiment_7172_v632_stale_memory_allocation_ab.json"),
    ("exp7173-cold-retention-rollback-audit", "Fresh-process self-learning retention and rollback audit", "results/experiment_7173_v632_cold_retention_rollback_audit.json"),
    ("exp7174-live-arc-adapter-path-withheld-ab", "Live ARC adapter-knowledge path-withheld generalization A/B", "results/experiment_7174_v632_live_arc_path_withheld_ab.json"),
    ("exp7175-fixed-magnetization-pair-swap-benchmark", "Fixed-magnetization pair-swap sampler exact benchmark", "results/experiment_7175_v632_fixed_magnetization_sampler.json"),
    ("exp7176-rust-fixed-magnetization-parity", "Rust fixed-magnetization sampler parity and throughput", "results/experiment_7176_v632_rust_fixed_magnetization_parity.json"),
    ("exp7177-sampler-hardware-placement-audit", "Cross-substrate fixed-magnetization sampler placement audit", "results/experiment_7177_v632_sampler_hardware_placement.json"),
    ("exp7178-v632-independent-capstone", "V632 independent evidence capstone and next handoff", "results/experiment_7178_v632_capstone.json"),
)

MARKDOWN_GATES = (
    "none",
    "none",
    "`exp7167-qwen38-claim-evidence-trace-capture.claim_evidence_trace_ready_score == 1`",
    "`exp7168-claim-evidence-energy-comparison.claim_evidence_comparison_complete_score == 1`",
    "none",
    "`exp7170-supersession-aware-constraint-stream.supersession_stream_ready_score == 1`",
    "`exp7170-supersession-aware-constraint-stream.supersession_stream_ready_score == 1`",
    "`exp7171-procedural-graph-constraint-memory-csl.procedural_memory_comparison_complete_score == 1` AND `exp7172-budgeted-stale-memory-allocation-ab.stale_memory_allocation_complete_score == 1`",
    "none",
    "none",
    "`exp7175-fixed-magnetization-pair-swap-benchmark.fixed_magnetization_benchmark_complete_score == 1`",
    "`exp7176-rust-fixed-magnetization-parity.rust_fixed_magnetization_comparison_complete_score == 1`",
    "none",
)

YAML_GATES = {
    7168: [(7167, "claim_evidence_trace_ready_score")],
    7169: [(7168, "claim_evidence_comparison_complete_score")],
    7171: [(7170, "supersession_stream_ready_score")],
    7172: [(7170, "supersession_stream_ready_score")],
    7173: [(7171, "procedural_memory_comparison_complete_score"), (7172, "stale_memory_allocation_complete_score")],
    7176: [(7175, "fixed_magnetization_benchmark_complete_score")],
    7177: [(7176, "rust_fixed_magnetization_comparison_complete_score")],
}

PRODUCER_FIELDS = {
    7167: "claim_evidence_trace_ready_score",
    7168: "claim_evidence_comparison_complete_score",
    7170: "supersession_stream_ready_score",
    7171: "procedural_memory_comparison_complete_score",
    7172: "stale_memory_allocation_complete_score",
    7175: "fixed_magnetization_benchmark_complete_score",
    7176: "rust_fixed_magnetization_comparison_complete_score",
}

SUBSTRATES = {
    7166: "aggregation",
    7167: "model_full_generation",
    7168: "no_model_load",
    7169: "no_model_load",
    7170: "no_model_load",
    7171: "cpu_exact_solver_or_simulator",
    7172: "cpu_exact_solver_or_simulator",
    7173: "no_model_load",
    7174: "model_full_generation",
    7175: "cpu_exact_solver_or_simulator",
    7176: "cpu_exact_solver_or_simulator",
    7177: "aggregation",
    7178: "aggregation",
}


def _markdown() -> str:
    """Build the SCENARIO-REPORT-7166-PARITY Markdown input alone."""

    rows = [
        f"| {order} | `{task_id}` | {title} | `{deliverable}` | {MARKDOWN_GATES[order - 1]} |"
        for order, (task_id, title, deliverable) in enumerate(MARKDOWN_ROWS, 1)
    ]
    return "\n".join(
        (
            "# V632 fixture",
            "",
            "**Milestone:** `2026.09.632`",
            "",
            "## Exact Task Contract",
            "",
            "| Order | Task ID | Exact title | Deliverable | Structured gate |",
            "|---:|---|---|---|---|",
            *rows,
            "",
            "## Model and Substrate Contract",
            "",
            "| Task | MODEL_SPECS | Work | Substrate class |",
            "|---|---|---|---|",
            "| Exp7167 | sole entry `unsloth/Qwen3.8-27B-GGUF`, `Qwen3.8-27B-Q4_K_M.gguf`, Q4_K_M | generation | `model_full_generation` |",
            "| Exp7174 | sole entry `unsloth/Qwen3.8-27B-GGUF`, `Qwen3.8-27B-Q4_K_M.gguf`, Q4_K_M | generation | `model_full_generation` |",
        )
    )


def _prompt(number: int) -> str:
    """Create one YAML prompt with every SCENARIO-REPORT-7166-DISCIPLINE duty."""

    model_line = (
        '- MODEL_SPECS: principle: "Use only unsloth/Qwen3.8-27B-GGUF, Qwen3.8-27B-Q4_K_M.gguf, and Q4_K_M."\n'
        if number in {7167, 7174}
        else ""
    )
    producer_line = (
        f'- {PRODUCER_FIELDS[number]}: principle: "This bare field gates an earlier result."\n'
        if number in PRODUCER_FIELDS
        else ""
    )
    return (
        "CONTEXT:\nThe independent sources must agree.\n"
        "Work at {project_root}. The execution date is {date}.\n"
        "EXISTING CODE TO READ FIRST:\nRead both sources.\n"
        "TASK:\nAudit the contract.\n"
        "CONCRETE STEPS:\n"
        "0. Print and flush a phase-start line.\n"
        "1. At every numbered phase boundary, print a flushed progress line. Print flushed start "
        "and end lines immediately before and after every model load, generation, benchmark, or "
        "subprocess that can take minutes. Print a flushed heartbeat at least every 300 seconds "
        "inside every long loop. Keep every stdout gap below 600 seconds.\n"
        "2. REQUIRED ARTIFACT FIELDS:\n"
        '- field_principles: principle: "Explain every field."\n'
        '- status: principle: "Record a terminal state."\n'
        '- preconditions_checked: principle: "Record inputs."\n'
        '- run_date: principle: "Bind the date."\n'
        '- inference_substrate: principle: "Record the source."\n'
        f'- inference_substrate_class: principle: "Use {SUBSTRATES[number]}, or blocked_no_run."\n'
        '- execution_venue: principle: "Record the host."\n'
        '- duration_s: principle: "Record wall time."\n'
        '- source_artifact_hashes: principle: "Bind inputs."\n'
        '- rows: principle: "Preserve units."\n'
        + model_line
        + producer_line
        + '- random_seed: principle: "Stabilize order."\n'
        '- reproducibility_checksum: principle: "Detect drift."\n'
        '- gate_check_summary: principle: "Name the failed check and values."\n'
        '- verifier_is_oracle: principle: "Use false."\n'
        '- verdict_class: principle: "Use positive | circular_positive | null | blocked | disqualified | partial."\n'
        '- honest_verdict: principle: "Record the result."\n'
        f"Run command: cd {{project_root}} && .venv/bin/python scripts/experiments/experiment_{number}_v632_fixture.py --date {{date}}\n"
        + mod.PROMPT_FINAL_LINE
    )


def _roadmap() -> dict[str, Any]:
    """Build the SCENARIO-REPORT-7166-PARITY YAML input independently."""

    tasks: list[dict[str, Any]] = []
    ids_by_number = {7166 + index: row[0] for index, row in enumerate(YAML_ROWS)}
    for index, (task_id, title, deliverable) in enumerate(YAML_ROWS):
        number = 7166 + index
        task: dict[str, Any] = {
            "id": task_id,
            "title": title,
            "track": "infrastructure",
            "priority": "high",
            "requires_gpu": number in {7167, 7174},
            "max_turns": 50,
            "estimated_wall_time_min": 30,
            "per_unit_rows": True,
            "milestone": "2026.09.632",
            "deliverable": deliverable,
            "prior_failures": [],
            "prompt": _prompt(number),
        }
        if number in {7166, 7167, 7174}:
            task["model"] = "opus"
        else:
            task["agent_type"] = "codex"
            task["model"] = "gpt-5.6-sol"
        if number == 7166:
            task["prior_failures"] = [
                {
                    "experiment_id": "exp7159-v631-exact-contract-preflight",
                    "verdict": "complete_disqualified_v631_markdown_yaml_contract_mismatch",
                    "addressed_by": "V632 compares all thirteen independent rows.",
                    "retire_if_same_verdict": True,
                }
            ]
        if number in YAML_GATES:
            task["gated_on"] = [
                {
                    "upstream": ids_by_number[producer],
                    "artifact_field": field,
                    "op": "==",
                    "value": 1,
                }
                for producer, field in YAML_GATES[number]
            ]
        tasks.append(task)
    return {
        "milestone": "2026.09.632",
        "milestone_title": "fixture",
        "milestone_doc": "fixture.md",
        "tasks": tasks,
    }


def _write_inputs(
    root: Path,
    roadmap: dict[str, Any] | None = None,
    *,
    next_authority: bool = True,
) -> Path:
    """Write isolated SCENARIO-REPORT-7166-PREFLIGHT sources."""

    required = {
        mod.DESIGN_PATH: _markdown(),
        mod.EXCLUSION_PATH: "retired: []\n",
        mod.PRIOR_ARTIFACT_PATH: "{}\n",
        mod.SPEC_PATH: "REQ-REPORT-7166\n",
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
        Path("CLAUDE.md"): "# instructions\n",
        Path("CODEX.md"): "# instructions\n",
        Path("research-program.md"): "# program\n",
    }
    authority = mod.NEXT_ROADMAP_PATH if next_authority else mod.ACTIVE_ROADMAP_PATH
    required[authority] = yaml.safe_dump(roadmap or _roadmap(), sort_keys=False)
    for relative, content in required.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    return root / mod.DEFAULT_OUTPUT_PATH


def test_req_report_7166_spec_defines_fields_and_scenarios() -> None:
    """REQ-REPORT-7166 names all result fields and focused scenarios."""

    text = (ROOT / mod.SPEC_PATH).read_text(encoding="utf-8")
    section = text[text.index("### REQ-REPORT-7166") :]
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert field in section
    for name in ("PARITY", "GATES", "DISCIPLINE", "PREFLIGHT", "ACTIVE", "COMMANDS", "ARTIFACT"):
        assert f"SCENARIO-REPORT-7166-{name}" in section


def test_scenario_report_7166_parity_uses_independent_readers() -> None:
    """SCENARIO-REPORT-7166-PARITY accepts thirteen separate matching rows."""

    result = mod.evaluate_contract(_markdown(), _roadmap(), set())
    assert result["passed"] is True
    assert len(result["task_contract_rows"]) == 13
    assert result["expected_id_order"] == [row[0] for row in MARKDOWN_ROWS]
    assert result["markdown_task_rows"] is not result["yaml_task_rows"]
    assert result["gate_contract_rows"][7]["expected"][1]["upstream"] == MARKDOWN_ROWS[6][0]


@pytest.mark.parametrize(
    ("name", "mutate"),
    (
        ("missing_tail_task", lambda value: value["tasks"].pop()),
        ("reordered_id", lambda value: value["tasks"].__setitem__(slice(4, 6), [value["tasks"][5], value["tasks"][4]])),
        ("changed_title", lambda value: value["tasks"][0].__setitem__("title", "changed")),
        ("changed_deliverable", lambda value: value["tasks"][0].__setitem__("deliverable", "results/changed.json")),
        ("malformed_gate", lambda value: value["tasks"][2]["gated_on"][0].__setitem__("op", "!=")),
        ("absent_producer_field", lambda value: value["tasks"][1].__setitem__("prompt", value["tasks"][1]["prompt"].replace("claim_evidence_trace_ready_score", "removed_field"))),
        ("incomplete_prior_failure", lambda value: value["tasks"][0]["prior_failures"][0].__setitem__("addressed_by", "")),
        ("missing_model_specs", lambda value: value["tasks"][1].__setitem__("prompt", value["tasks"][1]["prompt"].replace("MODEL_SPECS", "MODEL_INFO"))),
        ("wrong_substrate", lambda value: value["tasks"][1].__setitem__("prompt", value["tasks"][1]["prompt"].replace("model_full_generation", "model_bounded_generation"))),
        ("silent_loop_prompt", lambda value: value["tasks"][1].__setitem__("prompt", value["tasks"][1]["prompt"].replace("at least every 300 seconds", "periodically"))),
        ("wrong_final_prohibition", lambda value: value["tasks"][1].__setitem__("prompt", value["tasks"][1]["prompt"].replace(mod.PROMPT_FINAL_LINE, "Do not push."))),
    ),
)
def test_req_report_7166_required_mutations_fail(
    name: str, mutate: Callable[[dict[str, Any]], object]
) -> None:
    """REQ-REPORT-7166 rejects each required contract mutation."""

    roadmap = _roadmap()
    mutate(roadmap)
    result = mod.evaluate_contract(_markdown(), roadmap, set())
    assert result["passed"] is False, name


def test_scenario_report_7166_gates_and_metadata_fail_closed() -> None:
    """SCENARIO-REPORT-7166-GATES and DISCIPLINE reject other malformed data."""

    mutations: tuple[Callable[[dict[str, Any]], None], ...] = (
        lambda value: value["tasks"][2]["gated_on"][0].__setitem__("artifact_field", "nested.value"),
        lambda value: value["tasks"][2]["gated_on"][0].__setitem__("upstream", YAML_ROWS[-1][0]),
        lambda value: value["tasks"][1].__setitem__("agent_type", "gemini"),
        lambda value: value["tasks"][1].__setitem__("per_unit_rows", False),
        lambda value: value["tasks"][1].__setitem__("prompt", value["tasks"][1]["prompt"].replace("Qwen3.8-27B", "Qwen3.6-35B")),
        lambda value: value["tasks"][1].__setitem__("operator_override", "two\nlines"),
        lambda value: value["tasks"][1].__setitem__("prompt", value["tasks"][1]["prompt"].replace("{project_root}", "/tmp/project")),
        lambda value: value["tasks"][1].__setitem__("prompt", value["tasks"][1]["prompt"].replace("experiment_7167_v632_fixture.py", "wrong.py")),
    )
    for mutate in mutations:
        roadmap = _roadmap()
        mutate(roadmap)
        assert mod.evaluate_contract(_markdown(), roadmap, set())["passed"] is False


def test_scenario_report_7166_active_and_blocked_lifecycle(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7166-ACTIVE and PREFLIGHT preserve terminal causes."""

    short = _roadmap()
    short["tasks"] = short["tasks"][:4]
    output = _write_inputs(tmp_path, short, next_authority=False)
    disqualified = mod.build_artifact(tmp_path, "20260909", output_path=output, run_commands=False)
    assert disqualified["status"] == "complete"
    assert disqualified["verdict_class"] == "disqualified"
    assert disqualified["yaml_authority_path"] == str(mod.ACTIVE_ROADMAP_PATH)
    assert disqualified["observed_task_count"] == 4
    assert disqualified["gate_check_summary"] == {
        "failed_check": "yaml_task_count",
        "expected_value": 13,
        "observed_value": 4,
        "passed": False,
    }
    assert mod.validate_artifact(disqualified) == []

    (tmp_path / mod.ACTIVE_ROADMAP_PATH).unlink()
    blocked = mod.build_artifact(tmp_path, "20260909", output_path=output, run_commands=False)
    assert blocked["status"] == "complete"
    assert blocked["verdict_class"] == "blocked"
    assert blocked["inference_substrate_class"] == "blocked_no_run"
    assert blocked["gate_check_summary"]["failed_check"] == "yaml_authority_readable"
    assert mod.validate_artifact(blocked) == []


def test_scenario_report_7166_next_authority_and_optional_active_mismatch(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7166-PREFLIGHT prefers next and records active drift."""

    output = _write_inputs(tmp_path)
    active = _roadmap()
    active["tasks"][0]["title"] = "active changed"
    (tmp_path / mod.ACTIVE_ROADMAP_PATH).write_text(yaml.safe_dump(active, sort_keys=False), encoding="utf-8")
    artifact = mod.build_artifact(tmp_path, "20260909", output_path=output, run_commands=False)
    assert artifact["yaml_authority_path"] == str(mod.NEXT_ROADMAP_PATH)
    assert artifact["authority_consistency_rows"][0]["passed"] is False
    assert artifact["verdict_class"] == "disqualified"
    assert mod.validate_artifact(artifact) == []


def test_scenario_report_7166_commands_record_boundaries_and_timeouts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """SCENARIO-REPORT-7166-COMMANDS records exit codes and timeouts."""

    output = _write_inputs(tmp_path)
    artifact = mod.build_artifact(tmp_path, "20260909", output_path=output, run_commands=False)
    calls: list[list[str]] = []

    def fake_run(argv: list[str], **_kwargs: object) -> SimpleNamespace:
        calls.append(argv)
        if len(calls) == 6:
            assert mod.validate_artifact(json.loads(output.read_text(encoding="utf-8"))) == []
        return SimpleNamespace(returncode=0, stdout="clean", stderr="")

    monkeypatch.setattr(mod.subprocess, "run", fake_run)
    rows = mod.run_validation_commands(tmp_path, output, artifact)
    assert [row["name"] for row in rows] == list(mod.VALIDATION_COMMAND_NAMES)
    assert all(row["exit_code"] == 0 and row["passed"] for row in rows)
    assert len(calls) == 10
    assert "subprocess start" in capsys.readouterr().out

    def timeout(argv: list[str], **_kwargs: object) -> object:
        raise mod.subprocess.TimeoutExpired(argv, 300, output="partial", stderr=None)

    monkeypatch.setattr(mod.subprocess, "run", timeout)
    rows = mod.run_validation_commands(tmp_path, output, artifact)
    assert all(row["exit_code"] == 124 and "timed out" in row["stderr"] for row in rows)


def test_scenario_report_7166_artifact_rejects_forgery(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7166-ARTIFACT recomputes lifecycle and evidence."""

    output = _write_inputs(tmp_path)
    artifact = mod.build_artifact(tmp_path, "20260909", output_path=output, run_commands=False)
    assert mod._score_from_artifact(artifact) == 1
    mutations: tuple[tuple[str, Callable[[dict[str, Any]], None]], ...] = (
        ("field_principles_invalid", lambda value: value.__setitem__("field_principles", {})),
        ("status_invalid", lambda value: value.__setitem__("status", "running")),
        ("inference_substrate_invalid", lambda value: value.__setitem__("inference_substrate", "model")),
        ("execution_venue_invalid", lambda value: value.__setitem__("execution_venue", "gpu")),
        ("expected_task_count_invalid", lambda value: value.__setitem__("expected_task_count", 12)),
        ("random_seed_invalid", lambda value: value.__setitem__("random_seed", 1)),
        ("verifier_is_oracle_invalid", lambda value: value.__setitem__("verifier_is_oracle", True)),
        ("run_date_invalid", lambda value: value.__setitem__("run_date", "bad")),
        ("duration_s_invalid", lambda value: value.__setitem__("duration_s", True)),
        ("source_artifact_hashes_invalid", lambda value: value.__setitem__("source_artifact_hashes", [])),
        ("observed_task_count_invalid", lambda value: value.__setitem__("observed_task_count", 0)),
        ("observed_id_order_invalid", lambda value: value.__setitem__("observed_id_order", [])),
        ("expected_id_order_invalid", lambda value: value.__setitem__("expected_id_order", [])),
        ("rows_not_task_contract_rows", lambda value: value.__setitem__("rows", [])),
        ("preconditions_checked_invalid", lambda value: value.__setitem__("preconditions_checked", "bad")),
        ("inference_substrate_class_invalid", lambda value: value.__setitem__("inference_substrate_class", "bad")),
        ("verdict_class_not_derived", lambda value: value.__setitem__("verdict_class", "null")),
        ("honest_verdict_not_derived", lambda value: value.__setitem__("honest_verdict", "forged")),
        ("gate_check_summary_invalid", lambda value: value.__setitem__("gate_check_summary", {})),
        ("validation_command_rows_invalid", lambda value: value.__setitem__("validation_command_rows", [{"name": "bad", "exit_code": "zero"}])),
        ("reproducibility_checksum_invalid", lambda value: value.__setitem__("reproducibility_checksum", "sha256:forged")),
    )
    for error, mutate in mutations:
        changed = deepcopy(artifact)
        mutate(changed)
        assert error in mod.validate_artifact(changed)

    broken = deepcopy(artifact)
    broken["progress_contract_rows"][0]["passed"] = False
    assert mod._score_from_artifact(broken) == 0
    malformed = deepcopy(artifact)
    malformed["markdown_task_rows"] = None
    assert mod._score_from_artifact(malformed) == 0
    assert mod.validate_artifact("bad") == ["artifact_mapping_required"]
    missing = deepcopy(artifact)
    missing.pop("run_date")
    assert mod.validate_artifact(missing)[0] == "missing_required_field:run_date"


def test_req_report_7166_helpers_and_main_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-REPORT-7166 covers helper failures and both CLI outcomes."""

    bad = tmp_path / "bad.yaml"
    bad.write_text("[]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="YAML mapping required"):
        mod._load_yaml(bad)
    assert mod._required_block("no declarations") == ""
    assert mod._valid_task_id("bad", 7166) is False
    assert mod._valid_task_id("exp7165-wrong", 7165) is False
    assert mod._closed_verdict_enum("no verdict") is False
    assert mod._substrate_class("no substrate") is None
    assert mod._date_argument("20260909") == "20260909"
    with pytest.raises(SystemExit):
        mod.main(["--date", "20260230"])

    seen: list[Path] = []

    def fake_build(
        _root: Path, _date: str, *, output_path: Path, run_commands: bool
    ) -> dict[str, Any]:
        seen.append(output_path)
        assert run_commands is False
        return {}

    monkeypatch.setattr(mod, "build_artifact", fake_build)
    monkeypatch.setattr(mod, "validate_artifact", lambda _value: ["invalid"])
    assert mod.main(["--date", "20260909", "--output", "relative.json", "--skip-validation-commands"]) == 1
    assert seen[-1] == mod.REPO_ROOT / "relative.json"
    monkeypatch.setattr(mod, "validate_artifact", lambda _value: [])
    assert mod.main(["--date", "20260909", "--output", str(tmp_path / "absolute.json"), "--skip-validation-commands"]) == 0
    assert seen[-1] == tmp_path / "absolute.json"


def test_scenario_report_7166_precondition_parse_and_write_failures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-REPORT-7166-PREFLIGHT separates missing and malformed inputs."""

    output = _write_inputs(tmp_path)
    (tmp_path / mod.DESIGN_PATH).write_text("# no contract\n", encoding="utf-8")
    malformed = mod.build_artifact(tmp_path, "20260909", output_path=output, run_commands=False)
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
    rows, roadmap, authority = mod._preconditions(tmp_path, output)
    assert roadmap is not None and authority == mod.NEXT_ROADMAP_PATH
    assert rows[-1]["available"] is False


def test_scenario_report_7166_build_command_and_summary_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-REPORT-7166-COMMANDS integrates receipts and mismatch summaries."""

    output = _write_inputs(tmp_path)
    monkeypatch.setattr(mod, "run_validation_commands", lambda *_args: [])
    assert mod.build_artifact(tmp_path, "20260909", output_path=output, run_commands=True)["status"] == "complete"
    assert mod._failure_summary({"markdown_task_count": 12, "yaml_task_rows": [], "task_contract_rows": []})["failed_check"] == "markdown_task_count"
    assert mod._failure_summary({"markdown_task_count": 13, "yaml_task_rows": [], "task_contract_rows": []})["failed_check"] == "yaml_task_count"
    assert mod._failure_summary({"markdown_task_count": 13, "yaml_task_rows": [{}] * 13, "authority_consistency_rows": [{"passed": False}], "task_contract_rows": []})["failed_check"] == "active_yaml_contract_parity"
    assert mod._failure_summary({"markdown_task_count": 13, "yaml_task_rows": [{}] * 13, "authority_consistency_rows": [{"passed": True}], "task_contract_rows": [{"order": 7, "passed": False}]})["failed_check"] == "task_contract_order_7"


def test_scenario_report_7166_remaining_failure_evidence(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7166-ARTIFACT covers each stored failure decision."""

    assert mod._substrate_class("inference_substrate_class (aggregation)") == "aggregation"

    output = _write_inputs(tmp_path)
    (tmp_path / mod.EXCLUSION_PATH).write_text("[]\n", encoding="utf-8")
    preconditions, roadmap, authority = mod._preconditions(tmp_path, output)
    assert roadmap is not None and authority == mod.NEXT_ROADMAP_PATH
    assert next(row for row in preconditions if row["check"] == "exclusion_manifest_readable")[
        "available"
    ] is False

    _write_inputs(tmp_path)
    (tmp_path / mod.ACTIVE_ROADMAP_PATH).write_text("[]\n", encoding="utf-8")
    assert mod._authority_consistency_rows(tmp_path, mod.NEXT_ROADMAP_PATH, _roadmap())[0][
        "passed"
    ] is False

    (tmp_path / mod.ACTIVE_ROADMAP_PATH).unlink()
    positive = mod.build_artifact(tmp_path, "20260909", output_path=output, run_commands=False)
    changed = deepcopy(positive)
    changed["markdown_task_rows"][0]["id"] = "different"
    assert mod._score_from_artifact(changed) == 0
    changed = deepcopy(positive)
    changed["markdown_task_rows"][0]["id"] = "exp7166-wrong"
    changed["yaml_task_rows"][0]["id"] = "exp7166-wrong"
    assert mod._score_from_artifact(changed) == 0
    changed = deepcopy(positive)
    changed["task_contract_rows"][0]["passed"] = False
    assert mod._score_from_artifact(changed) == 0
    changed = deepcopy(positive)
    changed["v632_task_contract_conforms_score"] = 0
    assert "v632_task_contract_conforms_score_invalid" in mod.validate_artifact(changed)

    blocked_root = tmp_path / "blocked"
    blocked_output = _write_inputs(blocked_root, next_authority=False)
    (blocked_root / mod.ACTIVE_ROADMAP_PATH).unlink()
    blocked = mod.build_artifact(
        blocked_root, "20260909", output_path=blocked_output, run_commands=False
    )
    blocked["gate_check_summary"]["observed_value"] = "forged"
    assert "gate_check_summary_invalid" in mod.validate_artifact(blocked)

    short_root = tmp_path / "short"
    short = _roadmap()
    short["tasks"].pop()
    short_output = _write_inputs(short_root, short)
    disqualified = mod.build_artifact(
        short_root, "20260909", output_path=short_output, run_commands=False
    )
    disqualified["gate_check_summary"]["passed"] = True
    assert "gate_check_summary_invalid" in mod.validate_artifact(disqualified)
