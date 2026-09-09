"""Focused tests for the independent V629 active-roadmap contract preflight."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any, Callable

import pytest
import yaml

from carnot import experiment_7151_v629_contract_preflight as mod


ROOT = Path(__file__).resolve().parents[2]

# This Markdown fixture is independent from YAML_ROWS. Duplication is
# intentional because sharing one row source would hide a parser-side omission.
MARKDOWN_ROWS = (
    ("exp7151-v629-active-contract-preflight", "V629 active Markdown and YAML task-contract preflight", "results/experiment_7151_v629_contract_preflight.json"),
    ("exp7152-v629-source-and-cache-delta", "V629 execution-time source and SOTA cache delta", "results/experiment_7152_v629_source_delta.json"),
    ("exp7153-grounding-runtime-postfix-qualification", "Post-fix source-grounding runtime qualification", "results/experiment_7153_v629_grounding_runtime.json"),
    ("exp7154-qwen-dual-side-grounding-pilot", "Qwen dual-side source-grounding pilot", "results/experiment_7154_v629_qwen_dual_side_grounding.json"),
    ("exp7155-gemma-dual-side-grounding-replication", "Two-Gemma dual-side grounding replication", "results/experiment_7155_v629_gemma_dual_side_grounding.json"),
    ("exp7156-dual-side-grounding-causal-audit", "Independent dual-side grounding causal audit", "results/experiment_7156_v629_grounding_causal_audit.json"),
    ("exp7157-flowbalance-transaction-preflight", "FlowBalance memory transaction preflight", "results/experiment_7157_v629_flowbalance_transaction.json"),
    ("exp7158-procedure-family-memory-csl", "Prospective procedure-family continuous self-learning", "results/experiment_7158_v629_procedure_family_csl.json"),
    ("exp7159-memory-drift-poison-cold-audit", "Frozen memory drift, poison, and transfer audit", "results/experiment_7159_v629_memory_drift_poison_audit.json"),
    ("exp7160-state-localized-arc-invariance", "State-localized ARC same-state invariance study", "results/experiment_7160_v629_arc_state_invariance.json"),
    ("exp7161-rust-multiscale-exact-parity", "Rust corrected multiscale sampler exact parity", "results/experiment_7161_v629_rust_multiscale_parity.json"),
    ("exp7162-rust-multiscale-orchestration-profile", "Rust multiscale sampler orchestration profile", "results/experiment_7162_v629_rust_orchestration_profile.json"),
    ("exp7163-three-board-continuity", "KV260, PolarFire, and GateMate bounded continuity", "results/experiment_7163_v629_three_board_continuity.json"),
    ("exp7164-v629-capstone", "V629 independent evidence matrix and branch disposition", "results/experiment_7164_v629_capstone.json"),
)
MARKDOWN_GATES = {
    7154: "exp7153-grounding-runtime-postfix-qualification.grounding_runtime_ready_score == 1",
    7155: "exp7154-qwen-dual-side-grounding-pilot.qwen_dual_side_pilot_complete_score == 1",
    7156: "exp7155-gemma-dual-side-grounding-replication.gemma_dual_side_replication_complete_score == 1",
    7158: "exp7157-flowbalance-transaction-preflight.flowbalance_transaction_ready_score == 1",
    7159: "exp7158-procedure-family-memory-csl.procedure_family_csl_complete_score == 1",
    7162: "exp7161-rust-multiscale-exact-parity.rust_multiscale_exact_parity_score == 1",
}
MARKDOWN_CONTRACT = "\n".join(
    (
        "# Independent Markdown fixture",
        "",
        "**Milestone:** `2026.09.629`",
        "",
        "## Exact task contract",
        "",
        "| Order | Task ID | Exact title | Deliverable | Structured gate |",
        "|---:|---|---|---|---|",
        *(
            f"| {order} | `{task_id}` | {title} | `{deliverable}` | "
            f"{('`' + MARKDOWN_GATES[number] + '`') if number in MARKDOWN_GATES else 'none'} |"
            for order, (task_id, title, deliverable) in enumerate(MARKDOWN_ROWS, start=1)
            for number in (int(task_id[3:7]),)
        ),
    )
)

# The YAML fixture repeats the expected values without consulting MARKDOWN_ROWS.
YAML_ROWS = (
    ("exp7151-v629-active-contract-preflight", "V629 active Markdown and YAML task-contract preflight", "results/experiment_7151_v629_contract_preflight.json"),
    ("exp7152-v629-source-and-cache-delta", "V629 execution-time source and SOTA cache delta", "results/experiment_7152_v629_source_delta.json"),
    ("exp7153-grounding-runtime-postfix-qualification", "Post-fix source-grounding runtime qualification", "results/experiment_7153_v629_grounding_runtime.json"),
    ("exp7154-qwen-dual-side-grounding-pilot", "Qwen dual-side source-grounding pilot", "results/experiment_7154_v629_qwen_dual_side_grounding.json"),
    ("exp7155-gemma-dual-side-grounding-replication", "Two-Gemma dual-side grounding replication", "results/experiment_7155_v629_gemma_dual_side_grounding.json"),
    ("exp7156-dual-side-grounding-causal-audit", "Independent dual-side grounding causal audit", "results/experiment_7156_v629_grounding_causal_audit.json"),
    ("exp7157-flowbalance-transaction-preflight", "FlowBalance memory transaction preflight", "results/experiment_7157_v629_flowbalance_transaction.json"),
    ("exp7158-procedure-family-memory-csl", "Prospective procedure-family continuous self-learning", "results/experiment_7158_v629_procedure_family_csl.json"),
    ("exp7159-memory-drift-poison-cold-audit", "Frozen memory drift, poison, and transfer audit", "results/experiment_7159_v629_memory_drift_poison_audit.json"),
    ("exp7160-state-localized-arc-invariance", "State-localized ARC same-state invariance study", "results/experiment_7160_v629_arc_state_invariance.json"),
    ("exp7161-rust-multiscale-exact-parity", "Rust corrected multiscale sampler exact parity", "results/experiment_7161_v629_rust_multiscale_parity.json"),
    ("exp7162-rust-multiscale-orchestration-profile", "Rust multiscale sampler orchestration profile", "results/experiment_7162_v629_rust_orchestration_profile.json"),
    ("exp7163-three-board-continuity", "KV260, PolarFire, and GateMate bounded continuity", "results/experiment_7163_v629_three_board_continuity.json"),
    ("exp7164-v629-capstone", "V629 independent evidence matrix and branch disposition", "results/experiment_7164_v629_capstone.json"),
)
GATES = {
    7154: (7153, "grounding_runtime_ready_score"),
    7155: (7154, "qwen_dual_side_pilot_complete_score"),
    7156: (7155, "gemma_dual_side_replication_complete_score"),
    7158: (7157, "flowbalance_transaction_ready_score"),
    7159: (7158, "procedure_family_csl_complete_score"),
    7162: (7161, "rust_multiscale_exact_parity_score"),
}
PRIOR_IDS = {
    7151: ("exp7148-v628-contract-preflight",),
    7152: ("exp7149-v628-source-and-cache-delta", "exp6461-v556-sota-source-and-benchmark-delta"),
    7153: ("exp7150-source-grounding-preflight-repair", "exp7139-three-family-symbolic-grounding-ab"),
    7154: ("exp7139-three-family-symbolic-grounding-ab",),
    7155: ("exp7139-three-family-symbolic-grounding-ab",),
    7156: (),
    7157: ("exp7142-flowbalance-external-memory-csl",),
    7158: (),
    7159: (),
    7160: (),
    7161: ("exp7145-rust-multiscale-sampler-parity",),
    7162: ("exp7145-rust-multiscale-sampler-parity",),
    7163: ("exp7146-gatemate-changed-state-continuity",),
    7164: (),
}
MODEL_TASKS = {
    7153: ("unsloth/Qwen3.6-35B-A3B-GGUF",),
    7154: ("unsloth/Qwen3.6-35B-A3B-GGUF",),
    7155: ("unsloth/gemma-4-31B-it-GGUF", "unsloth/gemma-4-26B-A4B-it-GGUF"),
    7158: ("unsloth/Qwen3.6-35B-A3B-GGUF",),
    7159: ("unsloth/Qwen3.6-35B-A3B-GGUF",),
    7160: ("unsloth/Qwen3.6-35B-A3B-GGUF",),
}
SUBSTRATE_CLASSES = {
    7151: "aggregation",
    7152: "aggregation",
    7153: "model_full_generation",
    7154: "model_full_generation",
    7155: "model_full_generation",
    7156: "aggregation",
    7157: "no_model_load",
    7158: "model_full_generation",
    7159: "model_full_generation",
    7160: "model_full_generation",
    7161: "cpu_exact_solver_or_simulator",
    7162: "cpu_exact_solver_or_simulator",
    7163: "hardware_smoke",
    7164: "aggregation",
}
SUFFIXES = {
    7151: "contract_preflight",
    7152: "source_delta",
    7153: "grounding_runtime",
    7154: "qwen_dual_side_grounding",
    7155: "gemma_dual_side_grounding",
    7156: "grounding_causal_audit",
    7157: "flowbalance_transaction",
    7158: "procedure_family_csl",
    7159: "memory_drift_poison_audit",
    7160: "arc_state_invariance",
    7161: "rust_multiscale_parity",
    7162: "rust_orchestration_profile",
    7163: "three_board_continuity",
    7164: "capstone",
}
OVERRIDE_TASKS = {7163, 7164}


def _required_fields(number: int) -> list[str]:
    """Create YAML declarations without reading the Markdown fixture."""

    fields = [
        "field_principles for every field below",
        "preconditions_checked",
        "run_date",
        "inference_substrate",
        f"inference_substrate_class ({SUBSTRATE_CLASSES[number]} or blocked_no_run)",
        "execution_venue",
        "duration_s",
        "source_artifact_hashes",
        "rows",
    ]
    if number in MODEL_TASKS:
        fields.append("MODEL_SPECS")
    producer_field = GATES.get(number + 1, (None, None))[1]
    if producer_field:
        fields.append(producer_field)
    fields.extend(
        (
            "random_seed",
            "reproducibility_checksum",
            "gate_check_summary with failed check, expected value, and observed value",
            "verifier_is_oracle (false)",
            "verdict_class (positive | circular_positive | null | blocked | disqualified | partial)",
            "honest_verdict consistent with verdict_class",
        )
    )
    return fields


def _prompt(number: int) -> str:
    """Create one complete YAML prompt for contract-policy checks."""

    focused = f"tests/python/test_experiment_{number}_v629_{SUFFIXES[number]}.py"
    run_line = (
        "Run command: cd {project_root} && .venv/bin/python "
        f"scripts/experiments/experiment_{number}_v629_{SUFFIXES[number]}.py --date {{date}}"
    )
    models = "".join(
        f"MODEL_SPECS must include {model}.\n" for model in MODEL_TASKS.get(number, ())
    )
    return (
        "CONCRETE STEPS:\n"
        "0. Create the schema-complete running artifact before any check.\n"
        "1. At every numbered phase boundary, print a flushed progress line. Print flushed start "
        "and end lines immediately before and after every call that can take minutes, including a "
        "model load, generation, benchmark, or subprocess. Print a flushed heartbeat at least every "
        "300 seconds inside long loops. Keep every stdout gap below 600 seconds.\n"
        f"2. Add focused RED tests in {focused}.\n"
        + models
        + "REQUIRED ARTIFACT FIELDS: "
        + "; ".join(_required_fields(number))
        + ".\n\n"
        + run_line
        + "\n"
        + mod.PROMPT_FINAL_LINE
        + "\n"
    )


def _roadmap() -> dict[str, Any]:
    """Build active YAML data without parsing the Markdown fixture."""

    tasks: list[dict[str, Any]] = []
    for task_id, title, deliverable in YAML_ROWS:
        number = int(task_id[3:7])
        task: dict[str, Any] = {
            "id": task_id,
            "title": title,
            "track": "infrastructure",
            "priority": "high",
            "requires_gpu": number in MODEL_TASKS,
            "max_turns": 50,
            "estimated_wall_time_min": 45,
            "per_unit_rows": True,
            "milestone": "2026.09.629",
            "deliverable": deliverable,
            "prior_failures": [
                {
                    "experiment_id": prior_id,
                    "verdict": "blocked_prior_attempt",
                    "addressed_by": "V629 changes the method or execution boundary.",
                    "retire_if_same_verdict": True,
                }
                for prior_id in PRIOR_IDS[number]
            ],
            "prompt": _prompt(number),
        }
        if number in {7151, 7153, 7163}:
            task["model"] = "opus"
        elif number not in {7152, 7154, 7155, 7164}:
            task["agent_type"] = "codex"
            task["model"] = "gpt-5.6-sol"
        if number in GATES:
            upstream_number, field = GATES[number]
            upstream_id = next(
                row[0] for row in YAML_ROWS if row[0].startswith(f"exp{upstream_number}-")
            )
            task["gated_on"] = [
                {"upstream": upstream_id, "artifact_field": field, "op": "==", "value": 1}
            ]
        if number in OVERRIDE_TASKS:
            task["operator_override"] = (
                "2026-05-29 operator directive (standing): routine continuation — "
                "false-positive scope-match vs exp7000; this task advances V629."
            )
        tasks.append(task)
    return {
        "milestone": "2026.09.629",
        "milestone_title": "fixture",
        "milestone_doc": "fixture.md",
        "tasks": tasks,
    }


def _write_inputs(root: Path, roadmap: dict[str, Any] | None = None) -> Path:
    """Write temporary inputs so tests do not change the research record."""

    for relative in (mod.DESIGN_PATH, mod.ACTIVE_ROADMAP_PATH, mod.EXCLUSION_PATH):
        (root / relative).parent.mkdir(parents=True, exist_ok=True)
    (root / mod.DESIGN_PATH).write_text(MARKDOWN_CONTRACT, encoding="utf-8")
    (root / mod.ACTIVE_ROADMAP_PATH).write_text(
        yaml.safe_dump(roadmap or _roadmap(), sort_keys=False), encoding="utf-8"
    )
    (root / mod.EXCLUSION_PATH).write_text("retired: []\n", encoding="utf-8")
    return root / mod.DEFAULT_OUTPUT_PATH


def test_req_report_7151_spec_defines_contract_and_scenarios() -> None:
    """REQ-REPORT-7151 names every ID, field, and required scenario."""

    text = (ROOT / mod.SPEC_PATH).read_text(encoding="utf-8")
    section = text[text.index("### REQ-REPORT-7151") :]
    for task_id, _title, _deliverable in MARKDOWN_ROWS:
        assert task_id in section
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert field in section
    for name in ("PARITY", "GATES", "DISCIPLINE", "PREFLIGHT", "ACTIVE", "ARTIFACT"):
        assert f"SCENARIO-REPORT-7151-{name}" in section


def test_scenario_report_7151_parity_accepts_independent_rows() -> None:
    """SCENARIO-REPORT-7151-PARITY accepts separate matching sources."""

    result = mod.evaluate_contract(MARKDOWN_CONTRACT, _roadmap(), set())
    assert result["passed"] is True
    assert len(result["task_contract_rows"]) == 14
    assert len(result["gate_producer_rows"]) == 6
    assert len(result["operator_override_rows"]) == 14
    assert result["markdown_task_rows"] is not result["yaml_task_rows"]
    assert result["markdown_task_rows"][3]["gates"][0]["upstream"] == YAML_ROWS[2][0]


@pytest.mark.parametrize(
    "case", ("missing", "extra", "reordered", "duplicated", "renamed", "title", "deliverable")
)
def test_req_report_7151_row_mutations_fail(case: str) -> None:
    """REQ-REPORT-7151 rejects every row identity and parity mutation."""

    roadmap = _roadmap()
    if case == "missing":
        roadmap["tasks"].pop(6)
    elif case == "extra":
        roadmap["tasks"].append(deepcopy(roadmap["tasks"][-1]))
    elif case == "reordered":
        roadmap["tasks"][6], roadmap["tasks"][7] = roadmap["tasks"][7], roadmap["tasks"][6]
    elif case == "duplicated":
        roadmap["tasks"][6] = deepcopy(roadmap["tasks"][5])
    elif case == "renamed":
        roadmap["tasks"][0]["id"] += "-changed"
    elif case == "title":
        roadmap["tasks"][0]["title"] += " changed"
    else:
        roadmap["tasks"][0]["deliverable"] = "results/wrong.json"
    assert mod.evaluate_contract(MARKDOWN_CONTRACT, roadmap, set())["passed"] is False


def test_scenario_report_7151_active_short_roadmap_is_disqualified(
    tmp_path: Path,
) -> None:
    """SCENARIO-REPORT-7151-ACTIVE preserves the short active-YAML count."""

    roadmap = _roadmap()
    roadmap["tasks"] = roadmap["tasks"][:5]
    output = _write_inputs(tmp_path, roadmap)
    artifact = mod.build_artifact(tmp_path, "20260909", output_path=output)
    assert artifact["verdict_class"] == "disqualified"
    assert artifact["observed_task_count"] == 5
    assert artifact["gate_check_summary"] == {
        "failed_check": "yaml_task_count",
        "expected_value": 14,
        "observed_value": 5,
        "passed": False,
    }
    assert mod.validate_artifact(artifact) == []


def test_scenario_report_7151_malformed_sources_fail() -> None:
    """SCENARIO-REPORT-7151-PARITY rejects malformed source syntax."""

    for changed in (
        MARKDOWN_CONTRACT.replace("**Milestone:**", "**Release:**", 1),
        MARKDOWN_CONTRACT.replace("## Exact task contract", "## Tasks", 1),
        MARKDOWN_CONTRACT.replace("| 1 |", "| one |", 1),
        MARKDOWN_CONTRACT.replace(" == 1`", " is ready`", 1),
    ):
        with pytest.raises(ValueError):
            mod.parse_markdown_contract(changed)
    for malformed in ([], {"tasks": "bad"}, {"tasks": ["bad"]}):
        with pytest.raises(ValueError):
            mod.parse_yaml_contract(malformed)


def test_scenario_report_7151_gate_mutations_fail() -> None:
    """SCENARIO-REPORT-7151-GATES rejects wrong or unusable producers."""

    mutations: tuple[Callable[[dict[str, Any]], None], ...] = (
        lambda value: value["tasks"][3]["gated_on"][0].__setitem__("artifact_field", "nested.value"),
        lambda value: value["tasks"][3]["gated_on"][0].__setitem__("artifact_field", "missing_field"),
        lambda value: value["tasks"][3]["gated_on"][0].__setitem__("upstream", YAML_ROWS[-1][0]),
        lambda value: value["tasks"][0].__setitem__("gated_on", deepcopy(value["tasks"][3]["gated_on"])),
        lambda value: value["tasks"][13].__setitem__("gated_on", deepcopy(value["tasks"][3]["gated_on"])),
    )
    for mutate in mutations:
        roadmap = _roadmap()
        mutate(roadmap)
        assert mod.evaluate_contract(MARKDOWN_CONTRACT, roadmap, set())["passed"] is False


def test_scenario_report_7151_policy_mutations_fail() -> None:
    """SCENARIO-REPORT-7151-DISCIPLINE rejects policy drift."""

    mutations: tuple[Callable[[dict[str, Any]], None], ...] = (
        lambda value: value["tasks"][0]["prior_failures"][0].__setitem__("experiment_id", ""),
        lambda value: value["tasks"][0]["prior_failures"][0].__setitem__("verdict", ""),
        lambda value: value["tasks"][0]["prior_failures"][0].__setitem__("addressed_by", ""),
        lambda value: value["tasks"][0]["prior_failures"][0].__setitem__("retire_if_same_verdict", False),
        lambda value: value["tasks"][0].__setitem__("agent_type", "gemini"),
        lambda value: value["tasks"][7].__setitem__("model", "gpt-5.5"),
        lambda value: value["tasks"][2].__setitem__("per_unit_rows", False),
        lambda value: value["tasks"][2].__setitem__("prompt", value["tasks"][2]["prompt"].replace("MODEL_SPECS", "model_specs")),
        lambda value: value["tasks"][12].__setitem__("operator_override", "too short"),
        lambda value: value["tasks"][13].pop("operator_override"),
        lambda value: value["tasks"][0].__setitem__("prompt", value["tasks"][0]["prompt"].replace(mod.PROMPT_FINAL_LINE, "")),
        lambda value: value["tasks"][0].__setitem__("prompt", value["tasks"][0]["prompt"].replace("test_experiment_7151_v629_contract_preflight.py", "wrong.py")),
        lambda value: value["tasks"][0].__setitem__("prompt", value["tasks"][0]["prompt"].replace("at least every 300 seconds", "periodically")),
        lambda value: value["tasks"][0].__setitem__("prompt", value["tasks"][0]["prompt"].replace("below 600 seconds", "small")),
    )
    for mutate in mutations:
        roadmap = _roadmap()
        mutate(roadmap)
        assert mod.evaluate_contract(MARKDOWN_CONTRACT, roadmap, set())["passed"] is False


def test_scenario_report_7151_terminal_artifacts_and_running_checkpoints(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-REPORT-7151-PREFLIGHT preserves checkpoints and terminal states."""

    output = _write_inputs(tmp_path)
    writes: list[dict[str, Any]] = []
    real_write = mod._write_artifact

    def record_write(path: Path, artifact: dict[str, Any]) -> None:
        writes.append(deepcopy(artifact))
        real_write(path, artifact)

    monkeypatch.setattr(mod, "_write_artifact", record_write)
    positive = mod.build_artifact(tmp_path, "20260909", output_path=output)
    assert positive["verdict_class"] == "positive"
    assert positive["v629_task_contract_conforms_score"] == 1
    assert writes[0]["status"] == "running"
    assert any(
        row["status"] == "running"
        and row["gate_check_summary"]["observed_value"] == "preconditions_complete"
        for row in writes
    )
    assert json.loads(output.read_text(encoding="utf-8")) == positive
    assert mod.validate_artifact(positive) == []

    (tmp_path / mod.ACTIVE_ROADMAP_PATH).unlink()
    blocked = mod.build_artifact(tmp_path, "20260909", output_path=output)
    assert blocked["status"] == "complete"
    assert blocked["verdict_class"] == "blocked"
    assert blocked["inference_substrate_class"] == "blocked_no_run"
    assert blocked["gate_check_summary"]["failed_check"] == "v629_active_yaml_readable"
    assert mod.validate_artifact(blocked) == []
    progress = capsys.readouterr().out
    assert "phase 0 start" in progress
    assert "phase 3 end" in progress


@pytest.mark.parametrize(
    ("field", "value", "expected_error"),
    (
        ("execution_venue", "gpu", "execution_venue_invalid"),
        ("verifier_is_oracle", True, "verifier_is_oracle_invalid"),
        ("v629_task_contract_conforms_score", 0, "v629_task_contract_conforms_score_invalid"),
        ("honest_verdict", "complete_positive_forged", "honest_verdict_not_derived"),
        ("reproducibility_checksum", "sha256:forged", "reproducibility_checksum_invalid"),
    ),
)
def test_scenario_report_7151_artifact_validator_rejects_forgery(
    tmp_path: Path, field: str, value: object, expected_error: str
) -> None:
    """SCENARIO-REPORT-7151-ARTIFACT rejects forged derived state."""

    output = _write_inputs(tmp_path)
    artifact = mod.build_artifact(tmp_path, "20260909", output_path=output)
    artifact[field] = value
    assert expected_error in mod.validate_artifact(artifact)


def test_req_report_7151_cli_validation_and_output(tmp_path: Path) -> None:
    """REQ-REPORT-7151 validates dates and writes only the selected output."""

    with pytest.raises(ValueError):
        mod.load_yaml(tmp_path / "missing.yaml")
    assert mod.main(["--date", "20260909", "--output", str(tmp_path / "result.json")]) == 0
    with pytest.raises(SystemExit):
        mod.main(["--date", "20260230", "--output", str(tmp_path / "bad.json")])
