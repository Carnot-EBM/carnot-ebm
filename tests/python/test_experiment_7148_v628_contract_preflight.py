"""Focused tests for the independent V628 roadmap contract preflight."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any, Callable

import pytest
import yaml

from carnot import experiment_7148_v628_contract_preflight as mod


ROOT = Path(__file__).resolve().parents[2]
YAML_ROWS = (
    (
        "exp7148-v628-contract-preflight",
        "V628 Markdown and YAML task-contract preflight",
        "results/experiment_7148_v628_contract_preflight.json",
    ),
    (
        "exp7149-v628-source-and-cache-delta",
        "V628 execution-time source and SOTA cache delta",
        "results/experiment_7149_v628_source_delta.json",
    ),
    (
        "exp7150-source-grounding-preflight-repair",
        "Source-grounding runtime and blinding preflight repair",
        "results/experiment_7150_v628_grounding_preflight.json",
    ),
    (
        "exp7151-qwen-symbolic-grounding-pilot",
        "Bounded Qwen symbolic-grounding intervention pilot",
        "results/experiment_7151_v628_qwen_grounding_pilot.json",
    ),
    (
        "exp7152-gemma-symbolic-grounding-replication",
        "Two-Gemma symbolic-grounding replication",
        "results/experiment_7152_v628_gemma_grounding_replication.json",
    ),
    (
        "exp7153-flowbalance-memory-prefix-learning",
        "Bounded chronological verifier-balanced memory learning",
        "results/experiment_7153_v628_flowbalance_memory_prefix.json",
    ),
    (
        "exp7154-frozen-memory-future-transfer",
        "Frozen-memory later-event transfer audit",
        "results/experiment_7154_v628_memory_future_transfer.json",
    ),
    (
        "exp7155-state-localized-arc-fixture",
        "State-localized ARC trace and same-state invariance fixture",
        "results/experiment_7155_v628_state_localized_arc_fixture.json",
    ),
    (
        "exp7156-live-arc-context-localization-ab",
        "Paired live ARC state-localized context comparison",
        "results/experiment_7156_v628_live_arc_context_ab.json",
    ),
    (
        "exp7157-rust-multiscale-exact-parity",
        "Rust multiscale sampler exact parity",
        "results/experiment_7157_v628_rust_multiscale_parity.json",
    ),
    (
        "exp7158-rust-multiscale-orchestration-profile",
        "Rust multiscale throughput and orchestration profile",
        "results/experiment_7158_v628_rust_orchestration_profile.json",
    ),
    (
        "exp7159-v628-capstone",
        "V628 capstone and evidence reconciliation",
        "results/experiment_7159_v628_capstone.json",
    ),
)
MARKDOWN_GATES = {
    7151: "exp7150-source-grounding-preflight-repair.grounding_preflight_ready_score == 1",
    7152: "exp7151-qwen-symbolic-grounding-pilot.qwen_grounding_pilot_complete_score == 1",
    7154: "exp7153-flowbalance-memory-prefix-learning.flowbalance_micro_csl_complete_score == 1",
    7156: "exp7155-state-localized-arc-fixture.state_localized_arc_fixture_ready_score == 1",
    7158: "exp7157-rust-multiscale-exact-parity.rust_multiscale_parity_score == 1",
}
GATES = {
    7151: (7150, "grounding_preflight_ready_score"),
    7152: (7151, "qwen_grounding_pilot_complete_score"),
    7154: (7153, "flowbalance_micro_csl_complete_score"),
    7156: (7155, "state_localized_arc_fixture_ready_score"),
    7158: (7157, "rust_multiscale_parity_score"),
}
PRIOR_IDS = {
    7148: ("exp7136-v627-contract-preflight",),
    7149: ("exp6461-v556-sota-source-and-benchmark-delta",),
    7150: ("exp7139-three-family-symbolic-grounding-ab",),
    7151: ("exp7139-three-family-symbolic-grounding-ab",),
    7152: ("exp7139-three-family-symbolic-grounding-ab",),
    7153: ("exp7142-flowbalance-external-memory-csl",),
    7154: ("exp7143-flowbalance-memory-cold-audit",),
    7155: ("exp7144-rebudgeted-adapter-withheld-arc-loo",),
    7156: ("exp7144-rebudgeted-adapter-withheld-arc-loo",),
    7157: ("exp7145-rust-multiscale-sampler-parity",),
    7158: ("exp7145-rust-multiscale-sampler-parity",),
    7159: ("exp7147-v627-capstone",),
}
MODEL_TASKS = {
    7150: ("unsloth/Qwen3.6-35B-A3B-GGUF",),
    7151: ("unsloth/Qwen3.6-35B-A3B-GGUF",),
    7152: (
        "unsloth/gemma-4-31B-it-GGUF",
        "unsloth/gemma-4-26B-A4B-it-GGUF",
    ),
    7153: ("unsloth/Qwen3.6-35B-A3B-GGUF",),
    7154: ("unsloth/Qwen3.6-35B-A3B-GGUF",),
    7156: ("unsloth/Qwen3.6-35B-A3B-GGUF",),
}
SUBSTRATE_CLASSES = {
    7148: "aggregation",
    7149: "aggregation",
    7150: "model_full_generation",
    7151: "model_full_generation",
    7152: "model_full_generation",
    7153: "model_full_generation",
    7154: "model_full_generation",
    7155: "no_model_load",
    7156: "model_full_generation",
    7157: "cpu_exact_solver_or_simulator",
    7158: "cpu_exact_solver_or_simulator",
    7159: "aggregation",
}
FOCUSED_TESTS = {
    number: f"tests/python/test_experiment_{number}_v628_{suffix}.py"
    for number, suffix in {
        7148: "contract_preflight",
        7149: "source_delta",
        7150: "grounding_preflight",
        7151: "qwen_grounding_pilot",
        7152: "gemma_grounding_replication",
        7153: "flowbalance_memory_prefix",
        7154: "memory_future_transfer",
        7155: "state_localized_arc_fixture",
        7156: "live_arc_context_ab",
        7157: "rust_multiscale_parity",
        7158: "rust_orchestration_profile",
        7159: "capstone",
    }.items()
}
RUN_LINES = {
    number: (
        "Run command: cd {project_root} && .venv/bin/python "
        f"scripts/experiments/experiment_{number}_v628_{suffix}.py --date {{date}}"
    )
    for number, suffix in {
        7148: "contract_preflight",
        7149: "source_delta",
        7150: "grounding_preflight",
        7151: "qwen_grounding_pilot",
        7152: "gemma_grounding_replication",
        7153: "flowbalance_memory_prefix",
        7154: "memory_future_transfer",
        7155: "state_localized_arc_fixture",
        7156: "live_arc_context_ab",
        7157: "rust_multiscale_parity",
        7158: "rust_orchestration_profile",
        7159: "capstone",
    }.items()
}
MARKDOWN_CONTRACT = "\n".join(
    (
        "# Independent fixture",
        "",
        "**Milestone:** `2026.09.628`",
        "",
        "## Exact task contract",
        "",
        "| Order | Task ID | Title | Deliverable | Structured gate |",
        "|---:|---|---|---|---|",
        *(
            f"| {order} | `{task_id}` | {title} | `{deliverable}` | "
            f"{('`' + MARKDOWN_GATES[number] + '`') if number in MARKDOWN_GATES else 'none'} |"
            for order, (task_id, title, deliverable) in enumerate(YAML_ROWS, start=1)
            for number in (int(task_id[3:7]),)
        ),
    )
)


def _required_fields(number: int) -> list[str]:
    """Build fixture fields without reading the Markdown table."""

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
    if producer_field is not None:
        fields.append(producer_field)
    if number == 7155:
        fields.extend(("solve_provenance (development_proxy)", "game_level_solve_claimed (false)"))
    if number == 7156:
        fields.extend(
            (
                "solve_provenance (live_agent_self_discovery)",
                "target_adapter_used (false)",
                "game_source_used (false)",
                "offline_bfs_used (false)",
                "per_game_calibration_used (false)",
            )
        )
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
    """Create an execution prompt independently from the Markdown fixture."""

    model_lines = "".join(f"MODEL_SPECS must include {model}.\n" for model in MODEL_TASKS.get(number, ()))
    return (
        "CONCRETE STEPS:\n"
        "0. Create a schema-complete running artifact before any check.\n"
        "1. Print a flushed progress line at every numbered phase boundary. Print flushed start "
        "and end lines before and after every model load, generation, benchmark, and subprocess. "
        "Print a flushed line at least once every 300 seconds inside long loops. Keep every stdout "
        "gap below 600 seconds.\n"
        "2. Run the bounded task.\n"
        f"Add focused RED tests in {FOCUSED_TESTS[number]}.\n"
        + model_lines
        + "REQUIRED ARTIFACT FIELDS: "
        + "; ".join(_required_fields(number))
        + ".\n\n"
        + RUN_LINES[number]
        + "\n"
        + mod.PROMPT_FINAL_LINE
        + "\n"
    )


def _roadmap() -> dict[str, Any]:
    """Build YAML data without parsing or copying the Markdown fixture."""

    tasks: list[dict[str, Any]] = []
    for task_id, title, deliverable in YAML_ROWS:
        number = int(task_id[3:7])
        task: dict[str, Any] = {
            "id": task_id,
            "title": title,
            "track": "arc" if number in {7155, 7156} else "infrastructure",
            "priority": "high",
            "requires_gpu": number in MODEL_TASKS,
            "max_turns": 50,
            "estimated_wall_time_min": 45,
            "milestone": "2026.09.628",
            "deliverable": deliverable,
            "per_unit_rows": True,
            "prior_failures": [
                {
                    "experiment_id": prior_id,
                    "verdict": "complete_prior_result",
                    "addressed_by": "V628 changes the method or execution contract.",
                    "retire_if_same_verdict": True,
                }
                for prior_id in PRIOR_IDS[number]
            ],
            "prompt": _prompt(number),
        }
        if number == 7148:
            task["model"] = "opus"
        elif number not in {7149, 7159}:
            task["agent_type"] = "codex"
            task["model"] = "gpt-5.6-sol"
        if number in GATES:
            upstream_number, field = GATES[number]
            upstream_id = next(row[0] for row in YAML_ROWS if row[0].startswith(f"exp{upstream_number}-"))
            task["gated_on"] = [
                {"upstream": upstream_id, "artifact_field": field, "op": "==", "value": 1}
            ]
        tasks.append(task)
    return {
        "milestone": "2026.09.628",
        "milestone_title": "fixture",
        "milestone_doc": "fixture.md",
        "tasks": tasks,
    }


def _write_inputs(root: Path, roadmap: dict[str, Any] | None = None) -> Path:
    """Write temporary prerequisites so tests never mutate project records."""

    for relative in (mod.DESIGN_PATH, mod.NEXT_ROADMAP_PATH, mod.EXCLUSION_PATH):
        (root / relative).parent.mkdir(parents=True, exist_ok=True)
    (root / mod.DESIGN_PATH).write_text(MARKDOWN_CONTRACT, encoding="utf-8")
    (root / mod.NEXT_ROADMAP_PATH).write_text(
        yaml.safe_dump(roadmap or _roadmap(), sort_keys=False), encoding="utf-8"
    )
    (root / mod.EXCLUSION_PATH).write_text("retired: []\n", encoding="utf-8")
    return root / mod.DEFAULT_OUTPUT_PATH


def test_req_report_7148_spec_defines_contract_and_scenarios() -> None:
    """REQ-REPORT-7148 names every ID, output field, and scenario."""

    text = (ROOT / mod.SPEC_PATH).read_text(encoding="utf-8")
    section = text[text.index("### REQ-REPORT-7148") :]
    for task_id, _title, _deliverable in YAML_ROWS:
        assert task_id in section
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert field in section
    for name in ("PARITY", "GATES", "DISCIPLINE", "PREFLIGHT", "ARTIFACT"):
        assert f"SCENARIO-REPORT-7148-{name}" in section


def test_scenario_report_7148_parity_accepts_independent_rows() -> None:
    """SCENARIO-REPORT-7148-PARITY accepts separate matching sources."""

    result = mod.evaluate_contract(MARKDOWN_CONTRACT, _roadmap(), set())
    assert result["passed"] is True
    assert len(result["task_contract_rows"]) == 12
    assert len(result["gate_producer_rows"]) == 5
    assert len(result["prior_failure_rows"]) == 12
    assert result["markdown_task_rows"] is not result["yaml_task_rows"]
    assert result["markdown_task_rows"][3]["gates"][0]["upstream"] == YAML_ROWS[2][0]


@pytest.mark.parametrize(
    "case", ("missing", "extra", "reordered", "duplicated", "renamed", "title", "deliverable")
)
def test_req_report_7148_row_mutations_fail(case: str) -> None:
    """REQ-REPORT-7148 rejects all row identity and parity mutations."""

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
        roadmap["tasks"][0]["id"] += "-changed"
    elif case == "title":
        roadmap["tasks"][0]["title"] += " changed"
    else:
        roadmap["tasks"][0]["deliverable"] = "results/wrong.json"
    assert mod.evaluate_contract(MARKDOWN_CONTRACT, roadmap, set())["passed"] is False


def test_scenario_report_7148_malformed_sources_fail() -> None:
    """SCENARIO-REPORT-7148-PARITY rejects malformed source syntax."""

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


def test_scenario_report_7148_gate_mutations_fail() -> None:
    """SCENARIO-REPORT-7148-GATES rejects wrong or unusable producers."""

    mutations: tuple[Callable[[dict[str, Any]], None], ...] = (
        lambda value: value["tasks"][3]["gated_on"][0].__setitem__("artifact_field", "nested.value"),
        lambda value: value["tasks"][3]["gated_on"][0].__setitem__("artifact_field", "missing_field"),
        lambda value: value["tasks"][3]["gated_on"][0].__setitem__("upstream", YAML_ROWS[-1][0]),
        lambda value: value["tasks"][0].__setitem__("gated_on", deepcopy(value["tasks"][3]["gated_on"])),
        lambda value: value["tasks"][11].__setitem__("gated_on", deepcopy(value["tasks"][3]["gated_on"])),
    )
    for mutate in mutations:
        roadmap = _roadmap()
        mutate(roadmap)
        assert mod.evaluate_contract(MARKDOWN_CONTRACT, roadmap, set())["passed"] is False


def test_scenario_report_7148_policy_mutations_fail() -> None:
    """SCENARIO-REPORT-7148-DISCIPLINE rejects policy drift."""

    mutations: tuple[Callable[[dict[str, Any]], None], ...] = (
        lambda value: value["tasks"][0]["prior_failures"][0].__setitem__("experiment_id", ""),
        lambda value: value["tasks"][0]["prior_failures"][0].__setitem__("verdict", ""),
        lambda value: value["tasks"][0]["prior_failures"][0].__setitem__("addressed_by", ""),
        lambda value: value["tasks"][0]["prior_failures"][0].__setitem__("retire_if_same_verdict", False),
        lambda value: value["tasks"][0].__setitem__("agent_type", "gemini"),
        lambda value: value["tasks"][2].__setitem__("model", "gpt-5.5"),
        lambda value: value["tasks"][2].__setitem__("per_unit_rows", False),
        lambda value: value["tasks"][0].__setitem__("prompt", value["tasks"][0]["prompt"].replace("run_date", "date")),
        lambda value: value["tasks"][2].__setitem__("prompt", value["tasks"][2]["prompt"].replace("MODEL_SPECS", "model_specs")),
        lambda value: value["tasks"][8].__setitem__("prompt", value["tasks"][8]["prompt"].replace("live_agent_self_discovery", "development_proxy")),
        lambda value: value["tasks"][0].__setitem__("prompt", value["tasks"][0]["prompt"].replace(mod.PROMPT_FINAL_LINE, "")),
        lambda value: value["tasks"][0].__setitem__("prompt", value["tasks"][0]["prompt"].replace(FOCUSED_TESTS[7148], "tests/python/wrong.py")),
        lambda value: value["tasks"][0].__setitem__("prompt", value["tasks"][0]["prompt"].replace("at least once every 300 seconds", "periodically")),
        lambda value: value["tasks"][0].__setitem__("prompt", value["tasks"][0]["prompt"].replace("below 600 seconds", "small")),
    )
    for mutate in mutations:
        roadmap = _roadmap()
        mutate(roadmap)
        assert mod.evaluate_contract(MARKDOWN_CONTRACT, roadmap, set())["passed"] is False


def test_scenario_report_7148_blocked_disqualified_and_positive_artifacts(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """SCENARIO-REPORT-7148-PREFLIGHT preserves all terminal states."""

    output = _write_inputs(tmp_path)
    (tmp_path / mod.NEXT_ROADMAP_PATH).unlink()
    blocked = mod.build_artifact(tmp_path, "20260908", output_path=output)
    assert blocked["status"] == "complete"
    assert blocked["verdict_class"] == "blocked"
    assert blocked["inference_substrate_class"] == "blocked_no_run"
    assert blocked["gate_check_summary"]["failed_check"] == "v628_yaml_readable"
    assert mod.validate_artifact(blocked) == []

    changed = _roadmap()
    changed["tasks"][0]["title"] += " changed"
    output = _write_inputs(tmp_path, changed)
    disqualified = mod.build_artifact(tmp_path, "20260908", output_path=output)
    assert disqualified["verdict_class"] == "disqualified"
    assert disqualified["v628_task_contract_conforms_score"] == 0
    assert mod.validate_artifact(disqualified) == []

    output = _write_inputs(tmp_path)
    positive = mod.build_artifact(tmp_path, "20260908", output_path=output)
    assert positive["verdict_class"] == "positive"
    assert positive["v628_task_contract_conforms_score"] == 1
    assert mod.validate_artifact(positive) == []
    assert json.loads(output.read_text(encoding="utf-8")) == positive
    progress = capsys.readouterr().out
    assert "phase 0 start" in progress
    assert "phase 3 end" in progress


@pytest.mark.parametrize(
    ("field", "value", "expected_error"),
    (
        ("execution_venue", "gpu", "execution_venue_invalid"),
        ("verifier_is_oracle", True, "verifier_is_oracle_invalid"),
        ("v628_task_contract_conforms_score", 0, "v628_task_contract_conforms_score_invalid"),
        ("honest_verdict", "complete_positive_forged", "honest_verdict_not_derived"),
        ("reproducibility_checksum", "sha256:forged", "reproducibility_checksum_invalid"),
    ),
)
def test_scenario_report_7148_artifact_validator_rejects_forgery(
    tmp_path: Path, field: str, value: object, expected_error: str
) -> None:
    """SCENARIO-REPORT-7148-ARTIFACT rejects forged derived state."""

    output = _write_inputs(tmp_path)
    artifact = mod.build_artifact(tmp_path, "20260908", output_path=output)
    artifact[field] = value
    assert expected_error in mod.validate_artifact(artifact)


def test_req_report_7148_cli_validation_and_output(tmp_path: Path) -> None:
    """REQ-REPORT-7148 validates dates and writes only the selected output."""

    with pytest.raises(ValueError):
        mod.load_yaml(tmp_path / "missing.yaml")
    assert mod.main(["--date", "20260908", "--output", str(tmp_path / "result.json")]) == 0
    with pytest.raises(SystemExit):
        mod.main(["--date", "20260230", "--output", str(tmp_path / "bad.json")])
