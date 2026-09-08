"""Tests for REQ-REPORT-7136, the independent V627 contract preflight."""

from __future__ import annotations

from copy import deepcopy
import importlib.util
import json
from pathlib import Path
from typing import Any, Callable

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = ROOT / "python/carnot/experiment_7136_v627_contract_preflight.py"
MODULE_SPEC = importlib.util.spec_from_file_location(
    "experiment_7136_v627_contract_preflight", MODULE_PATH
)
assert MODULE_SPEC is not None and MODULE_SPEC.loader is not None
mod = importlib.util.module_from_spec(MODULE_SPEC)
MODULE_SPEC.loader.exec_module(mod)

MARKDOWN_CONTRACT = """# V627 design

**Milestone:** `2026.09.627`

## Exact task contract

| Order | Task ID | Title | Deliverable | Structured gate |
|---:|---|---|---|---|
| 1 | `exp7136-v627-contract-preflight` | V627 Markdown and YAML task-contract preflight | `results/experiment_7136_v627_contract_preflight.json` | none |
| 2 | `exp7137-v627-source-and-cache-delta` | V627 execution-time source and SOTA cache delta | `results/experiment_7137_v627_source_delta.json` | none |
| 3 | `exp7138-source-grounded-relational-fixture` | Source-grounded relational hallucination fixture | `results/experiment_7138_v627_relational_fixture.json` | none |
| 4 | `exp7139-three-family-symbolic-grounding-ab` | Three-family symbolic grounding comparison | `results/experiment_7139_v627_symbolic_grounding_ab.json` | `exp7138-source-grounded-relational-fixture.source_grounding_fixture_ready_score == 1` |
| 5 | `exp7140-symbolic-intervention-causal-audit` | Independent symbolic intervention and causal audit | `results/experiment_7140_v627_symbolic_intervention_audit.json` | `exp7139-three-family-symbolic-grounding-ab.symbolic_grounding_complete_score == 1` |
| 6 | `exp7141-v626-chronological-csl-stream` | Immutable V626 chronological self-learning stream | `results/experiment_7141_v627_csl_event_stream.json` | none |
| 7 | `exp7142-flowbalance-external-memory-csl` | Verifier-balanced external-memory continuous self-learning | `results/experiment_7142_v627_flowbalance_memory_csl.json` | `exp7141-v626-chronological-csl-stream.csl_event_stream_ready_score == 1` |
| 8 | `exp7143-flowbalance-memory-cold-audit` | Cold retention and negative-transfer audit | `results/experiment_7143_v627_flowbalance_memory_cold_audit.json` | `exp7142-flowbalance-external-memory-csl.flowbalance_memory_csl_complete_score == 1` |
| 9 | `exp7144-rebudgeted-adapter-withheld-arc-loo` | Rebudgeted adapter-withheld ARC LOO cell | `results/experiment_7144_v627_rebudgeted_arc_loo.json` | none |
| 10 | `exp7145-rust-multiscale-sampler-parity` | Rust multiscale sampler parity and throughput | `results/experiment_7145_v627_rust_multiscale_sampler.json` | none |
| 11 | `exp7146-gatemate-changed-state-continuity` | GateMate changed-state continuity with one-action stop | `results/experiment_7146_v627_gatemate_changed_state.json` | none |
| 12 | `exp7147-v627-capstone` | V627 independent evidence matrix and branch disposition | `results/experiment_7147_v627_capstone.json` | none |
"""

YAML_ROWS = (
    ("exp7136-v627-contract-preflight", "V627 Markdown and YAML task-contract preflight", "results/experiment_7136_v627_contract_preflight.json"),
    ("exp7137-v627-source-and-cache-delta", "V627 execution-time source and SOTA cache delta", "results/experiment_7137_v627_source_delta.json"),
    ("exp7138-source-grounded-relational-fixture", "Source-grounded relational hallucination fixture", "results/experiment_7138_v627_relational_fixture.json"),
    ("exp7139-three-family-symbolic-grounding-ab", "Three-family symbolic grounding comparison", "results/experiment_7139_v627_symbolic_grounding_ab.json"),
    ("exp7140-symbolic-intervention-causal-audit", "Independent symbolic intervention and causal audit", "results/experiment_7140_v627_symbolic_intervention_audit.json"),
    ("exp7141-v626-chronological-csl-stream", "Immutable V626 chronological self-learning stream", "results/experiment_7141_v627_csl_event_stream.json"),
    ("exp7142-flowbalance-external-memory-csl", "Verifier-balanced external-memory continuous self-learning", "results/experiment_7142_v627_flowbalance_memory_csl.json"),
    ("exp7143-flowbalance-memory-cold-audit", "Cold retention and negative-transfer audit", "results/experiment_7143_v627_flowbalance_memory_cold_audit.json"),
    ("exp7144-rebudgeted-adapter-withheld-arc-loo", "Rebudgeted adapter-withheld ARC LOO cell", "results/experiment_7144_v627_rebudgeted_arc_loo.json"),
    ("exp7145-rust-multiscale-sampler-parity", "Rust multiscale sampler parity and throughput", "results/experiment_7145_v627_rust_multiscale_sampler.json"),
    ("exp7146-gatemate-changed-state-continuity", "GateMate changed-state continuity with one-action stop", "results/experiment_7146_v627_gatemate_changed_state.json"),
    ("exp7147-v627-capstone", "V627 independent evidence matrix and branch disposition", "results/experiment_7147_v627_capstone.json"),
)

GATES = {
    7139: (7138, "source_grounding_fixture_ready_score"),
    7140: (7139, "symbolic_grounding_complete_score"),
    7142: (7141, "csl_event_stream_ready_score"),
    7143: (7142, "flowbalance_memory_csl_complete_score"),
}
PRIOR_IDS = {
    7136: ("exp7109-v624-contract-preflight", "exp7121-v625-contract-preflight"),
    7137: ("exp6461-v556-sota-source-and-benchmark-delta",),
    7138: ("exp6984-exact-contrast-fixture",),
    7139: ("exp3670-facts-row-real-benchmark", "exp5163-mmlu-pro-verifier-rescale-v473"),
    7140: ("exp3670-facts-row-real-benchmark",),
    7141: ("exp6265-chronological-two-timescale-csl-ab", "exp6277-chronological-certified-csl-ab"),
    7142: ("exp6978-transactional-constraint-self-learning", "exp7131-model-facing-fixed-schema-csl"),
    7143: ("exp6963-queue-memory-cold-audit",),
    7144: ("exp7127-adapter-withheld-arc-loo-cell", "exp7128-arc-loo-causal-provenance-audit"),
    7145: ("exp5714-one-axis-tempering-rust-parity",),
    7146: ("exp6525-gatemate-changed-state-continuity", "exp6559-gatemate-changed-state-continuity"),
    7147: ("exp6922-v605-independent-capstone",),
}
FOCUSED_TESTS = {
    7136: "tests/python/test_experiment_7136_v627_contract_preflight.py",
    7137: "tests/python/test_experiment_7137_v627_source_delta.py",
    7138: "tests/python/test_experiment_7138_v627_relational_fixture.py",
    7139: "tests/python/test_experiment_7139_v627_symbolic_grounding_ab.py",
    7140: "tests/python/test_experiment_7140_v627_symbolic_intervention_audit.py",
    7141: "tests/python/test_experiment_7141_v627_csl_event_stream.py",
    7142: "tests/python/test_experiment_7142_v627_flowbalance_memory_csl.py",
    7143: "tests/python/test_experiment_7143_v627_flowbalance_memory_cold_audit.py",
    7144: "tests/python/test_experiment_7127_v626_adapter_withheld_arc_loo.py",
    7145: "tests/python/test_experiment_7145_v627_rust_multiscale_sampler.py",
    7146: "tests/python/test_experiment_7146_v627_gatemate_changed_state.py",
    7147: "tests/python/test_experiment_7147_v627_capstone.py",
}
RUN_LINES = {
    7136: "Run command: cd {project_root} && .venv/bin/python scripts/experiments/experiment_7136_v627_contract_preflight.py --date {date}",
    7137: "Run command: cd {project_root} && .venv/bin/python scripts/experiments/experiment_7137_v627_source_delta.py --date {date}",
    7138: "Run command: cd {project_root} && .venv/bin/python scripts/experiments/experiment_7138_v627_relational_fixture.py --date {date}",
    7139: "Run command: cd {project_root} && .venv/bin/python scripts/experiments/experiment_7139_v627_symbolic_grounding_ab.py --date {date}",
    7140: "Run command: cd {project_root} && .venv/bin/python scripts/experiments/experiment_7140_v627_symbolic_intervention_audit.py --date {date}",
    7141: "Run command: cd {project_root} && .venv/bin/python scripts/experiments/experiment_7141_v627_csl_event_stream.py --date {date}",
    7142: "Run command: cd {project_root} && .venv/bin/python scripts/experiments/experiment_7142_v627_flowbalance_memory_csl.py --date {date}",
    7143: "Run command: cd {project_root} && .venv/bin/python scripts/experiments/experiment_7143_v627_flowbalance_memory_cold_audit.py --date {date}",
    7145: "Run command: cd {project_root} && .venv/bin/python scripts/experiments/experiment_7145_v627_rust_multiscale_sampler.py --date {date}",
    7146: "Run command: cd {project_root} && .venv/bin/python scripts/experiments/experiment_7146_v627_gatemate_changed_state.py --date {date}",
    7147: "Run command: cd {project_root} && .venv/bin/python scripts/experiments/experiment_7147_v627_capstone.py --date {date}",
}
RUN_LINES[7144] = (
    "Run command: cd {project_root} && PYTHONPATH=python .venv/bin/python "
    "python/carnot/experiment_7127_v626_adapter_withheld_arc_loo.py --date {date} "
    "--output results/experiment_7144_v627_rebudgeted_arc_loo.json --raw-root "
    "/tmp/carnot-exp7144-v627-rebudgeted-arc-loo --actions 25 "
    "--generation-max-tokens 1024"
)


def _prompt(number: int) -> str:
    """Build a YAML prompt without consulting the Markdown table fixture."""

    planned_class = {
        7139: "model_full_generation",
        7142: "model_full_generation",
        7144: "model_full_generation",
        7138: "cpu_exact_solver_or_simulator",
        7145: "cpu_exact_solver_or_simulator",
    }.get(number, "no_model_load" if number not in {7136, 7137, 7147} else "aggregation")
    fields = [
        "field_principles for every field below",
        "preconditions_checked",
        "run_date",
        "inference_substrate",
        f"inference_substrate_class ({planned_class} or blocked_no_run)",
        "execution_venue (host)",
        "duration_s",
        "source_artifact_hashes",
    ]
    if number != 7146:
        fields.append("rows")
    if number in {7139, 7142, 7144}:
        fields.append("MODEL_SPECS")
    if number in {7138, 7139, 7141, 7142}:
        fields.append(GATES.get(number + 1, (None, "unused"))[1])
    if number == 7144:
        fields.extend(
            (
                "per_game_results",
                "arc_loop_solve (false)",
                "solve_provenance (development_proxy)",
                "game_level_solve_claimed (false)",
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
    models = ""
    if number == 7139:
        models = "MODEL_SPECS must contain " + ", ".join(mod.MANDATED_MODELS) + ". Run all models.\n"
    elif number in {7142, 7144}:
        models = f"MODEL_SPECS must include {mod.MANDATED_MODELS[0]}. Run the model.\n"
    return (
        "TASK:\nCompare or audit preserved rows.\n"
        f"Add focused RED tests in {FOCUSED_TESTS[number]}.\n"
        + models
        + "REQUIRED ARTIFACT FIELDS: "
        + "; ".join(fields)
        + ".\n\n"
        + RUN_LINES[number]
        + "\n"
        + mod.PROMPT_FINAL_LINE
        + "\n"
    )


def _roadmap() -> dict[str, Any]:
    """Build YAML data from a fixture separate from the Markdown string."""

    tasks: list[dict[str, Any]] = []
    for task_id, title, deliverable in YAML_ROWS:
        number = int(task_id[3:7])
        task: dict[str, Any] = {
            "id": task_id,
            "title": title,
            "track": "arc" if number == 7144 else "infrastructure",
            "priority": "high",
            "requires_gpu": number in {7139, 7142, 7144},
            "max_turns": 50,
            "estimated_wall_time_min": 45,
            "milestone": "2026.09.627",
            "deliverable": deliverable,
            "prior_failures": [
                {
                    "experiment_id": prior_id,
                    "verdict": "complete_prior_result",
                    "addressed_by": "V627 changes the method or input contract.",
                    "retire_if_same_verdict": True,
                }
                for prior_id in PRIOR_IDS[number]
            ],
            "prompt": _prompt(number),
        }
        if number != 7146:
            task["per_unit_rows"] = True
        if number in {7136, 7146}:
            task["model"] = "opus"
        elif number not in {7137, 7147}:
            task["agent_type"] = "codex"
            task["model"] = "gpt-5.6-sol"
        if number in GATES:
            upstream_number, field = GATES[number]
            upstream_id = next(row[0] for row in YAML_ROWS if row[0].startswith(f"exp{upstream_number}-"))
            task["gated_on"] = [{"upstream": upstream_id, "artifact_field": field, "op": "==", "value": 1}]
        tasks.append(task)
    return {
        "milestone": "2026.09.627",
        "milestone_title": "fixture",
        "milestone_doc": "fixture.md",
        "tasks": tasks,
    }


def _write_inputs(root: Path, roadmap: dict[str, Any] | None = None) -> Path:
    """Write only temporary prerequisites used by artifact tests."""

    design = root / mod.DESIGN_PATH
    active = root / mod.ACTIVE_ROADMAP_PATH
    exclusion = root / mod.EXCLUSION_PATH
    for path in (design, active, exclusion):
        path.parent.mkdir(parents=True, exist_ok=True)
    design.write_text(MARKDOWN_CONTRACT, encoding="utf-8")
    active.write_text(yaml.safe_dump(roadmap or _roadmap(), sort_keys=False), encoding="utf-8")
    exclusion.write_text("retired: []\n", encoding="utf-8")
    return root / mod.DEFAULT_OUTPUT_PATH


def test_req_report_7136_spec_defines_contract_and_scenarios() -> None:
    """REQ-REPORT-7136 names all IDs, fields, and replay scenarios."""

    text = (ROOT / mod.SPEC_PATH).read_text(encoding="utf-8")
    section = text[text.index("### REQ-REPORT-7136") :]
    for task_id, _title, _deliverable in YAML_ROWS:
        assert task_id in section
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert field in section
    for name in ("PARITY", "GATES", "DISCIPLINE", "PREFLIGHT", "ARTIFACT"):
        assert f"SCENARIO-REPORT-7136-{name}" in section


def test_scenario_report_7136_parity_accepts_independent_rows() -> None:
    """SCENARIO-REPORT-7136-PARITY accepts separate 12-row sources."""

    result = mod.evaluate_contract(MARKDOWN_CONTRACT, _roadmap(), set())
    assert result["passed"] is True
    assert len(result["task_contract_rows"]) == 12
    assert len(result["gate_producer_rows"]) == 4
    assert len(result["prior_failure_rows"]) == 18
    assert result["markdown_task_rows"] is not result["yaml_task_rows"]
    assert result["markdown_task_rows"][3]["gates"][0]["upstream"] == YAML_ROWS[2][0]


@pytest.mark.parametrize(
    "case", ("missing", "extra", "reordered", "duplicated", "renamed", "title", "deliverable")
)
def test_req_report_7136_row_mutations_fail(case: str) -> None:
    """REQ-REPORT-7136 rejects every row identity and parity mutation."""

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


def test_scenario_report_7136_malformed_sources_fail() -> None:
    """SCENARIO-REPORT-7136-PARITY rejects malformed source syntax."""

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


def test_scenario_report_7136_gate_mutations_fail() -> None:
    """SCENARIO-REPORT-7136-GATES rejects wrong or unusable producers."""

    mutations: tuple[Callable[[dict[str, Any]], None], ...] = (
        lambda value: value["tasks"][3]["gated_on"][0].__setitem__("artifact_field", "nested.value"),
        lambda value: value["tasks"][3]["gated_on"][0].__setitem__("artifact_field", "missing_field"),
        lambda value: value["tasks"][3]["gated_on"][0].__setitem__("upstream", YAML_ROWS[-1][0]),
        lambda value: value["tasks"][0].__setitem__("gated_on", deepcopy(value["tasks"][3]["gated_on"])),
        lambda value: value["tasks"][8].__setitem__("gated_on", deepcopy(value["tasks"][3]["gated_on"])),
        lambda value: value["tasks"][8].__setitem__("requires", [YAML_ROWS[7][0]]),
    )
    for mutate in mutations:
        roadmap = _roadmap()
        mutate(roadmap)
        assert mod.evaluate_contract(MARKDOWN_CONTRACT, roadmap, set())["passed"] is False


def test_scenario_report_7136_policy_mutations_fail() -> None:
    """SCENARIO-REPORT-7136-DISCIPLINE rejects route and prompt drift."""

    mutations: tuple[Callable[[dict[str, Any]], None], ...] = (
        lambda value: value["tasks"][0]["prior_failures"][0].__setitem__("addressed_by", ""),
        lambda value: value["tasks"][0]["prior_failures"][0].__setitem__("retire_if_same_verdict", False),
        lambda value: value["tasks"][0].__setitem__("agent_type", "gemini"),
        lambda value: value["tasks"][2].__setitem__("model", "gpt-5.5"),
        lambda value: value["tasks"][2].__setitem__("per_unit_rows", False),
        lambda value: value["tasks"][0].__setitem__("prompt", value["tasks"][0]["prompt"].replace("run_date", "date")),
        lambda value: value["tasks"][3].__setitem__("prompt", value["tasks"][3]["prompt"].replace("MODEL_SPECS", "model_specs")),
        lambda value: value["tasks"][8].__setitem__("prompt", value["tasks"][8]["prompt"].replace("development_proxy", "offline_reproduced")),
        lambda value: value["tasks"][0].__setitem__("prompt", value["tasks"][0]["prompt"].replace(mod.PROMPT_FINAL_LINE, "")),
        lambda value: value["tasks"][0].__setitem__("prompt", value["tasks"][0]["prompt"].replace(FOCUSED_TESTS[7136], "tests/python/wrong.py")),
    )
    for mutate in mutations:
        roadmap = _roadmap()
        mutate(roadmap)
        assert mod.evaluate_contract(MARKDOWN_CONTRACT, roadmap, set())["passed"] is False


def test_scenario_report_7136_blocked_and_disqualified_artifacts(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7136-PREFLIGHT preserves blocked and mismatch states."""

    output = _write_inputs(tmp_path)
    (tmp_path / mod.DESIGN_PATH).unlink()
    blocked = mod.build_artifact(tmp_path, "20260908", output_path=output)
    assert blocked["status"] == "complete"
    assert blocked["verdict_class"] == "blocked"
    assert blocked["inference_substrate_class"] == "blocked_no_run"
    assert blocked["gate_check_summary"]["failed_check"] == "v627_markdown_readable"
    assert mod.validate_artifact(blocked) == []

    output = _write_inputs(tmp_path)
    roadmap = _roadmap()
    roadmap["tasks"][0]["title"] += " changed"
    (tmp_path / mod.ACTIVE_ROADMAP_PATH).write_text(
        yaml.safe_dump(roadmap, sort_keys=False), encoding="utf-8"
    )
    disqualified = mod.build_artifact(tmp_path, "20260908", output_path=output)
    assert disqualified["verdict_class"] == "disqualified"
    assert disqualified["v627_task_contract_conforms_score"] == 0
    assert mod.validate_artifact(disqualified) == []


def test_scenario_report_7136_artifact_recomputes_and_rejects_tampering(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7136-ARTIFACT recomputes the score and checksum."""

    output = _write_inputs(tmp_path)
    artifact = mod.build_artifact(tmp_path, "20260908", output_path=output)
    assert json.loads(output.read_text(encoding="utf-8"))["status"] == "complete"
    assert artifact["v627_task_contract_conforms_score"] == 1
    assert artifact["verdict_class"] == "positive"
    assert mod.validate_artifact(artifact) == []
    for field, value, expected_error in (
        ("execution_venue", "container", "execution_venue_invalid"),
        ("v627_task_contract_conforms_score", 0, "v627_task_contract_conforms_score_invalid"),
        ("verdict_class", "partial", "verdict_class_not_derived"),
    ):
        changed = deepcopy(artifact)
        changed[field] = value
        changed["reproducibility_checksum"] = mod.reproducibility_checksum(changed)
        assert expected_error in mod.validate_artifact(changed)
    changed = deepcopy(artifact)
    changed["task_contract_rows"][0]["passed"] = False
    assert "v627_task_contract_conforms_score_invalid" in mod.validate_artifact(changed)
    assert "reproducibility_checksum_invalid" in mod.validate_artifact(changed)


def test_req_report_7136_validation_rejects_non_artifacts() -> None:
    """REQ-REPORT-7136 rejects missing fields and invalid dates."""

    assert mod.validate_artifact([]) == ["artifact_mapping_required"]
    assert mod.validate_artifact({}) == ["missing_required_field:agent_routing_rows"]
    with pytest.raises(ValueError):
        mod.parse_yaml_contract({"tasks": [{"id": "exp1", "prompt": "x", "gated_on": ["bad"]}]})


def test_req_report_7136_parser_defenses_cover_invalid_shapes(tmp_path: Path) -> None:
    """REQ-REPORT-7136 rejects hidden structures and malformed YAML values."""

    with pytest.raises(ValueError):
        mod._parse_scalar("[1]")
    with pytest.raises(ValueError):
        mod.parse_markdown_contract(
            "**Milestone:** 2026.09.627\n\n## Exact task contract\n\n"
            "| Order | Task ID | Title | Deliverable | Structured gate |\n"
            "|---:|---|---|---|---|\n"
        )
    for bad_row in (
        "| 1 | `bad-id` | title | `results/x.json` | none |",
        "| 1 | `exp7136-x` |  | `results/x.json` | none |",
        "| 1 | `exp7136-x` | title |  | none |",
    ):
        text = (
            "**Milestone:** 2026.09.627\n\n## Exact task contract\n\n"
            "| Order | Task ID | Title | Deliverable | Structured gate |\n"
            "|---:|---|---|---|---|\n"
            + bad_row
        )
        with pytest.raises(ValueError):
            mod.parse_markdown_contract(text)
    assert mod._required_block("ordinary prose") == ""
    for task in (
        {"id": 1, "prompt": "x"},
        {"id": "exp1", "prompt": "x", "prior_failures": "bad"},
    ):
        with pytest.raises(ValueError):
            mod.parse_yaml_contract({"tasks": [task]})
    empty = tmp_path / "empty.yaml"
    empty.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError):
        mod.load_yaml(empty)


def test_req_report_7136_manifest_and_empty_prior_helpers() -> None:
    """REQ-REPORT-7136 normalizes retired IDs and fails closed on no lineage."""

    assert mod.retired_experiment_ids([]) == set()
    manifest = {
        "metadata": ["ignored"],
        "retired": [
            "ignored",
            {
                "experiment_id": 7136,
                "experiment_ids": ["exp7137-x", "bad"],
                "un_retired_experiment_ids": ["exp7137-x"],
            },
        ],
    }
    assert mod.retired_experiment_ids(manifest) == {"exp7136"}
    task = {
        "order": 1,
        "number": 9999,
        "id": "exp9999-fixture",
        "prior_failures": [],
    }
    assert mod._prior_rows(task)[0]["passed"] is False


def test_req_report_7136_precondition_and_parse_failures(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """SCENARIO-REPORT-7136-PREFLIGHT records YAML and parse failures."""

    output = _write_inputs(tmp_path)
    (tmp_path / mod.ACTIVE_ROADMAP_PATH).write_text("[]\n", encoding="utf-8")
    rows, roadmap, exclusion = mod._preconditions(tmp_path, output)
    assert roadmap is None
    assert exclusion == {"retired": []}
    assert next(row for row in rows if row["check"] == "v627_active_yaml_readable")["available"] is False

    def fail_writable(**_kwargs: Any) -> Any:
        raise OSError("closed")

    monkeypatch.setattr(mod.tempfile, "NamedTemporaryFile", fail_writable)
    assert mod._artifact_path_writable(output) == (False, "OSError: closed")
    monkeypatch.undo()

    output = _write_inputs(tmp_path)
    (tmp_path / mod.DESIGN_PATH).write_text("not a contract\n", encoding="utf-8")
    artifact = mod.build_artifact(tmp_path, "20260908", output_path=output)
    assert artifact["gate_check_summary"]["failed_check"] == "contract_parse"
    assert artifact["verdict_class"] == "disqualified"
    assert mod.validate_artifact(artifact) == []


def test_req_report_7136_failure_summary_names_count_failures() -> None:
    """REQ-REPORT-7136 reports Markdown and YAML count failures directly."""

    assert mod._failure_summary(
        {"markdown_task_count": 11, "yaml_task_rows": [], "task_contract_rows": []}
    )["failed_check"] == "markdown_task_count"
    assert mod._failure_summary(
        {"markdown_task_count": 12, "yaml_task_rows": [], "task_contract_rows": []}
    )["failed_check"] == "yaml_task_count"


def test_req_report_7136_score_helper_rejects_each_stored_row_fault(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7136-ARTIFACT fails each stored-score prerequisite."""

    output = _write_inputs(tmp_path)
    artifact = mod.build_artifact(tmp_path, "20260908", output_path=output)
    changes: tuple[Callable[[dict[str, Any]], None], ...] = (
        lambda value: value.__setitem__("markdown_task_rows", None),
        lambda value: value["markdown_task_rows"].pop(),
        lambda value: value["markdown_task_rows"][0].__setitem__("id", "exp0-wrong"),
        lambda value: value["yaml_task_rows"][0].__setitem__("id", "exp0-wrong"),
        lambda value: value["task_contract_rows"][0].__setitem__("passed", False),
        lambda value: value.__setitem__("focused_test_rows", []),
    )
    for mutate in changes:
        changed = deepcopy(artifact)
        mutate(changed)
        assert mod._contract_score_from_artifact(changed) == 0


def test_req_report_7136_validator_reports_every_metadata_fault(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7136-ARTIFACT validates all required metadata."""

    output = _write_inputs(tmp_path)
    artifact = mod.build_artifact(tmp_path, "20260908", output_path=output)
    cases: tuple[tuple[str, Any, str], ...] = (
        ("field_principles", {}, "field_principles_invalid"),
        ("status", "running", "status_invalid"),
        ("inference_substrate", "wrong", "inference_substrate_invalid"),
        ("expected_task_count", 11, "expected_task_count_invalid"),
        ("expected_id_order", [], "expected_id_order_invalid"),
        ("verifier_is_oracle", True, "verifier_is_oracle_invalid"),
        ("random_seed", 0, "random_seed_invalid"),
        ("duration_s", True, "duration_s_invalid"),
        ("run_date", "2026-09-08", "run_date_invalid"),
        ("source_artifact_hashes", [], "source_artifact_hashes_invalid"),
        ("observed_task_count", 0, "observed_task_count_invalid"),
        ("observed_id_order", [], "observed_id_order_invalid"),
        ("rows", [], "rows_not_task_contract_rows"),
        ("preconditions_checked", [{}], "preconditions_checked_invalid"),
        ("inference_substrate_class", "blocked_no_run", "inference_substrate_class_invalid"),
        ("honest_verdict", "wrong", "honest_verdict_not_derived"),
        ("gate_check_summary", {}, "gate_check_summary_invalid"),
    )
    for field, value, expected_error in cases:
        changed = deepcopy(artifact)
        changed[field] = value
        changed["reproducibility_checksum"] = mod.reproducibility_checksum(changed)
        assert expected_error in mod.validate_artifact(changed)
    changed = deepcopy(artifact)
    changed["gate_check_summary"]["passed"] = False
    changed["reproducibility_checksum"] = mod.reproducibility_checksum(changed)
    assert "gate_check_summary_invalid" in mod.validate_artifact(changed)

    blocked_output = _write_inputs(tmp_path / "blocked")
    (tmp_path / "blocked" / mod.DESIGN_PATH).unlink()
    blocked = mod.build_artifact(tmp_path / "blocked", "20260908", output_path=blocked_output)
    blocked["gate_check_summary"]["observed_value"] = "wrong"
    blocked["reproducibility_checksum"] = mod.reproducibility_checksum(blocked)
    assert "gate_check_summary_invalid" in mod.validate_artifact(blocked)


def test_req_report_7136_date_and_main_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """REQ-REPORT-7136 validates dates and exposes both CLI exit paths."""

    assert mod._date_argument("20260908") == "20260908"
    with pytest.raises(Exception, match="date must be a real YYYYMMDD value"):
        mod._date_argument("20260230")
    _write_inputs(tmp_path)
    monkeypatch.setattr(mod, "REPO_ROOT", tmp_path)
    assert mod.main(["--date", "20260908", "--output", "result.json"]) == 0
    monkeypatch.setattr(mod, "validate_artifact", lambda _artifact: ["forced"])
    assert mod.main(["--date", "20260908", "--output", str(tmp_path / "bad.json")]) == 1
