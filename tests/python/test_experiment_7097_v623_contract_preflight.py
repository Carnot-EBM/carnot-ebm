"""Tests for REQ-REPORT-7097, the independent V623 contract preflight."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any, Callable

import pytest
import yaml

from carnot import experiment_7097_v623_contract_preflight as mod


ROOT = Path(__file__).resolve().parents[2]

YAML_TASK_ROWS = (
    (
        "exp7097-v623-contract-preflight",
        "V623 Markdown and YAML task-contract preflight",
        "results/experiment_7097_v623_contract_preflight.json",
    ),
    (
        "exp7098-v623-execution-sota-ingestion",
        "V623 execution-time SOTA ingestion and claim-boundary audit",
        "results/experiment_7098_v623_sota_ingestion.json",
    ),
    (
        "exp7099-adapter-withheld-live-path-preflight",
        "Adapter-withheld ARC live-path preflight",
        "results/experiment_7099_v623_adapter_withheld_preflight.json",
    ),
    (
        "exp7100-adapter-withheld-arc-loo-measurement",
        "Mandatory adapter-withheld ARC leave-one-game-out measurement",
        "results/experiment_7100_v623_adapter_withheld_loo.json",
    ),
    (
        "exp7101-adapter-withheld-arc-cold-audit",
        "Independent adapter-withheld ARC provenance and leakage audit",
        "results/experiment_7101_v623_adapter_withheld_cold_audit.json",
    ),
    (
        "exp7102-feasibility-projected-action-energy",
        "Exact feasibility projection and analytic ARC action-energy comparison",
        "results/experiment_7102_v623_feasibility_action_energy.json",
    ),
    (
        "exp7103-adapter-withheld-energy-live-ab",
        "Adapter-withheld feasibility-energy live A/B",
        "results/experiment_7103_v623_adapter_withheld_energy_live_ab.json",
    ),
    (
        "exp7104-degree16-action-energy-portability",
        "Degree-16 action-energy software portability receipt",
        "results/experiment_7104_v623_degree16_action_energy_portability.json",
    ),
    (
        "exp7105-sealed-exact-constraint-stream",
        "Sealed 144-event exact constraint stream",
        "results/experiment_7105_v623_exact_constraint_stream.json",
    ),
    (
        "exp7106-delayed-commit-procedural-memory-csl",
        "Delayed-commit procedural-memory continuous self-learning A/B",
        "results/experiment_7106_v623_procedural_memory_csl.json",
    ),
    (
        "exp7107-continual-memory-cold-audit",
        "Fresh-process continual-memory retention and rollback audit",
        "results/experiment_7107_v623_continual_memory_cold_audit.json",
    ),
    (
        "exp7108-v623-capstone",
        "V623 independent evidence matrix and branch disposition",
        "results/experiment_7108_v623_capstone.json",
    ),
)

MARKDOWN_CONTRACT = """# Independent V623 design fixture

**Milestone:** `2026.09.623`

## Exact Task Contract

| Order | Full task ID | Title | Deliverable | Structured gate |
|---:|---|---|---|---|
| 1 | `exp7097-v623-contract-preflight` | V623 Markdown and YAML task-contract preflight | `results/experiment_7097_v623_contract_preflight.json` | none |
| 2 | `exp7098-v623-execution-sota-ingestion` | V623 execution-time SOTA ingestion and claim-boundary audit | `results/experiment_7098_v623_sota_ingestion.json` | none |
| 3 | `exp7099-adapter-withheld-live-path-preflight` | Adapter-withheld ARC live-path preflight | `results/experiment_7099_v623_adapter_withheld_preflight.json` | none |
| 4 | `exp7100-adapter-withheld-arc-loo-measurement` | Mandatory adapter-withheld ARC leave-one-game-out measurement | `results/experiment_7100_v623_adapter_withheld_loo.json` | `exp7099-adapter-withheld-live-path-preflight.adapter_withheld_live_path_ready_score == 1` |
| 5 | `exp7101-adapter-withheld-arc-cold-audit` | Independent adapter-withheld ARC provenance and leakage audit | `results/experiment_7101_v623_adapter_withheld_cold_audit.json` | `exp7100-adapter-withheld-arc-loo-measurement.adapter_withheld_loo_complete_score == 1` |
| 6 | `exp7102-feasibility-projected-action-energy` | Exact feasibility projection and analytic ARC action-energy comparison | `results/experiment_7102_v623_feasibility_action_energy.json` | `exp7100-adapter-withheld-arc-loo-measurement.adapter_withheld_loo_complete_score == 1`; `exp7101-adapter-withheld-arc-cold-audit.adapter_withheld_audit_ready_score == 1` |
| 7 | `exp7103-adapter-withheld-energy-live-ab` | Adapter-withheld feasibility-energy live A/B | `results/experiment_7103_v623_adapter_withheld_energy_live_ab.json` | `exp7102-feasibility-projected-action-energy.projected_action_energy_comparison_complete_score == 1` |
| 8 | `exp7104-degree16-action-energy-portability` | Degree-16 action-energy software portability receipt | `results/experiment_7104_v623_degree16_action_energy_portability.json` | `exp7102-feasibility-projected-action-energy.projected_action_energy_comparison_complete_score == 1` |
| 9 | `exp7105-sealed-exact-constraint-stream` | Sealed 144-event exact constraint stream | `results/experiment_7105_v623_exact_constraint_stream.json` | none |
| 10 | `exp7106-delayed-commit-procedural-memory-csl` | Delayed-commit procedural-memory continuous self-learning A/B | `results/experiment_7106_v623_procedural_memory_csl.json` | `exp7105-sealed-exact-constraint-stream.exact_constraint_stream_ready_score == 1` |
| 11 | `exp7107-continual-memory-cold-audit` | Fresh-process continual-memory retention and rollback audit | `results/experiment_7107_v623_continual_memory_cold_audit.json` | `exp7106-delayed-commit-procedural-memory-csl.procedural_memory_comparison_complete_score == 1` |
| 12 | `exp7108-v623-capstone` | V623 independent evidence matrix and branch disposition | `results/experiment_7108_v623_capstone.json` | none |

### Exp7097 - preflight
**Prior failures:** Exp7076, Exp7084, and Exp7091 found contract mismatches. V623 replaces both views.

### Exp7098 - ingestion
**Prior failures:** Exp6198 found no accepted source. V623 advances the boundary. Exp7092 is a successful method reference.

### Exp7099 - path preflight
**Prior failures:** Exp6122 lacked causal receipts. Exp5766 found no held-out gain. This attempt uses the E3 route.

### Exp7100 - LOO
**Prior failures:** Exp4330, Exp5766, and Exp6122 found no adapter-free advance. V623 freezes current cells.

### Exp7101 - cold audit
**Prior failures:** Exp6122 lacked direct receipts. V623 audits the ledger emitted by Exp7100.

### Exp7102 - action energy
**Prior failures:** Exp5712 was null and Exp6949 was gate-blocked. V623 uses a fixed analytic energy.

### Exp7103 - live A/B
**Prior failures:** Exp5712 and Exp5766 found no live held-out gain. V623 tests the reachable loop.

### Exp7104 - portability
**Prior failures:** Exp7083 and Exp7090 were blocked. V623 uses current action energy.

### Exp7105 - stream
**Prior failures:** Exp7070 had too few events. V623 creates an independent stream.

### Exp7106 - memory
**Prior failures:** Exp6978 and Exp7070 lacked value or evidence. V623 changes both.

### Exp7107 - cold audit
**Prior failures:** Exp6979 was null and Exp7071 was blocked. V623 audits the completed stream.

### Exp7108 - capstone
**Prior failures:** Exp6952 was partial. V623 keeps the capstone ungated.
"""

GATES = {
    7100: [(7099, "adapter_withheld_live_path_ready_score")],
    7101: [(7100, "adapter_withheld_loo_complete_score")],
    7102: [
        (7100, "adapter_withheld_loo_complete_score"),
        (7101, "adapter_withheld_audit_ready_score"),
    ],
    7103: [(7102, "projected_action_energy_comparison_complete_score")],
    7104: [(7102, "projected_action_energy_comparison_complete_score")],
    7106: [(7105, "exact_constraint_stream_ready_score")],
    7107: [(7106, "procedural_memory_comparison_complete_score")],
}
PRODUCED_FIELDS = {
    7099: ("adapter_withheld_live_path_ready_score",),
    7100: ("adapter_withheld_loo_complete_score",),
    7101: ("adapter_withheld_audit_ready_score",),
    7102: ("projected_action_energy_comparison_complete_score",),
    7105: ("exact_constraint_stream_ready_score",),
    7106: ("procedural_memory_comparison_complete_score",),
}
PRIOR_IDS = {
    7097: (
        "exp7076-v620-contract-preflight",
        "exp7084-v621-contract-preflight",
        "exp7091-v622-contract-preflight",
    ),
    7098: ("exp6198-v537-post-marker-source-scope-audit",),
    7099: (
        "exp5766-arc-loo-component-interaction-audit",
        "exp6122-arc-primitive-reachability-loo",
    ),
    7100: (
        "exp4330-arc-adapter-free-discovery-sweep-shallow-tail",
        "exp5766-arc-loo-component-interaction-audit",
        "exp6122-arc-primitive-reachability-loo",
    ),
    7101: ("exp6122-arc-primitive-reachability-loo",),
    7102: ("exp5712-arc-relational-goal-energy-live-ab", "exp6949-arc-branch-energy"),
    7103: (
        "exp5712-arc-relational-goal-energy-live-ab",
        "exp5766-arc-loo-component-interaction-audit",
    ),
    7104: (
        "exp7083-entrance-ising-degree16-parity",
        "exp7090-entrance-ising-degree16-sampling-receipt",
    ),
    7105: ("exp7070-v619-bcit-self-learning",),
    7106: (
        "exp6978-transactional-constraint-self-learning",
        "exp7070-v619-bcit-self-learning",
    ),
    7107: ("exp6979-self-learning-cold-audit", "exp7071-bcit-drift-rollback-audit"),
    7108: ("exp6952-v608-capstone",),
}
SUBSTRATE_CLASSES = {
    7097: "aggregation",
    7098: "no_model_load",
    7099: "model_full_generation",
    7100: "model_full_generation",
    7101: "aggregation",
    7102: "aggregation",
    7103: "model_full_generation",
    7104: "no_model_load",
    7105: "no_model_load",
    7106: "no_model_load",
    7107: "aggregation",
    7108: "aggregation",
}
RUN_SCRIPTS = {
    7097: "experiment_7097_v623_contract_preflight.py",
    7098: "experiment_7098_v623_sota_ingestion.py",
    7099: "experiment_7099_v623_adapter_withheld_preflight.py",
    7100: "experiment_7100_v623_adapter_withheld_loo.py",
    7101: "experiment_7101_v623_adapter_withheld_cold_audit.py",
    7102: "experiment_7102_v623_feasibility_action_energy.py",
    7103: "experiment_7103_v623_adapter_withheld_energy_live_ab.py",
    7104: "experiment_7104_v623_degree16_action_energy_portability.py",
    7105: "experiment_7105_v623_exact_constraint_stream.py",
    7106: "experiment_7106_v623_procedural_memory_csl.py",
    7107: "experiment_7107_v623_continual_memory_cold_audit.py",
    7108: "experiment_7108_v623_capstone.py",
}


def _prompt(number: int) -> str:
    """Build policy prose without consulting the Markdown fixture."""

    fields = list(mod.TASK_REQUIRED_FIELDS) + list(PRODUCED_FIELDS.get(number, ()))
    lines = ["Compare every per-unit row under matched controls."]
    if number in {7099, 7100, 7103}:
        fields.append("MODEL_SPECS")
        lines.append(
            "MODEL_SPECS must contain and execute unsloth/Qwen3.8-27B-GGUF as the "
            "pinned ARC primary arm and unsloth/Qwen3.6-35B-A3B-GGUF as the "
            "mandatory SOTA headline control."
        )
    lines.extend(
        [
            "REQUIRED ARTIFACT FIELDS: "
            + "; ".join(fields)
            + f"; inference_substrate_class ({SUBSTRATE_CLASSES[number]}, or blocked_no_run on a blocked precondition)"
            + "; execution_venue (host)"
            + "; gate_check_summary with failed check, expected value, and observed value"
            + "; verdict_class (positive | circular_positive | null | blocked | disqualified | partial).",
            "Run command: cd {project_root} && .venv/bin/python scripts/experiments/"
            f"{RUN_SCRIPTS[number]} --date {{date}}",
            mod.PROMPT_FINAL_LINE,
        ]
    )
    return "\n".join(lines)


def _roadmap() -> dict[str, Any]:
    """Build a YAML fixture without parsing or reusing the Markdown fixture."""

    tasks: list[dict[str, Any]] = []
    for task_id, title, deliverable in YAML_TASK_ROWS:
        number = int(task_id[3:7])
        task: dict[str, Any] = {
            "id": task_id,
            "title": title,
            "track": "infrastructure",
            "priority": "critical",
            "agent_type": "codex",
            "model": "gpt-5.6-sol",
            "requires_gpu": number in {7099, 7100, 7103},
            "max_turns": 50,
            "estimated_wall_time_min": 30,
            "per_unit_rows": True,
            "milestone": "2026.09.623",
            "deliverable": deliverable,
            "prompt": _prompt(number),
            "prior_failures": [
                {
                    "experiment_id": prior,
                    "verdict": "complete_null_prior",
                    "addressed_by": "V623 changes the failed evidence source and tests it again.",
                    "retire_if_same_verdict": True,
                }
                for prior in PRIOR_IDS[number]
            ],
        }
        if number in GATES:
            task["gated_on"] = [
                {
                    "upstream": next(
                        row[0] for row in YAML_TASK_ROWS if row[0].startswith(f"exp{upstream}-")
                    ),
                    "artifact_field": field,
                    "op": "==",
                    "value": 1,
                }
                for upstream, field in GATES[number]
            ]
        tasks.append(task)
    return {
        "milestone": "2026.09.623",
        "milestone_title": "fixture",
        "milestone_doc": "openspec/change-proposals/research-roadmap-vNEXT.md",
        "tasks": tasks,
    }


def _write_preconditions(root: Path, roadmap: dict[str, Any] | None = None) -> None:
    """Create only the runtime files that the audit can read."""

    values = {
        mod.DESIGN_PATH: MARKDOWN_CONTRACT,
        mod.ACTIVE_ROADMAP_PATH: yaml.safe_dump(roadmap or _roadmap()),
        mod.EXCLUSION_PATH: "retired: []\nretired_extras: []\n",
    }
    for relative, content in values.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")


def test_req_report_7097_spec_defines_exact_contract() -> None:
    """REQ-REPORT-7097 names all rows, evidence fields, and scenarios."""

    spec = (ROOT / mod.SPEC_PATH).read_text(encoding="utf-8")
    section = spec[spec.index("### REQ-REPORT-7097") :]
    assert mod.EXPECTED_TASK_COUNT == len(YAML_TASK_ROWS) == 12
    assert mod.EXPECTED_TASK_IDS == tuple(row[0] for row in YAML_TASK_ROWS)
    assert mod.EXPECTED_GATE_COUNT == 8
    for scenario in ("PARITY", "GATES", "DISCIPLINE", "PREFLIGHT", "ARTIFACT"):
        assert f"SCENARIO-REPORT-7097-{scenario}" in section
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert field in section


def test_scenario_report_7097_parity_accepts_twelve_independent_rows() -> None:
    """SCENARIO-REPORT-7097-PARITY accepts exact independent sources."""

    result = mod.evaluate_contract(MARKDOWN_CONTRACT, _roadmap(), set())
    assert result["passed"] is True
    assert len(result["task_contract_rows"]) == 12
    assert len(result["gate_producer_rows"]) == 8
    assert result["markdown_task_rows"] is not result["yaml_task_rows"]
    assert result["markdown_task_rows"][5]["gates"][1]["upstream"] == YAML_TASK_ROWS[4][0]
    assert all(row["passed"] for row in result["task_contract_rows"])


@pytest.mark.parametrize(
    "case", ("missing", "extra", "reordered", "duplicated", "title", "deliverable")
)
def test_req_report_7097_row_mutations_fail(case: str) -> None:
    """REQ-REPORT-7097 rejects missing, extra, reordered, duplicated, or changed rows."""

    roadmap = _roadmap()
    if case == "missing":
        roadmap["tasks"].pop(5)
    elif case == "extra":
        roadmap["tasks"].append(deepcopy(roadmap["tasks"][-1]))
    elif case == "reordered":
        roadmap["tasks"][5], roadmap["tasks"][6] = roadmap["tasks"][6], roadmap["tasks"][5]
    elif case == "duplicated":
        roadmap["tasks"][5] = deepcopy(roadmap["tasks"][4])
    elif case == "title":
        roadmap["tasks"][0]["title"] += " changed"
    else:
        roadmap["tasks"][0]["deliverable"] = "results/wrong.json"
    assert mod.evaluate_contract(MARKDOWN_CONTRACT, roadmap, set())["passed"] is False


def test_scenario_report_7097_markdown_and_gate_syntax_are_strict() -> None:
    """SCENARIO-REPORT-7097-GATES rejects malformed Markdown and YAML gates."""

    with pytest.raises(ValueError, match="task row"):
        mod.parse_markdown_contract(MARKDOWN_CONTRACT.replace("| 1 |", "| 1 | extra |", 1))
    with pytest.raises(ValueError, match="structured gate"):
        mod.parse_markdown_contract(MARKDOWN_CONTRACT.replace(" == 1`", " is ready`", 1))
    with pytest.raises(ValueError, match="section is missing"):
        mod.parse_markdown_contract(MARKDOWN_CONTRACT.replace("## Exact Task Contract", "## Other"))
    with pytest.raises(ValueError, match="task row"):
        mod.parse_markdown_contract(MARKDOWN_CONTRACT.replace("exp7097-v623", "not-an-exp", 1))
    with pytest.raises(ValueError, match="table is missing"):
        mod.parse_markdown_contract("**Milestone:** `2026.09.623`\n\n## Exact Task Contract\n")
    assert len(mod.parse_markdown_contract(MARKDOWN_CONTRACT)["tasks"]) == 12
    roadmap = _roadmap()
    roadmap["tasks"][3]["gated_on"][0]["op"] = None
    assert mod.evaluate_contract(MARKDOWN_CONTRACT, roadmap, set())["passed"] is False


def test_scenario_report_7097_gates_require_earlier_producer_bare_owned_fields() -> None:
    """SCENARIO-REPORT-7097-GATES rejects bad producers and a gated capstone."""

    mutations: tuple[Callable[[dict[str, Any]], None], ...] = (
        lambda value: value["tasks"][3]["gated_on"][0].__setitem__(
            "artifact_field", "nested.value"
        ),
        lambda value: value["tasks"][3]["gated_on"][0].__setitem__(
            "artifact_field", "missing_field"
        ),
        lambda value: value["tasks"][3]["gated_on"][0].__setitem__(
            "upstream", value["tasks"][-1]["id"]
        ),
        lambda value: value["tasks"][-1].__setitem__(
            "gated_on", deepcopy(value["tasks"][3]["gated_on"])
        ),
        lambda value: value["tasks"][3]["gated_on"][0].__setitem__(
            "upstream", value["tasks"][0]["id"]
        ),
    )
    for mutate in mutations:
        roadmap = _roadmap()
        mutate(roadmap)
        assert mod.evaluate_contract(MARKDOWN_CONTRACT, roadmap, set())["passed"] is False


def test_scenario_report_7097_prior_failures_require_all_four_fields() -> None:
    """SCENARIO-REPORT-7097-DISCIPLINE rejects incomplete or mismatched priors."""

    for field, value in (
        ("experiment_id", ""),
        ("verdict", ""),
        ("addressed_by", ""),
        ("retire_if_same_verdict", False),
    ):
        roadmap = _roadmap()
        roadmap["tasks"][0]["prior_failures"][0][field] = value
        assert mod.evaluate_contract(MARKDOWN_CONTRACT, roadmap, set())["passed"] is False
    roadmap = _roadmap()
    roadmap["tasks"][0]["prior_failures"] = []
    assert mod.evaluate_contract(MARKDOWN_CONTRACT, roadmap, set())["passed"] is False
    parsed = mod.parse_markdown_contract(MARKDOWN_CONTRACT)
    assert parsed["tasks"][1]["prior_failure_ids"] == ["exp6198"]
    assert parsed["tasks"][4]["prior_failure_ids"] == ["exp6122"]


def test_scenario_report_7097_execution_rules_fail_closed() -> None:
    """SCENARIO-REPORT-7097-DISCIPLINE checks routes, classes, venues, and tails."""

    mutations: tuple[Callable[[dict[str, Any]], None], ...] = (
        lambda value: value["tasks"][4].__setitem__("per_unit_rows", False),
        lambda value: value["tasks"][0].__setitem__("model", "wrong"),
        lambda value: value["tasks"][0].__setitem__("agent_type", "gemini"),
        lambda value: value["tasks"][0].__setitem__("max_turns", 0),
        lambda value: value["tasks"][0].__setitem__(
            "prompt", value["tasks"][0]["prompt"].replace("aggregation, or", "unknown, or")
        ),
        lambda value: value["tasks"][0].__setitem__(
            "prompt",
            value["tasks"][0]["prompt"].replace(
                "execution_venue (host)", "execution_venue (container)"
            ),
        ),
        lambda value: value["tasks"][0].__setitem__(
            "prompt", value["tasks"][0]["prompt"].replace(" | partial", " | deferred")
        ),
        lambda value: value["tasks"][0].__setitem__(
            "prompt", value["tasks"][0]["prompt"].replace("observed value", "observation")
        ),
        lambda value: value["tasks"][0].__setitem__(
            "prompt", value["tasks"][0]["prompt"].replace(mod.PROMPT_FINAL_LINE, "")
        ),
        lambda value: value["tasks"][0].__setitem__(
            "prompt", value["tasks"][0]["prompt"].replace("--date {date}", "--date wrong")
        ),
    )
    for mutate in mutations:
        roadmap = _roadmap()
        mutate(roadmap)
        assert mod.evaluate_contract(MARKDOWN_CONTRACT, roadmap, set())["passed"] is False
    assert mod.evaluate_contract(MARKDOWN_CONTRACT, _roadmap(), {"exp7103"})["passed"] is False


def test_scenario_report_7097_model_policy_requires_current_headline_cells() -> None:
    """SCENARIO-REPORT-7097-DISCIPLINE rejects missing or legacy model cells."""

    mutations: tuple[Callable[[dict[str, Any]], None], ...] = (
        lambda value: value["tasks"][2].__setitem__("requires_gpu", False),
        lambda value: value["tasks"][2].__setitem__(
            "prompt", value["tasks"][2]["prompt"].replace("MODEL_SPECS", "models_used")
        ),
        lambda value: value["tasks"][2].__setitem__(
            "prompt", value["tasks"][2]["prompt"].replace(mod.HEADLINE_MODEL, "legacy/model")
        ),
        lambda value: value["tasks"][2].__setitem__(
            "prompt", value["tasks"][2]["prompt"].replace(mod.ARC_LIVE_MODEL, "legacy/model")
        ),
        lambda value: value["tasks"][2].__setitem__(
            "prompt", value["tasks"][2]["prompt"].replace("execute", "list")
        ),
        lambda value: value["tasks"][2].__setitem__(
            "prompt", value["tasks"][2]["prompt"].replace("headline", "secondary")
        ),
        lambda value: value["tasks"][2].__setitem__(
            "prompt",
            value["tasks"][2]["prompt"]
            + "\nMODEL_SPECS may use Qwen3.5-0.8B as the headline fallback.",
        ),
        lambda value: value["tasks"][1].__setitem__(
            "prompt",
            value["tasks"][1]["prompt"].replace("no_model_load, or", "model_full_generation, or"),
        ),
    )
    for mutate in mutations:
        roadmap = _roadmap()
        mutate(roadmap)
        assert mod.evaluate_contract(MARKDOWN_CONTRACT, roadmap, set())["passed"] is False


def test_scenario_report_7097_preflight_separates_terminal_classes(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7097-PREFLIGHT separates positive, mismatch, and no-run."""

    _write_preconditions(tmp_path)
    positive = mod.build_artifact(tmp_path, "20260907", output_path=tmp_path / "positive.json")
    assert positive["verdict_class"] == "positive"
    assert positive["execution_venue"] == "host"
    assert mod.validate_artifact(positive) == []

    changed = _roadmap()
    changed["tasks"].pop()
    _write_preconditions(tmp_path, changed)
    disqualified = mod.build_artifact(tmp_path, "20260907", output_path=tmp_path / "bad.json")
    assert disqualified["verdict_class"] == "disqualified"
    assert disqualified["v623_task_contract_conforms_score"] == 0
    assert mod.validate_artifact(disqualified) == []

    (tmp_path / mod.DESIGN_PATH).unlink()
    blocked = mod.build_artifact(tmp_path, "20260907", output_path=tmp_path / "blocked.json")
    assert blocked["verdict_class"] == "blocked"
    assert blocked["inference_substrate_class"] == "blocked_no_run"
    assert blocked["gate_check_summary"]["failed_check"] == "v623_markdown_readable"
    assert mod.validate_artifact(blocked) == []


def test_scenario_report_7097_artifact_validator_rejects_forgery(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7097-ARTIFACT recomputes evidence and headlines."""

    _write_preconditions(tmp_path)
    good = mod.build_artifact(tmp_path, "20260907", output_path=tmp_path / "good.json")
    cases = (
        ("field_principles", {}, "field_principles_invalid"),
        ("inference_substrate", "wrong", "inference_substrate_invalid"),
        ("inference_substrate_class", "blocked_no_run", "inference_substrate_class_invalid"),
        ("execution_venue", "container", "execution_venue_invalid"),
        ("v623_task_contract_conforms_score", 0, "v623_task_contract_conforms_score_invalid"),
        ("verdict_class", "partial", "verdict_class_not_derived"),
        ("honest_verdict", "complete_wrong", "honest_verdict_not_derived"),
        ("gate_check_summary", {}, "gate_check_summary_invalid"),
    )
    for field, value, error in cases:
        changed = deepcopy(good)
        changed[field] = value
        changed["reproducibility_checksum"] = mod.reproducibility_checksum(changed)
        assert error in mod.validate_artifact(changed)
    changed = deepcopy(good)
    changed["task_contract_rows"][0]["passed"] = False
    changed["reproducibility_checksum"] = mod.reproducibility_checksum(changed)
    assert "v623_task_contract_conforms_score_invalid" in mod.validate_artifact(changed)
    changed = deepcopy(good)
    changed["reproducibility_checksum"] = "sha256:bad"
    assert "reproducibility_checksum_invalid" in mod.validate_artifact(changed)


def test_req_report_7097_cli_parse_and_writable_error_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-REPORT-7097 covers command, parse-error, and writable-path behavior."""

    with pytest.raises(Exception, match="YYYYMMDD"):
        mod._date_argument("2026-09-07")
    with pytest.raises(SystemExit):
        mod.main([])
    assert mod.main(["--validate-artifact", str(tmp_path / "missing.json")]) == 1
    _write_preconditions(tmp_path)
    assert mod.main(["--date", "20260907", "--root", str(tmp_path), "--output", "result.json"]) == 0
    artifact_path = tmp_path / "result.json"
    assert mod.main(["--validate-artifact", str(artifact_path)]) == 0
    (tmp_path / mod.DESIGN_PATH).write_text("readable but malformed\n", encoding="utf-8")
    bad = mod.build_artifact(tmp_path, "20260907", output_path=tmp_path / "malformed.json")
    assert bad["gate_check_summary"]["failed_check"] == "contract_parse"
    assert mod.validate_artifact(bad) == []
    monkeypatch.setattr(
        mod.tempfile,
        "NamedTemporaryFile",
        lambda **_kwargs: (_ for _ in ()).throw(OSError("read only")),
    )
    assert mod._artifact_path_writable(tmp_path / "blocked.json") == (False, "OSError: read only")


def test_req_report_7097_validator_rejects_shapes_and_derives_summaries(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7097-ARTIFACT derives all count and row diagnostics."""

    _write_preconditions(tmp_path)
    good = mod.build_artifact(tmp_path, "20260907", output_path=tmp_path / "good.json")
    mutations: tuple[Callable[[dict[str, Any]], None], ...] = (
        lambda value: value["markdown_task_rows"][0].__setitem__("id", "exp9999-wrong"),
        lambda value: value["yaml_task_rows"][0].__setitem__("id", "exp9999-wrong"),
        lambda value: value["task_contract_rows"][0].__setitem__("order", 99),
        lambda value: value.__setitem__("title_parity_rows", []),
        lambda value: value["title_parity_rows"][0].__setitem__("passed", False),
        lambda value: value["gate_producer_rows"].pop(),
        lambda value: value.__setitem__("prior_failure_rows", value["prior_failure_rows"][:11]),
    )
    for mutate in mutations:
        changed = deepcopy(good)
        mutate(changed)
        assert mod._contract_score_from_artifact(changed) == 0

    missing = deepcopy(good)
    missing.pop("rows")
    assert mod.validate_artifact(missing) == ["missing_required_field:rows"]
    for field, value, error in (
        ("verifier_is_oracle", True, "verifier_is_oracle_invalid"),
        ("expected_task_count", 11, "expected_task_count_invalid"),
        ("expected_id_order", [], "expected_id_order_invalid"),
        ("active_roadmap_path", "next.yaml", "active_roadmap_path_invalid"),
        ("staging_file_required_at_execution", True, "staging_file_required_invalid"),
        ("random_seed", 0, "random_seed_invalid"),
        ("duration_s", -1, "duration_s_invalid"),
        ("observed_task_count", 0, "observed_task_count_invalid"),
        ("observed_id_order", [], "observed_id_order_invalid"),
        ("rows", [], "rows_not_task_contract_rows"),
    ):
        changed = deepcopy(good)
        changed[field] = value
        changed["reproducibility_checksum"] = mod.reproducibility_checksum(changed)
        assert error in mod.validate_artifact(changed)

    assert (
        mod._contract_failure_summary(
            {"markdown_task_count": 11, "yaml_task_rows": [], "task_contract_rows": []}
        )["failed_check"]
        == "markdown_task_count"
    )
    assert (
        mod._contract_failure_summary(
            {"markdown_task_count": 12, "yaml_task_rows": [], "task_contract_rows": []}
        )["failed_check"]
        == "yaml_task_count"
    )
    assert (
        mod._contract_failure_summary(
            {
                "markdown_task_count": 12,
                "yaml_task_rows": [{}] * 12,
                "task_contract_rows": [{"order": 4, "passed": False}],
            }
        )["failed_check"]
        == "task_contract_order_4"
    )

    markdown_short = deepcopy(good)
    markdown_short["markdown_task_rows"].pop()
    markdown_short["v623_task_contract_conforms_score"] = 0
    markdown_short["verdict_class"] = "disqualified"
    markdown_short["honest_verdict"] = mod.DISQUALIFIED_VERDICT
    markdown_short["gate_check_summary"] = {
        "failed_check": "markdown_task_count",
        "expected_value": 12,
        "observed_value": 11,
        "passed": False,
    }
    markdown_short["reproducibility_checksum"] = mod.reproducibility_checksum(markdown_short)
    assert mod.validate_artifact(markdown_short) == []


def test_req_report_7097_preconditions_report_missing_yaml_inputs(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7097-PREFLIGHT preserves each missing YAML prerequisite."""

    rows, roadmap, exclusion = mod._preconditions(tmp_path, tmp_path / "result.json")
    assert roadmap is exclusion is None
    assert {row["check"] for row in rows if not row["available"]} >= {
        "v623_markdown_readable",
        "v623_active_yaml_readable",
        "exclusion_manifest_readable",
    }


def test_req_report_7097_yaml_parser_rejects_malformed_shapes() -> None:
    """REQ-REPORT-7097 keeps malformed YAML separate from Markdown parsing."""

    with pytest.raises(ValueError, match="mapping"):
        mod.parse_yaml_contract([])
    with pytest.raises(ValueError, match="tasks list"):
        mod.parse_yaml_contract({"milestone": mod.MILESTONE})
    value = _roadmap()
    value["tasks"][0] = "bad"
    with pytest.raises(ValueError, match="task 1"):
        mod.parse_yaml_contract(value)
    value = _roadmap()
    value["tasks"][0]["gated_on"] = ["bad"]
    with pytest.raises(ValueError, match="malformed gate"):
        mod.parse_yaml_contract(value)


def test_req_report_7097_helper_boundaries_are_fail_closed(tmp_path: Path) -> None:
    """REQ-REPORT-7097 covers parser and manifest boundary values."""

    with pytest.raises(ValueError, match="must be scalar"):
        mod._parse_scalar("[1]")
    gate = mod._parse_markdown_gate_cell("exp9999-missing.ready == 1", {})
    assert gate[0]["upstream"] == "exp9999-missing"
    without_prior = MARKDOWN_CONTRACT.replace("**Prior failures:**", "**History:**", 1)
    assert len(mod.parse_markdown_contract(without_prior)["tasks"]) == 12
    assert mod._required_block("no declaration") == ""

    value = _roadmap()
    value["tasks"][0]["prior_failures"] = {"bad": "shape"}
    with pytest.raises(ValueError, match="malformed list"):
        mod.parse_yaml_contract(value)
    empty = tmp_path / "empty.yaml"
    empty.write_text("[]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="mapping required"):
        mod.load_yaml(empty)

    assert mod.retired_experiment_ids([]) == set()
    manifest = {
        "other": [],
        "retired": [
            "bad-row",
            {"experiment_ids": [7100], "un_retired_experiment_ids": [7100]},
            {"experiment_id": 7099},
        ],
    }
    assert mod.retired_experiment_ids(manifest) == {"exp7099"}
    assert mod._legacy_small_headline_substitution("Qwen3.5-0.8B is transport-only.") is False
    assert mod._expected_prompt_tail(9999) is None


def test_req_report_7097_real_active_roadmap_conforms(tmp_path: Path) -> None:
    """REQ-REPORT-7097 confirms the active 12-row V623 contract."""

    artifact = mod.build_artifact(ROOT, "20260907", output_path=tmp_path / "real.json")
    assert artifact["verdict_class"] == "positive"
    assert artifact["expected_task_count"] == artifact["observed_task_count"] == 12
    assert artifact["v623_task_contract_conforms_score"] == 1
    assert artifact["gate_check_summary"] == {
        "failed_check": None,
        "expected_value": 1,
        "observed_value": 1,
        "passed": True,
    }
    assert mod.validate_artifact(artifact) == []


def test_req_report_7097_cli_rejects_invalid_artifacts_and_generated_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-REPORT-7097 returns failure for invalid stored or generated data."""

    bad = tmp_path / "bad.json"
    bad.write_text("{}\n", encoding="utf-8")
    assert mod.main(["--validate-artifact", str(bad)]) == 1
    listed = tmp_path / "list.json"
    listed.write_text("[]\n", encoding="utf-8")
    assert mod.main(["--validate-artifact", str(listed)]) == 1
    _write_preconditions(tmp_path)
    monkeypatch.setattr(mod, "validate_artifact", lambda _artifact: ["forced failure"])
    assert (
        mod.main(["--date", "20260907", "--root", str(tmp_path), "--output", "invalid.json"]) == 1
    )
