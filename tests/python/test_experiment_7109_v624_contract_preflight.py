"""Tests for REQ-REPORT-7109, the independent V624 contract preflight."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any, Callable

import pytest
import yaml

from carnot import experiment_7109_v624_contract_preflight as mod


ROOT = Path(__file__).resolve().parents[2]

YAML_TASK_ROWS = (
    (
        "exp7109-v624-contract-preflight",
        "V624 Markdown and YAML task-contract preflight",
        "results/experiment_7109_v624_contract_preflight.json",
    ),
    (
        "exp7110-evidence-ingress-quarantine",
        "Forward evidence ingress quarantine and date contract",
        "results/experiment_7110_v624_evidence_ingress_quarantine.json",
    ),
    (
        "exp7111-forward-arc-provenance-canary",
        "Forward ARC evaluation provenance and dashboard canary",
        "results/experiment_7111_v624_arc_provenance_canary.json",
    ),
    (
        "exp7112-v624-execution-sota-ingestion",
        "V624 execution-time SOTA ingestion and claim-boundary audit",
        "results/experiment_7112_v624_sota_ingestion.json",
    ),
    (
        "exp7113-arc-generation-liveness-recovery",
        "Bounded ARC local-generation liveness and receipt recovery",
        "results/experiment_7113_v624_arc_generation_liveness.json",
    ),
    (
        "exp7114-adapter-withheld-arc-loo-measurement",
        "Mandatory adapter-withheld ARC leave-one-game-out measurement",
        "results/experiment_7114_v624_adapter_withheld_loo.json",
    ),
    (
        "exp7115-adapter-withheld-arc-cold-audit",
        "Independent adapter-withheld ARC provenance and leakage audit",
        "results/experiment_7115_v624_adapter_withheld_cold_audit.json",
    ),
    (
        "exp7116-sota-constraint-episode-bank",
        "SOTA constraint proposal and verified paraphrase episode bank",
        "results/experiment_7116_v624_sota_constraint_episode_bank.json",
    ),
    (
        "exp7117-exact-verify-revise-loop",
        "Exact think-verify-revise constraint loop on held-out paraphrases",
        "results/experiment_7117_v624_exact_verify_revise_loop.json",
    ),
    (
        "exp7118-principle-step-memory-csl",
        "Principle-level step-aligned continuous memory versus matched controls",
        "results/experiment_7118_v624_principle_step_memory_csl.json",
    ),
    (
        "exp7119-multi-iteration-memory-cold-audit",
        "Fresh-process multi-iteration memory retention and paraphrase audit",
        "results/experiment_7119_v624_multi_iteration_memory_cold_audit.json",
    ),
    (
        "exp7120-v624-capstone",
        "V624 independent evidence matrix and branch disposition",
        "results/experiment_7120_v624_capstone.json",
    ),
)

MARKDOWN_CONTRACT = """# Independent V624 design fixture

**Milestone:** `2026.09.624`

## Exact Task Contract

| Order | Full task ID | Title | Deliverable | Structured gate |
|---:|---|---|---|---|
| 1 | `exp7109-v624-contract-preflight` | V624 Markdown and YAML task-contract preflight | `results/experiment_7109_v624_contract_preflight.json` | none |
| 2 | `exp7110-evidence-ingress-quarantine` | Forward evidence ingress quarantine and date contract | `results/experiment_7110_v624_evidence_ingress_quarantine.json` | none |
| 3 | `exp7111-forward-arc-provenance-canary` | Forward ARC evaluation provenance and dashboard canary | `results/experiment_7111_v624_arc_provenance_canary.json` | none |
| 4 | `exp7112-v624-execution-sota-ingestion` | V624 execution-time SOTA ingestion and claim-boundary audit | `results/experiment_7112_v624_sota_ingestion.json` | none |
| 5 | `exp7113-arc-generation-liveness-recovery` | Bounded ARC local-generation liveness and receipt recovery | `results/experiment_7113_v624_arc_generation_liveness.json` | none |
| 6 | `exp7114-adapter-withheld-arc-loo-measurement` | Mandatory adapter-withheld ARC leave-one-game-out measurement | `results/experiment_7114_v624_adapter_withheld_loo.json` | `exp7113-arc-generation-liveness-recovery.arc_generation_liveness_ready_score == 1` |
| 7 | `exp7115-adapter-withheld-arc-cold-audit` | Independent adapter-withheld ARC provenance and leakage audit | `results/experiment_7115_v624_adapter_withheld_cold_audit.json` | `exp7110-evidence-ingress-quarantine.evidence_ingress_quarantine_ready_score == 1`; `exp7114-adapter-withheld-arc-loo-measurement.adapter_withheld_loo_complete_score == 1` |
| 8 | `exp7116-sota-constraint-episode-bank` | SOTA constraint proposal and verified paraphrase episode bank | `results/experiment_7116_v624_sota_constraint_episode_bank.json` | none |
| 9 | `exp7117-exact-verify-revise-loop` | Exact think-verify-revise constraint loop on held-out paraphrases | `results/experiment_7117_v624_exact_verify_revise_loop.json` | `exp7116-sota-constraint-episode-bank.sota_constraint_episode_bank_ready_score == 1` |
| 10 | `exp7118-principle-step-memory-csl` | Principle-level step-aligned continuous memory versus matched controls | `results/experiment_7118_v624_principle_step_memory_csl.json` | `exp7117-exact-verify-revise-loop.verify_revise_loop_complete_score == 1` |
| 11 | `exp7119-multi-iteration-memory-cold-audit` | Fresh-process multi-iteration memory retention and paraphrase audit | `results/experiment_7119_v624_multi_iteration_memory_cold_audit.json` | `exp7110-evidence-ingress-quarantine.evidence_ingress_quarantine_ready_score == 1`; `exp7118-principle-step-memory-csl.procedural_memory_comparison_complete_score == 1` |
| 12 | `exp7120-v624-capstone` | V624 independent evidence matrix and branch disposition | `results/experiment_7120_v624_capstone.json` | none |

### Exp7109 - preflight
**Prior failures:** Exp7076, Exp7084, and Exp7091 found contract mismatches. V624 replaces both views.

### Exp7110 - ingress
No prior failure is claimed.

### Exp7111 - provenance canary
No prior failure is claimed.

### Exp7112 - ingestion
**Prior failure:** Exp6198 found no accepted source. V624 advances the boundary.

### Exp7113 - liveness
**Prior failure:** Exp7099 produced no generation cells. V624 adds request-level receipts.

### Exp7114 - LOO
**Prior failure:** Exp7100 was gate-blocked. V624 uses a current liveness producer.

### Exp7115 - cold audit
No prior failure is claimed.

### Exp7116 - episode bank
No prior failure is claimed.

### Exp7117 - verify-revise
No prior failure is claimed.

### Exp7118 - memory
**Prior failures:** Exp6978 was null and Exp7070 blocked. V624 changes the bank and controls.

### Exp7119 - cold audit
**Prior failures:** Exp6979 was null and Exp7071 blocked. V624 audits current evidence.

### Exp7120 - capstone
**Prior failures:** Exp6952 was partial and Exp7108 consumed flagged evidence. V624 quarantines inputs.
"""

GATES = {
    7114: [(7113, "arc_generation_liveness_ready_score")],
    7115: [
        (7110, "evidence_ingress_quarantine_ready_score"),
        (7114, "adapter_withheld_loo_complete_score"),
    ],
    7117: [(7116, "sota_constraint_episode_bank_ready_score")],
    7118: [(7117, "verify_revise_loop_complete_score")],
    7119: [
        (7110, "evidence_ingress_quarantine_ready_score"),
        (7118, "procedural_memory_comparison_complete_score"),
    ],
}
PRODUCED_FIELDS = {
    7110: ("evidence_ingress_quarantine_ready_score",),
    7113: ("arc_generation_liveness_ready_score",),
    7114: ("adapter_withheld_loo_complete_score", "solve_provenance"),
    7116: ("sota_constraint_episode_bank_ready_score",),
    7117: ("verify_revise_loop_complete_score",),
    7118: ("procedural_memory_comparison_complete_score",),
}
PRIOR_IDS = {
    7109: (
        "exp7076-v620-contract-preflight",
        "exp7084-v621-contract-preflight",
        "exp7091-v622-contract-preflight",
    ),
    7110: (),
    7111: (),
    7112: ("exp6198-v537-post-marker-source-scope-audit",),
    7113: ("exp7099-adapter-withheld-live-path-preflight",),
    7114: ("exp7100-adapter-withheld-arc-loo-measurement",),
    7115: (),
    7116: (),
    7117: (),
    7118: (
        "exp6978-transactional-constraint-self-learning",
        "exp7070-v619-bcit-self-learning",
    ),
    7119: ("exp6979-self-learning-cold-audit", "exp7071-bcit-drift-rollback-audit"),
    7120: ("exp6952-v608-capstone", "exp7108-v623-capstone"),
}
SUBSTRATE_CLASSES = {
    7109: "aggregation",
    7110: "aggregation",
    7111: "no_model_load",
    7112: "no_model_load",
    7113: "model_bounded_generation",
    7114: "model_bounded_generation",
    7115: "aggregation",
    7116: "model_full_generation",
    7117: "model_full_generation",
    7118: "model_full_generation",
    7119: "aggregation",
    7120: "aggregation",
}
RUN_SCRIPTS = {
    7109: "experiment_7109_v624_contract_preflight.py",
    7110: "experiment_7110_v624_evidence_ingress_quarantine.py",
    7111: "experiment_7111_v624_arc_provenance_canary.py",
    7112: "experiment_7112_v624_sota_ingestion.py",
    7113: "experiment_7113_v624_arc_generation_liveness.py",
    7114: "experiment_7114_v624_adapter_withheld_loo.py",
    7115: "experiment_7115_v624_adapter_withheld_cold_audit.py",
    7116: "experiment_7116_v624_sota_constraint_episode_bank.py",
    7117: "experiment_7117_v624_exact_verify_revise_loop.py",
    7118: "experiment_7118_v624_principle_step_memory_csl.py",
    7119: "experiment_7119_v624_multi_iteration_memory_cold_audit.py",
    7120: "experiment_7120_v624_capstone.py",
}


def _prompt(number: int) -> str:
    """Build policy prose without consulting the Markdown fixture."""

    fields = list(mod.TASK_REQUIRED_FIELDS) + list(PRODUCED_FIELDS.get(number, ()))
    lines = ["Compare every per-unit row under matched controls."]
    if number in {7113, 7114}:
        fields.append("MODEL_SPECS")
        lines.append(
            "MODEL_SPECS must contain and execute unsloth/Qwen3.8-27B-GGUF as the "
            "pinned ARC primary arm and unsloth/Qwen3.6-35B-A3B-GGUF as the "
            "mandatory SOTA headline control."
        )
    if number in {7116, 7117, 7118}:
        fields.append("MODEL_SPECS")
        lines.append(
            "MODEL_SPECS must contain and execute unsloth/Qwen3.6-35B-A3B-GGUF "
            "and unsloth/gemma-4-26B-A4B-it-GGUF in mandatory SOTA headline cells."
        )
    if number == 7114:
        lines.append("This ARC task reports a game-level solve with solve provenance.")
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
            "track": "arc" if number in {7111, 7113, 7114, 7115} else "infrastructure",
            "priority": "critical",
            "agent_type": "codex",
            "model": "gpt-5.6-sol",
            "requires_gpu": number in {7113, 7114, 7116, 7117, 7118},
            "max_turns": 50,
            "estimated_wall_time_min": 30,
            "per_unit_rows": True,
            "milestone": "2026.09.624",
            "deliverable": deliverable,
            "prompt": _prompt(number),
            "prior_failures": [
                {
                    "experiment_id": prior,
                    "verdict": "complete_null_prior",
                    "addressed_by": "V624 changes the failed evidence source and tests it again.",
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
        "milestone": "2026.09.624",
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


def test_req_report_7109_spec_defines_exact_contract() -> None:
    """REQ-REPORT-7109 names all rows, evidence fields, and scenarios."""

    spec = (ROOT / mod.SPEC_PATH).read_text(encoding="utf-8")
    section = spec[spec.index("### REQ-REPORT-7109") :]
    assert mod.EXPECTED_TASK_COUNT == len(YAML_TASK_ROWS) == 12
    assert mod.EXPECTED_TASK_IDS == tuple(row[0] for row in YAML_TASK_ROWS)
    assert mod.EXPECTED_GATE_COUNT == 7
    assert mod.LEGAL_SUBSTRATE_CLASSES == {
        "aggregation",
        "no_model_load",
        "model_load_no_generation",
        "model_bounded_generation",
        "model_full_generation",
        "blocked_no_run",
    }
    for scenario in ("PARITY", "GATES", "DISCIPLINE", "PREFLIGHT", "ARTIFACT"):
        assert f"SCENARIO-REPORT-7109-{scenario}" in section
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert field in section


def test_scenario_report_7109_parity_accepts_twelve_independent_rows() -> None:
    """SCENARIO-REPORT-7109-PARITY accepts exact independent sources."""

    result = mod.evaluate_contract(MARKDOWN_CONTRACT, _roadmap(), set())
    assert result["passed"] is True
    assert len(result["task_contract_rows"]) == 12
    assert len(result["gate_producer_rows"]) == 7
    assert result["markdown_task_rows"] is not result["yaml_task_rows"]
    assert result["markdown_task_rows"][6]["gates"][1]["upstream"] == YAML_TASK_ROWS[5][0]
    assert all(row["passed"] for row in result["task_contract_rows"])


@pytest.mark.parametrize(
    "case", ("missing", "extra", "reordered", "duplicated", "title", "deliverable")
)
def test_req_report_7109_row_mutations_fail(case: str) -> None:
    """REQ-REPORT-7109 rejects missing, extra, reordered, duplicated, or changed rows."""

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


def test_scenario_report_7109_markdown_and_gate_syntax_are_strict() -> None:
    """SCENARIO-REPORT-7109-GATES rejects malformed Markdown and YAML gates."""

    with pytest.raises(ValueError, match="task row"):
        mod.parse_markdown_contract(MARKDOWN_CONTRACT.replace("| 1 |", "| 1 | extra |", 1))
    with pytest.raises(ValueError, match="structured gate"):
        mod.parse_markdown_contract(MARKDOWN_CONTRACT.replace(" == 1`", " is ready`", 1))
    with pytest.raises(ValueError, match="section is missing"):
        mod.parse_markdown_contract(MARKDOWN_CONTRACT.replace("## Exact Task Contract", "## Other"))
    with pytest.raises(ValueError, match="task row"):
        mod.parse_markdown_contract(MARKDOWN_CONTRACT.replace("exp7109-v624", "not-an-exp", 1))
    with pytest.raises(ValueError, match="table is missing"):
        mod.parse_markdown_contract("**Milestone:** `2026.09.624`\n\n## Exact Task Contract\n")
    assert len(mod.parse_markdown_contract(MARKDOWN_CONTRACT)["tasks"]) == 12
    roadmap = _roadmap()
    roadmap["tasks"][5]["gated_on"][0]["op"] = None
    assert mod.evaluate_contract(MARKDOWN_CONTRACT, roadmap, set())["passed"] is False


def test_scenario_report_7109_gates_require_earlier_producer_bare_owned_fields() -> None:
    """SCENARIO-REPORT-7109-GATES rejects bad producers and a gated capstone."""

    mutations: tuple[Callable[[dict[str, Any]], None], ...] = (
        lambda value: value["tasks"][5]["gated_on"][0].__setitem__(
            "artifact_field", "nested.value"
        ),
        lambda value: value["tasks"][5]["gated_on"][0].__setitem__(
            "artifact_field", "missing_field"
        ),
        lambda value: value["tasks"][5]["gated_on"][0].__setitem__(
            "upstream", value["tasks"][-1]["id"]
        ),
        lambda value: value["tasks"][-1].__setitem__(
            "gated_on", deepcopy(value["tasks"][5]["gated_on"])
        ),
        lambda value: value["tasks"][5]["gated_on"][0].__setitem__(
            "upstream", value["tasks"][0]["id"]
        ),
    )
    for mutate in mutations:
        roadmap = _roadmap()
        mutate(roadmap)
        assert mod.evaluate_contract(MARKDOWN_CONTRACT, roadmap, set())["passed"] is False


def test_scenario_report_7109_prior_failures_require_all_four_fields() -> None:
    """SCENARIO-REPORT-7109-DISCIPLINE rejects incomplete or mismatched priors."""

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
    assert parsed["tasks"][3]["prior_failure_ids"] == ["exp6198"]
    assert parsed["tasks"][4]["prior_failure_ids"] == ["exp7099"]


def test_scenario_report_7109_execution_rules_fail_closed() -> None:
    """SCENARIO-REPORT-7109-DISCIPLINE checks routes, classes, venues, and tails."""

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
            "prompt", value["tasks"][0]["prompt"].replace("run_date", "execution_date")
        ),
        lambda value: value["tasks"][0].__setitem__(
            "prompt", value["tasks"][0]["prompt"].replace(mod.PROMPT_FINAL_LINE, "")
        ),
        lambda value: value["tasks"][0].__setitem__(
            "prompt", value["tasks"][0]["prompt"].replace("--date {date}", "--date wrong")
        ),
        lambda value: value["tasks"][5].__setitem__(
            "prompt", value["tasks"][5]["prompt"].replace("solve_provenance", "provenance")
        ),
    )
    for mutate in mutations:
        roadmap = _roadmap()
        mutate(roadmap)
        assert mod.evaluate_contract(MARKDOWN_CONTRACT, roadmap, set())["passed"] is False
    assert mod.evaluate_contract(MARKDOWN_CONTRACT, _roadmap(), {"exp7113"})["passed"] is False


def test_scenario_report_7109_model_policy_requires_current_headline_cells() -> None:
    """SCENARIO-REPORT-7109-DISCIPLINE rejects missing or legacy model cells."""

    mutations: tuple[Callable[[dict[str, Any]], None], ...] = (
        lambda value: value["tasks"][4].__setitem__("requires_gpu", False),
        lambda value: value["tasks"][4].__setitem__(
            "prompt", value["tasks"][4]["prompt"].replace("MODEL_SPECS", "models_used")
        ),
        lambda value: value["tasks"][4].__setitem__(
            "prompt", value["tasks"][4]["prompt"].replace(mod.HEADLINE_MODEL, "legacy/model")
        ),
        lambda value: value["tasks"][4].__setitem__(
            "prompt", value["tasks"][4]["prompt"].replace(mod.ARC_LIVE_MODEL, "legacy/model")
        ),
        lambda value: value["tasks"][4].__setitem__(
            "prompt", value["tasks"][4]["prompt"].replace("execute", "list")
        ),
        lambda value: value["tasks"][4].__setitem__(
            "prompt", value["tasks"][4]["prompt"].replace("headline", "secondary")
        ),
        lambda value: value["tasks"][4].__setitem__(
            "prompt",
            value["tasks"][4]["prompt"]
            + "\nMODEL_SPECS may use Qwen3.5-0.8B as the headline fallback.",
        ),
        lambda value: value["tasks"][1].__setitem__(
            "prompt",
            value["tasks"][1]["prompt"].replace("aggregation, or", "model_full_generation, or"),
        ),
        lambda value: value["tasks"][7].__setitem__(
            "prompt", value["tasks"][7]["prompt"].replace(mod.SELF_LEARNING_MODEL, "legacy/model")
        ),
    )
    for mutate in mutations:
        roadmap = _roadmap()
        mutate(roadmap)
        assert mod.evaluate_contract(MARKDOWN_CONTRACT, roadmap, set())["passed"] is False


def test_scenario_report_7109_preflight_separates_terminal_classes(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7109-PREFLIGHT separates positive, mismatch, and no-run."""

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
    assert disqualified["v624_task_contract_conforms_score"] == 0
    assert mod.validate_artifact(disqualified) == []

    (tmp_path / mod.DESIGN_PATH).unlink()
    blocked = mod.build_artifact(tmp_path, "20260907", output_path=tmp_path / "blocked.json")
    assert blocked["verdict_class"] == "blocked"
    assert blocked["inference_substrate_class"] == "blocked_no_run"
    assert blocked["gate_check_summary"]["failed_check"] == "v624_markdown_readable"
    assert mod.validate_artifact(blocked) == []


def test_scenario_report_7109_artifact_validator_rejects_forgery(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7109-ARTIFACT recomputes evidence and headlines."""

    _write_preconditions(tmp_path)
    good = mod.build_artifact(tmp_path, "20260907", output_path=tmp_path / "good.json")
    cases = (
        ("field_principles", {}, "field_principles_invalid"),
        ("inference_substrate", "wrong", "inference_substrate_invalid"),
        ("inference_substrate_class", "blocked_no_run", "inference_substrate_class_invalid"),
        ("execution_venue", "container", "execution_venue_invalid"),
        ("v624_task_contract_conforms_score", 0, "v624_task_contract_conforms_score_invalid"),
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
    assert "v624_task_contract_conforms_score_invalid" in mod.validate_artifact(changed)
    changed = deepcopy(good)
    changed["reproducibility_checksum"] = "sha256:bad"
    assert "reproducibility_checksum_invalid" in mod.validate_artifact(changed)


def test_req_report_7109_cli_parse_and_writable_error_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-REPORT-7109 covers command, parse-error, and writable-path behavior."""

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


def test_req_report_7109_validator_rejects_shapes_and_derives_summaries(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7109-ARTIFACT derives all count and row diagnostics."""

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
    markdown_short["v624_task_contract_conforms_score"] = 0
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


def test_req_report_7109_preconditions_report_missing_yaml_inputs(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7109-PREFLIGHT preserves each missing YAML prerequisite."""

    rows, roadmap, exclusion = mod._preconditions(tmp_path, tmp_path / "result.json")
    assert roadmap is exclusion is None
    assert {row["check"] for row in rows if not row["available"]} >= {
        "v624_markdown_readable",
        "v624_active_yaml_readable",
        "exclusion_manifest_readable",
    }


def test_req_report_7109_yaml_parser_rejects_malformed_shapes() -> None:
    """REQ-REPORT-7109 keeps malformed YAML separate from Markdown parsing."""

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


def test_req_report_7109_helper_boundaries_are_fail_closed(tmp_path: Path) -> None:
    """REQ-REPORT-7109 covers parser and manifest boundary values."""

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


def test_req_report_7109_real_active_roadmap_records_truncation(tmp_path: Path) -> None:
    """REQ-REPORT-7109 records the active six-row V624 mismatch without repair."""

    artifact = mod.build_artifact(ROOT, "20260907", output_path=tmp_path / "real.json")
    assert artifact["verdict_class"] == "disqualified"
    assert artifact["expected_task_count"] == 12
    assert artifact["observed_task_count"] == 6
    assert artifact["v624_task_contract_conforms_score"] == 0
    assert artifact["gate_check_summary"] == {
        "failed_check": "yaml_task_count",
        "expected_value": 12,
        "observed_value": 6,
        "passed": False,
    }
    assert mod.validate_artifact(artifact) == []


def test_req_report_7109_cli_rejects_invalid_artifacts_and_generated_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-REPORT-7109 returns failure for invalid stored or generated data."""

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
