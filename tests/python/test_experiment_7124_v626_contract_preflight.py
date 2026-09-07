"""Tests for REQ-REPORT-7124, the independent V626 contract preflight."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any, Callable

import pytest
import yaml

from carnot import experiment_7124_v626_contract_preflight as mod


ROOT = Path(__file__).resolve().parents[2]

MARKDOWN_CONTRACT = """# V626 design

**Milestone:** 2026.09.626

## Exact task contract

| Order | Task ID | Title | Deliverable | Structured gate |
|---:|---|---|---|---|
| 1 | `exp7124-v626-contract-preflight` | V626 Markdown and YAML task-contract preflight | `results/experiment_7124_v626_contract_preflight.json` | none |
| 2 | `exp7125-v626-source-delta` | V626 execution-time SOTA source and model delta | `results/experiment_7125_v626_source_delta.json` | none |
| 3 | `exp7126-arc-loo-phase-receipt-forensics` | ARC leave-one-game-out phase and receipt forensics | `results/experiment_7126_v626_arc_loo_phase_receipts.json` | none |
| 4 | `exp7127-adapter-withheld-arc-loo-cell` | Ungated adapter-withheld ARC leave-one-game-out cell | `results/experiment_7127_v626_adapter_withheld_arc_loo.json` | none |
| 5 | `exp7128-arc-loo-causal-provenance-audit` | ARC leave-one-game-out causal provenance audit | `results/experiment_7128_v626_arc_loo_causal_audit.json` | none |
| 6 | `exp7129-hardness-controlled-sota-constraint-bank` | Hardness-controlled three-family SOTA constraint bank | `results/experiment_7129_v626_sota_constraint_bank.json` | none |
| 7 | `exp7130-verifier-committed-uncertainty-routing` | Verifier-committed uncertainty routing, gated on Exp7129 bank | `results/experiment_7130_v626_verifier_committed_routing.json` | `exp7129.sota_constraint_bank_ready_score == 1` |
| 8 | `exp7131-model-facing-fixed-schema-csl` | Model-facing fixed-schema continuous self-learning, gated on Exp7129 bank | `results/experiment_7131_v626_model_facing_csl.json` | `exp7129.sota_constraint_bank_ready_score == 1` |
| 9 | `exp7132-directional-memory-portability-audit` | Directional memory portability audit, gated on Exp7131 learning | `results/experiment_7132_v626_memory_portability_audit.json` | `exp7131.model_facing_csl_complete_score == 1` |
| 10 | `exp7133-wcrg-multiscale-sampler-prototype` | WCRG-inspired corrected multiscale sampler prototype | `results/experiment_7133_v626_multiscale_sampler_prototype.json` | none |
| 11 | `exp7134-frustrated-ising-sampler-benchmark` | Frustrated-Ising sampler benchmark, gated on Exp7133 prototype | `results/experiment_7134_v626_multiscale_sampler_benchmark.json` | `exp7133.multiscale_sampler_ready_score == 1` |
| 12 | `exp7135-v626-capstone` | V626 independent evidence matrix and branch disposition | `results/experiment_7135_v626_capstone.json` | none |

## Model contract

The permitted headline repositories are:

- `unsloth/Qwen3.6-35B-A3B-GGUF`
- `unsloth/gemma-4-31B-it-GGUF`
- `unsloth/gemma-4-26B-A4B-it-GGUF`
"""

YAML_ROWS = (
    (
        "exp7124-v626-contract-preflight",
        "V626 Markdown and YAML task-contract preflight",
        "results/experiment_7124_v626_contract_preflight.json",
    ),
    (
        "exp7125-v626-source-delta",
        "V626 execution-time SOTA source and model delta",
        "results/experiment_7125_v626_source_delta.json",
    ),
    (
        "exp7126-arc-loo-phase-receipt-forensics",
        "ARC leave-one-game-out phase and receipt forensics",
        "results/experiment_7126_v626_arc_loo_phase_receipts.json",
    ),
    (
        "exp7127-adapter-withheld-arc-loo-cell",
        "Ungated adapter-withheld ARC leave-one-game-out cell",
        "results/experiment_7127_v626_adapter_withheld_arc_loo.json",
    ),
    (
        "exp7128-arc-loo-causal-provenance-audit",
        "ARC leave-one-game-out causal provenance audit",
        "results/experiment_7128_v626_arc_loo_causal_audit.json",
    ),
    (
        "exp7129-hardness-controlled-sota-constraint-bank",
        "Hardness-controlled three-family SOTA constraint bank",
        "results/experiment_7129_v626_sota_constraint_bank.json",
    ),
    (
        "exp7130-verifier-committed-uncertainty-routing",
        "Verifier-committed uncertainty routing, gated on Exp7129 bank",
        "results/experiment_7130_v626_verifier_committed_routing.json",
    ),
    (
        "exp7131-model-facing-fixed-schema-csl",
        "Model-facing fixed-schema continuous self-learning, gated on Exp7129 bank",
        "results/experiment_7131_v626_model_facing_csl.json",
    ),
    (
        "exp7132-directional-memory-portability-audit",
        "Directional memory portability audit, gated on Exp7131 learning",
        "results/experiment_7132_v626_memory_portability_audit.json",
    ),
    (
        "exp7133-wcrg-multiscale-sampler-prototype",
        "WCRG-inspired corrected multiscale sampler prototype",
        "results/experiment_7133_v626_multiscale_sampler_prototype.json",
    ),
    (
        "exp7134-frustrated-ising-sampler-benchmark",
        "Frustrated-Ising sampler benchmark, gated on Exp7133 prototype",
        "results/experiment_7134_v626_multiscale_sampler_benchmark.json",
    ),
    (
        "exp7135-v626-capstone",
        "V626 independent evidence matrix and branch disposition",
        "results/experiment_7135_v626_capstone.json",
    ),
)

GATES = {
    7130: (7129, "sota_constraint_bank_ready_score"),
    7131: (7129, "sota_constraint_bank_ready_score"),
    7132: (7131, "model_facing_csl_complete_score"),
    7134: (7133, "multiscale_sampler_ready_score"),
}
LOCAL_LLM_TASKS = {7127, 7129, 7130, 7131, 7132}
ARC_LEVEL_TASKS = {7127, 7128}


def _prompt(number: int) -> str:
    """Build one YAML prompt without consulting the Markdown table fixture."""

    substrate = "model_bounded_generation" if number in LOCAL_LLM_TASKS else "aggregation"
    if number in {7133, 7134}:
        substrate = "cpu_exact_solver_or_simulator"
    extra_fields: list[str] = []
    model_text = ""
    if number in LOCAL_LLM_TASKS:
        model_text = (
            "Declare MODEL_SPECS and execute a headline cell with "
            "unsloth/Qwen3.6-35B-A3B-GGUF.\n"
        )
        extra_fields.append("MODEL_SPECS")
    if number in ARC_LEVEL_TASKS:
        extra_fields.extend(("level_rows", "solve_provenance"))
    producer_fields = {
        7129: "sota_constraint_bank_ready_score",
        7131: "model_facing_csl_complete_score",
        7133: "multiscale_sampler_ready_score",
    }
    if number in producer_fields:
        extra_fields.append(producer_fields[number])
    fields = [
        "field_principles for every field below",
        "preconditions_checked",
        "run_date",
        "inference_substrate",
        f"inference_substrate_class ({substrate} or blocked_no_run)",
        "execution_venue (host)",
        "duration_s",
        "source_artifact_hashes",
        "rows",
        *extra_fields,
        "random_seed",
        "reproducibility_checksum",
        "gate_check_summary with failed check, expected value, and observed value",
        "verifier_is_oracle (false)",
        "verdict_class (positive | circular_positive | null | blocked | disqualified | partial)",
        "honest_verdict consistent with verdict_class",
    ]
    return (
        "EXISTING CODE TO READ FIRST:\n"
        f"- {{project_root}}/fixture_{number}.txt\n\n"
        "TASK:\nCompare paired controls and preserve each unit row.\n"
        + model_text
        + "REQUIRED ARTIFACT FIELDS: "
        + "; ".join(fields)
        + ".\n\n"
        + "Run command: cd {project_root} && .venv/bin/python scripts/experiments/"
        + mod.RUN_SCRIPTS[number]
        + " --date {date}\n"
        + mod.PROMPT_FINAL_LINE
        + "\n"
    )


def _roadmap() -> dict[str, Any]:
    """Build YAML task objects from data separate from the Markdown fixture."""

    tasks: list[dict[str, Any]] = []
    for task_id, title, deliverable in YAML_ROWS:
        number = int(task_id[3:7])
        task: dict[str, Any] = {
            "id": task_id,
            "title": title,
            "track": "arc" if number in {7126, 7127, 7128} else "infrastructure",
            "priority": "high",
            "agent_type": "codex",
            "model": "gpt-5.6-sol",
            "requires_gpu": number in LOCAL_LLM_TASKS,
            "max_turns": 40,
            "estimated_wall_time_min": 60,
            "per_unit_rows": True,
            "milestone": "2026.09.626",
            "deliverable": deliverable,
            "prior_failures": [],
            "prompt": _prompt(number),
        }
        if number == 7124:
            task["prior_failures"] = [
                {
                    "experiment_id": "exp7121-v625-contract-preflight",
                    "verdict": "complete_disqualified_v625_markdown_yaml_contract_mismatch",
                    "addressed_by": "V626 checks a literal 12-row contract.",
                    "retire_if_same_verdict": True,
                }
            ]
        if number in GATES:
            upstream_number, field = GATES[number]
            task["gated_on"] = [
                {
                    "upstream": next(
                        row[0] for row in YAML_ROWS if row[0].startswith(f"exp{upstream_number}-")
                    ),
                    "artifact_field": field,
                    "op": "==",
                    "value": 1,
                }
            ]
        tasks.append(task)
    return {
        "milestone": "2026.09.626",
        "milestone_title": "fixture",
        "milestone_doc": "fixture.md",
        "tasks": tasks,
    }


def _write_inputs(root: Path, roadmap: dict[str, Any] | None = None) -> None:
    """Write only temporary prerequisites used by a preflight test."""

    for relative in (mod.DESIGN_PATH, mod.ACTIVE_ROADMAP_PATH, mod.EXCLUSION_PATH):
        (root / relative).parent.mkdir(parents=True, exist_ok=True)
    (root / mod.DESIGN_PATH).write_text(MARKDOWN_CONTRACT, encoding="utf-8")
    (root / mod.ACTIVE_ROADMAP_PATH).write_text(
        yaml.safe_dump(roadmap or _roadmap(), sort_keys=False), encoding="utf-8"
    )
    (root / mod.EXCLUSION_PATH).write_text("retired: []\n", encoding="utf-8")
    for number in range(7124, 7136):
        (root / f"fixture_{number}.txt").write_text("fixture\n", encoding="utf-8")


def test_req_report_7124_spec_defines_contract_and_scenarios() -> None:
    """REQ-REPORT-7124 names all task IDs, fields, and replay cases."""

    text = (ROOT / mod.SPEC_PATH).read_text(encoding="utf-8")
    section = text[text.index("### REQ-REPORT-7124") :]
    for task_id, _title, _deliverable in YAML_ROWS:
        assert task_id in section
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert field in section
    for name in ("PARITY", "GATES", "DISCIPLINE", "PREFLIGHT", "ARTIFACT"):
        assert f"SCENARIO-REPORT-7124-{name}" in section


def test_scenario_report_7124_parity_accepts_independent_rows(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7124-PARITY accepts 12 independently parsed rows."""

    _write_inputs(tmp_path)
    result = mod.evaluate_contract(MARKDOWN_CONTRACT, _roadmap(), set(), root=tmp_path)
    assert result["passed"] is True
    assert len(result["task_contract_rows"]) == 12
    assert len(result["gate_producer_rows"]) == 4
    assert len(result["existing_path_rows"]) == 12
    assert result["markdown_task_rows"] is not result["yaml_task_rows"]
    assert result["markdown_task_rows"][6]["gates"][0]["upstream"] == YAML_ROWS[5][0]


@pytest.mark.parametrize(
    "case", ("missing", "extra", "reordered", "duplicated", "renamed", "title", "deliverable")
)
def test_req_report_7124_row_mutations_fail(case: str, tmp_path: Path) -> None:
    """REQ-REPORT-7124 rejects missing, extra, reordered, duplicated, or renamed rows."""

    _write_inputs(tmp_path)
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
        roadmap["tasks"][0]["id"] += "-renamed"
    elif case == "title":
        roadmap["tasks"][0]["title"] += " changed"
    else:
        roadmap["tasks"][0]["deliverable"] = "results/wrong.json"
    assert mod.evaluate_contract(MARKDOWN_CONTRACT, roadmap, set(), root=tmp_path)["passed"] is False


def test_scenario_report_7124_malformed_rows_and_gates_fail() -> None:
    """SCENARIO-REPORT-7124-GATES rejects malformed Markdown and YAML shapes."""

    for changed in (
        MARKDOWN_CONTRACT.replace("| 1 |", "| one |", 1),
        MARKDOWN_CONTRACT.replace("| 1 |", "| 1 | extra |", 1),
        MARKDOWN_CONTRACT.replace(" == 1`", " is ready`", 1),
    ):
        with pytest.raises(ValueError):
            mod.parse_markdown_contract(changed)
    with pytest.raises(ValueError):
        mod.parse_yaml_contract([])
    malformed = _roadmap()
    malformed["tasks"][0] = "bad"
    with pytest.raises(ValueError):
        mod.parse_yaml_contract(malformed)
    malformed = _roadmap()
    malformed["tasks"][6]["gated_on"] = ["bad"]
    with pytest.raises(ValueError):
        mod.parse_yaml_contract(malformed)


def test_scenario_report_7124_policy_mutations_fail(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7124-DISCIPLINE rejects every requested policy drift."""

    _write_inputs(tmp_path)
    mutations: tuple[Callable[[dict[str, Any]], None], ...] = (
        lambda value: value["tasks"][6]["gated_on"][0].__setitem__(
            "artifact_field", "nested.value"
        ),
        lambda value: value["tasks"][6]["gated_on"][0].__setitem__("artifact_field", "missing"),
        lambda value: value["tasks"][6]["gated_on"][0].__setitem__(
            "upstream", value["tasks"][-1]["id"]
        ),
        lambda value: value["tasks"][3].__setitem__(
            "gated_on", deepcopy(value["tasks"][6]["gated_on"])
        ),
        lambda value: value["tasks"][-1].__setitem__(
            "gated_on", deepcopy(value["tasks"][6]["gated_on"])
        ),
        lambda value: value["tasks"][3].__setitem__("requires", ["exp7126"]),
        lambda value: value["tasks"][0]["prior_failures"][0].__setitem__("addressed_by", ""),
        lambda value: value["tasks"][0].__setitem__("estimated_wall_time_min", 71),
        lambda value: value["tasks"][0].__setitem__("agent_type", "gemini"),
        lambda value: value["tasks"][0].__setitem__("model", "gpt-5.5"),
        lambda value: value["tasks"][0].__setitem__("per_unit_rows", False),
        lambda value: value["tasks"][0].__setitem__(
            "prompt", value["tasks"][0]["prompt"].replace("run_date", "date")
        ),
        lambda value: value["tasks"][0].__setitem__(
            "prompt", value["tasks"][0]["prompt"].replace("for every field below", "")
        ),
        lambda value: value["tasks"][0].__setitem__(
            "prompt", value["tasks"][0]["prompt"].replace("observed value", "observation")
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
            "prompt", value["tasks"][0]["prompt"].replace(mod.PROMPT_FINAL_LINE, "")
        ),
        lambda value: value["tasks"][3].__setitem__(
            "prompt", value["tasks"][3]["prompt"].replace("MODEL_SPECS", "models")
        ),
        lambda value: value["tasks"][3].__setitem__(
            "prompt", value["tasks"][3]["prompt"].replace("headline", "secondary")
        ),
        lambda value: value["tasks"][3].__setitem__(
            "prompt", value["tasks"][3]["prompt"].replace("solve_provenance", "provenance")
        ),
        lambda value: value["tasks"][0].__setitem__(
            "prompt", value["tasks"][0]["prompt"].replace("fixture_7124.txt", "missing.txt")
        ),
    )
    for mutate in mutations:
        roadmap = _roadmap()
        mutate(roadmap)
        assert mod.evaluate_contract(MARKDOWN_CONTRACT, roadmap, set(), root=tmp_path)["passed"] is False
    assert mod.evaluate_contract(MARKDOWN_CONTRACT, _roadmap(), {"exp7129"}, root=tmp_path)["passed"] is False


def test_scenario_report_7124_terminal_artifacts_and_validator(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7124-ARTIFACT separates positive, mismatch, and blocked states."""

    _write_inputs(tmp_path)
    positive = mod.build_artifact(tmp_path, "20260907", output_path=tmp_path / "positive.json")
    assert positive["verdict_class"] == "positive"
    assert positive["v626_task_contract_conforms_score"] == 1
    assert mod.validate_artifact(positive) == []

    short = _roadmap()
    short["tasks"].pop()
    _write_inputs(tmp_path, short)
    disqualified = mod.build_artifact(tmp_path, "20260907", output_path=tmp_path / "bad.json")
    assert disqualified["verdict_class"] == "disqualified"
    assert disqualified["gate_check_summary"]["failed_check"] == "yaml_task_count"
    assert mod.validate_artifact(disqualified) == []

    (tmp_path / mod.DESIGN_PATH).unlink()
    blocked = mod.build_artifact(tmp_path, "20260907", output_path=tmp_path / "blocked.json")
    assert blocked["verdict_class"] == "blocked"
    assert blocked["inference_substrate_class"] == "blocked_no_run"
    assert mod.validate_artifact(blocked) == []

    forged = deepcopy(positive)
    forged["task_contract_rows"][0]["passed"] = False
    forged["reproducibility_checksum"] = mod.reproducibility_checksum(forged)
    assert "v626_task_contract_conforms_score_invalid" in mod.validate_artifact(forged)
    forged = deepcopy(positive)
    forged["reproducibility_checksum"] = "sha256:bad"
    assert "reproducibility_checksum_invalid" in mod.validate_artifact(forged)


def test_req_report_7124_cli_and_boundaries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-REPORT-7124 covers CLI, date, writable, and manifest boundaries."""

    with pytest.raises(Exception, match="YYYYMMDD"):
        mod._date_argument("2026-09-07")
    assert mod.main([]) == 2
    assert mod.main(["--validate-artifact", str(tmp_path / "missing.json")]) == 1
    _write_inputs(tmp_path)
    assert mod.main(["--date", "20260907", "--root", str(tmp_path), "--output", "out.json"]) == 0
    output = tmp_path / "out.json"
    assert mod.main(["--validate-artifact", str(output)]) == 0
    assert json.loads(output.read_text(encoding="utf-8"))["experiment_id"] == 7124

    monkeypatch.setattr(
        mod.tempfile,
        "NamedTemporaryFile",
        lambda **_kwargs: (_ for _ in ()).throw(OSError("read only")),
    )
    assert mod._artifact_path_writable(tmp_path / "x.json") == (False, "OSError: read only")
    assert mod.retired_experiment_ids({"retired": [{"experiment_id": 7129}]}) == {"exp7129"}
    assert mod.retired_experiment_ids([]) == set()


def test_req_report_7124_parser_and_preflight_defenses(tmp_path: Path) -> None:
    """REQ-REPORT-7124 covers malformed inputs and both mismatch summaries."""

    with pytest.raises(ValueError, match="scalar"):
        mod._parse_scalar("[1]")
    with pytest.raises(ValueError, match="milestone"):
        mod.parse_markdown_contract("## Exact task contract\n")
    with pytest.raises(ValueError, match="section"):
        mod.parse_markdown_contract("**Milestone:** 2026.09.626\n")
    bad_id = MARKDOWN_CONTRACT.replace("exp7124-v626-contract-preflight", "bad-id", 1)
    with pytest.raises(ValueError, match="task row"):
        mod.parse_markdown_contract(bad_id)
    with pytest.raises(ValueError, match="table is missing"):
        mod.parse_markdown_contract(
            "**Milestone:** 2026.09.626\n\n## Exact task contract\n\n"
            "| Order | Task ID | Title | Deliverable | Structured gate |\n"
            "|---:|---|---|---|---|\n"
        )
    assert mod._required_block("no required fields") == ""
    assert mod._existing_code_paths("no input section") == []
    with pytest.raises(ValueError, match="tasks list"):
        mod.parse_yaml_contract({"milestone": mod.MILESTONE})
    malformed = _roadmap()
    malformed["tasks"][0]["prior_failures"] = {"bad": "shape"}
    with pytest.raises(ValueError, match="malformed list"):
        mod.parse_yaml_contract(malformed)
    malformed = _roadmap()
    malformed["tasks"][0]["prompt"] = None
    with pytest.raises(ValueError, match="malformed id or prompt"):
        mod.parse_yaml_contract(malformed)

    invalid_yaml = tmp_path / "invalid.yaml"
    invalid_yaml.write_text("[]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="mapping required"):
        mod.load_yaml(invalid_yaml)
    manifest = {
        "other": [],
        "retired": [
            "bad row",
            {"experiment_ids": [7129], "un_retired_experiment_ids": [7129]},
        ],
    }
    assert mod.retired_experiment_ids(manifest) == set()
    assert mod._expected_prompt_tail(9999) is None

    rows, roadmap, exclusion = mod._preconditions(tmp_path / "absent", tmp_path / "out.json")
    assert roadmap is exclusion is None
    assert sum(not row["available"] for row in rows) == 3
    assert (
        mod._contract_failure_summary(
            {"markdown_task_count": 11, "yaml_task_rows": [], "task_contract_rows": []}
        )["failed_check"]
        == "markdown_task_count"
    )
    assert (
        mod._contract_failure_summary(
            {
                "markdown_task_count": 12,
                "yaml_task_rows": [{}] * 12,
                "task_contract_rows": [{"order": 6, "passed": False}],
            }
        )["failed_check"]
        == "task_contract_order_6"
    )

    _write_inputs(tmp_path)
    (tmp_path / mod.DESIGN_PATH).write_text("malformed but readable\n", encoding="utf-8")
    artifact = mod.build_artifact(tmp_path, "20260907", output_path=tmp_path / "malformed.json")
    assert artifact["gate_check_summary"]["failed_check"] == "contract_parse"
    assert mod.validate_artifact(artifact) == []


def test_scenario_report_7124_validator_rejects_forged_boundaries(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7124-ARTIFACT rejects forged metadata and row shapes."""

    _write_inputs(tmp_path)
    good = mod.build_artifact(tmp_path, "20260907", output_path=tmp_path / "good.json")
    cases = (
        ("field_principles", {}, "field_principles_invalid"),
        ("inference_substrate", "shared parser", "inference_substrate_invalid"),
        ("execution_venue", "container", "execution_venue_invalid"),
        ("expected_task_count", 11, "expected_task_count_invalid"),
        ("expected_id_order", [], "expected_id_order_invalid"),
        ("active_roadmap_path", "next.yaml", "active_roadmap_path_invalid"),
        ("staging_file_required_at_execution", True, "staging_file_required_invalid"),
        ("verifier_is_oracle", True, "verifier_is_oracle_invalid"),
        ("random_seed", 0, "random_seed_invalid"),
        ("duration_s", -1, "duration_s_invalid"),
        ("run_date", "2026-09-07", "run_date_invalid"),
        ("observed_task_count", 0, "observed_task_count_invalid"),
        ("observed_id_order", [], "observed_id_order_invalid"),
        ("rows", [], "rows_not_task_contract_rows"),
        ("inference_substrate_class", "blocked_no_run", "inference_substrate_class_invalid"),
        ("gate_check_summary", {}, "gate_check_summary_invalid"),
    )
    for field, value, expected_error in cases:
        changed = deepcopy(good)
        changed[field] = value
        changed["reproducibility_checksum"] = mod.reproducibility_checksum(changed)
        assert expected_error in mod.validate_artifact(changed)

    assert mod.validate_artifact([]) == ["artifact_mapping_required"]
    missing = deepcopy(good)
    missing.pop("rows")
    assert mod.validate_artifact(missing) == ["missing_required_field:rows"]
    for mutate in (
        lambda value: value.__setitem__("markdown_task_rows", None),
        lambda value: value["markdown_task_rows"].pop(),
        lambda value: value["markdown_task_rows"][0].__setitem__("id", "exp9999-bad"),
        lambda value: value["yaml_task_rows"][0].__setitem__("id", "exp9999-bad"),
        lambda value: value.__setitem__("title_parity_rows", []),
    ):
        changed = deepcopy(good)
        mutate(changed)
        assert mod._contract_score_from_artifact(changed) == 0

    blocked = deepcopy(good)
    blocked["preconditions_checked"][0]["available"] = False
    blocked["inference_substrate_class"] = "blocked_no_run"
    blocked["v626_task_contract_conforms_score"] = 0
    blocked["verdict_class"] = "blocked"
    blocked["honest_verdict"] = mod.BLOCKED_VERDICT
    blocked["gate_check_summary"] = {
        "failed_check": "wrong",
        "expected_value": "wrong",
        "observed_value": "wrong",
        "passed": False,
    }
    blocked["reproducibility_checksum"] = mod.reproducibility_checksum(blocked)
    assert "gate_check_summary_invalid" in mod.validate_artifact(blocked)


def test_req_report_7124_cli_rejects_invalid_generated_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-REPORT-7124 returns failure when post-write validation fails."""

    _write_inputs(tmp_path)
    monkeypatch.setattr(mod, "validate_artifact", lambda _artifact: ["forced failure"])
    assert (
        mod.main(["--date", "20260907", "--root", str(tmp_path), "--output", "invalid.json"])
        == 1
    )


def test_scenario_report_7124_artifact_declares_recognized_aggregation(
    tmp_path: Path,
) -> None:
    """SCENARIO-REPORT-7124-ARTIFACT avoids a false live-model classification."""

    from scripts.adversarial_verify import verify_artifact

    _write_inputs(tmp_path)
    artifact = mod.build_artifact(tmp_path, "20260907", output_path=tmp_path / "artifact.json")
    path = tmp_path / "artifact.json"
    path.write_text(json.dumps(artifact), encoding="utf-8")
    report = verify_artifact(path)
    assert report["flags"] == []
