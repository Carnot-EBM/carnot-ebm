"""Tests for the independent V620 active-roadmap contract preflight."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any, Callable

import pytest
import yaml

from carnot import experiment_7076_v620_contract_preflight as mod


ROOT = Path(__file__).resolve().parents[2]

YAML_TASK_ROWS = (
    (
        "exp7076-v620-contract-preflight",
        "V620 Markdown and YAML task-contract preflight",
        "results/experiment_7076_v620_contract_preflight.json",
    ),
    (
        "exp7077-v620-sota-ingestion",
        "V620 verifier, EBM, learning, and hardware source ingestion",
        "results/experiment_7077_v620_sota_ingestion.json",
    ),
    (
        "exp7078-gpu-lease-journal-migration",
        "GPU lease journal schema migration and stale-owner recovery",
        "results/experiment_7078_v620_gpu_lease_migration.json",
    ),
    (
        "exp7079-gpu-lease-fresh-process-audit",
        "Fresh-process dual-GPU lease compatibility audit",
        "results/experiment_7079_v620_gpu_lease_audit.json",
    ),
    (
        "exp7080-recovered-three-family-entrance-bank",
        "Recovered three-family SOTA entrance proposal bank",
        "results/experiment_7080_v620_three_family_entrance_bank.json",
    ),
    (
        "exp7081-entrance-bank-set-sufficiency-audit",
        "Set-level entrance-bank sufficiency and conflict audit",
        "results/experiment_7081_v620_entrance_bank_sufficiency_audit.json",
    ),
    (
        "exp7082-entrance-energy-likelihood-controls",
        "Entrance energy versus likelihood and structural controls",
        "results/experiment_7082_v620_entrance_energy_controls.json",
    ),
    (
        "exp7083-entrance-ising-degree16-parity",
        "Entrance QUBO, Ising, distribution, and degree-16 parity",
        "results/experiment_7083_v620_entrance_ising_degree16_parity.json",
    ),
    (
        "exp7084-large-immutable-bcit-stream",
        "Large immutable exact-outcome BCIT stream",
        "results/experiment_7084_v620_large_bcit_stream.json",
    ),
    (
        "exp7085-bcit-prospective-self-learning",
        "Prospective context-bound continuous self-learning comparison",
        "results/experiment_7085_v620_bcit_self_learning.json",
    ),
    (
        "exp7086-bcit-cold-drift-audit",
        "Fresh-process BCIT retention, drift, and rollback audit",
        "results/experiment_7086_v620_bcit_cold_drift_audit.json",
    ),
    (
        "exp7087-single-credit-arc-supervisor-refinement",
        "Single-credit ARC supervisor redirect refinement",
        "results/experiment_7087_v620_arc_supervisor_refinement.json",
    ),
    (
        "exp7088-v620-capstone",
        "V620 evidence matrix and release-or-retire handoff",
        "results/experiment_7088_v620_capstone.json",
    ),
)

MARKDOWN_CONTRACT = """# Independent V620 design fixture

**Milestone:** 2026.09.620

## Exact Task Contract

| Order | Task ID | Title | Deliverable | Structured gate |
|---:|---|---|---|---|
| 1 | `exp7076-v620-contract-preflight` | V620 Markdown and YAML task-contract preflight | `results/experiment_7076_v620_contract_preflight.json` | none |
| 2 | `exp7077-v620-sota-ingestion` | V620 verifier, EBM, learning, and hardware source ingestion | `results/experiment_7077_v620_sota_ingestion.json` | none |
| 3 | `exp7078-gpu-lease-journal-migration` | GPU lease journal schema migration and stale-owner recovery | `results/experiment_7078_v620_gpu_lease_migration.json` | none |
| 4 | `exp7079-gpu-lease-fresh-process-audit` | Fresh-process dual-GPU lease compatibility audit | `results/experiment_7079_v620_gpu_lease_audit.json` | `exp7078-gpu-lease-journal-migration.gpu_lease_compatibility_ready_score == 1` |
| 5 | `exp7080-recovered-three-family-entrance-bank` | Recovered three-family SOTA entrance proposal bank | `results/experiment_7080_v620_three_family_entrance_bank.json` | `exp7079-gpu-lease-fresh-process-audit.gpu_lease_cold_audit_ready_score == 1` |
| 6 | `exp7081-entrance-bank-set-sufficiency-audit` | Set-level entrance-bank sufficiency and conflict audit | `results/experiment_7081_v620_entrance_bank_sufficiency_audit.json` | `exp7080-recovered-three-family-entrance-bank.entrance_proposal_bank_complete_score == 1` |
| 7 | `exp7082-entrance-energy-likelihood-controls` | Entrance energy versus likelihood and structural controls | `results/experiment_7082_v620_entrance_energy_controls.json` | `exp7081-entrance-bank-set-sufficiency-audit.entrance_support_audit_ready_score == 1` AND `entrance_selector_headroom_ready_score == 1` |
| 8 | `exp7083-entrance-ising-degree16-parity` | Entrance QUBO, Ising, distribution, and degree-16 parity | `results/experiment_7083_v620_entrance_ising_degree16_parity.json` | `exp7082-entrance-energy-likelihood-controls.entrance_energy_comparison_complete_score == 1` |
| 9 | `exp7084-large-immutable-bcit-stream` | Large immutable exact-outcome BCIT stream | `results/experiment_7084_v620_large_bcit_stream.json` | none |
| 10 | `exp7085-bcit-prospective-self-learning` | Prospective context-bound continuous self-learning comparison | `results/experiment_7085_v620_bcit_self_learning.json` | `exp7084-large-immutable-bcit-stream.bcit_stream_ready_score == 1` |
| 11 | `exp7086-bcit-cold-drift-audit` | Fresh-process BCIT retention, drift, and rollback audit | `results/experiment_7086_v620_bcit_cold_drift_audit.json` | `exp7085-bcit-prospective-self-learning.bcit_comparison_complete_score == 1` |
| 12 | `exp7087-single-credit-arc-supervisor-refinement` | Single-credit ARC supervisor redirect refinement | `results/experiment_7087_v620_arc_supervisor_refinement.json` | none |
| 13 | `exp7088-v620-capstone` | V620 evidence matrix and release-or-retire handoff | `results/experiment_7088_v620_capstone.json` | none |

### Exp7076 - V620 Markdown and YAML task-contract preflight

This scope records `exp7050-v618-active-contract-preflight` as a prior failure.

### Exp7079 - Fresh-process dual-GPU lease compatibility audit
**Gate:**
`exp7078-gpu-lease-journal-migration.gpu_lease_compatibility_ready_score == 1`

### Exp7080 - Recovered three-family SOTA entrance proposal bank
**Gate:**
`exp7079-gpu-lease-fresh-process-audit.gpu_lease_cold_audit_ready_score == 1`

### Exp7081 - Set-level entrance-bank sufficiency and conflict audit
**Gate:**
`exp7080-recovered-three-family-entrance-bank.entrance_proposal_bank_complete_score == 1`

### Exp7082 - Entrance energy versus likelihood and structural controls
**Gates:**
- `exp7081-entrance-bank-set-sufficiency-audit.entrance_support_audit_ready_score == 1`
- `exp7081-entrance-bank-set-sufficiency-audit.entrance_selector_headroom_ready_score == 1`

### Exp7083 - Entrance QUBO, Ising, distribution, and degree-16 parity
**Gate:**
`exp7082-entrance-energy-likelihood-controls.entrance_energy_comparison_complete_score == 1`

### Exp7085 - Prospective context-bound continuous self-learning comparison
**Gate:**
`exp7084-large-immutable-bcit-stream.bcit_stream_ready_score == 1`

### Exp7086 - Fresh-process BCIT retention, drift, and rollback audit
**Gate:**
`exp7085-bcit-prospective-self-learning.bcit_comparison_complete_score == 1`
"""

GATES = {
    7079: [(7078, "gpu_lease_compatibility_ready_score")],
    7080: [(7079, "gpu_lease_cold_audit_ready_score")],
    7081: [(7080, "entrance_proposal_bank_complete_score")],
    7082: [
        (7081, "entrance_support_audit_ready_score"),
        (7081, "entrance_selector_headroom_ready_score"),
    ],
    7083: [(7082, "entrance_energy_comparison_complete_score")],
    7085: [(7084, "bcit_stream_ready_score")],
    7086: [(7085, "bcit_comparison_complete_score")],
}
PRODUCED_FIELDS = {
    7078: ("gpu_lease_compatibility_ready_score",),
    7079: ("gpu_lease_cold_audit_ready_score",),
    7080: ("entrance_proposal_bank_complete_score",),
    7081: ("entrance_support_audit_ready_score", "entrance_selector_headroom_ready_score"),
    7082: ("entrance_energy_comparison_complete_score",),
    7084: ("bcit_stream_ready_score",),
    7085: ("bcit_comparison_complete_score",),
}
LIVE_TASKS = {7080}


def _prompt(number: int) -> str:
    """Create one YAML-only prompt fixture without using Markdown task data."""

    substrate = (
        "live_local_sota_gguf_cuda_llamacpp" if number in LIVE_TASKS else mod.INFERENCE_SUBSTRATE
    )
    fields = [
        "field_principles",
        "preconditions_checked",
        f"inference_substrate ({substrate})",
        "duration_s",
        "rows",
        "verdict_class (positive | circular_positive | null | blocked | disqualified | partial)",
        "random_seed",
        "reproducibility_checksum",
        "gate_check_summary with failed check, expected value, and observed value",
        *PRODUCED_FIELDS.get(number, ()),
        "honest_verdict with a terminal prefix",
    ]
    lines = ["Compare every per-unit row under matched controls."]
    if number in LIVE_TASKS:
        fields.insert(-1, "MODEL_SPECS")
        lines.append("MODEL_SPECS resolves through cached_sota_pair().")
        lines.extend(f"Use {model}." for model in mod.EXPECTED_LLM_MODELS[number])
        lines.append("No legacy-small headline fallback is allowed.")
    lines.extend(
        [
            "REQUIRED ARTIFACT FIELDS: " + "; ".join(fields) + ".",
            "Run command: fixture",
            mod.PROMPT_TAIL,
        ]
    )
    return "\n".join(lines)


def _roadmap() -> dict[str, Any]:
    """Build the YAML fixture from its own literal task list."""

    tasks = []
    for task_id, title, deliverable in YAML_TASK_ROWS:
        number = int(task_id[3:7])
        task: dict[str, Any] = {
            "id": task_id,
            "title": title,
            "track": "infrastructure",
            "priority": "critical",
            "agent_type": "claude",
            "model": "opus",
            "requires_gpu": number in LIVE_TASKS,
            "max_turns": 100,
            "estimated_wall_time_min": 30,
            "per_unit_rows": True,
            "milestone": mod.MILESTONE,
            "deliverable": deliverable,
            "prompt": _prompt(number),
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
        if number == 7076:
            task["prior_failures"] = [
                {
                    "experiment_id": "exp7050-v618-active-contract-preflight",
                    "verdict": "complete_disqualified_v618_markdown_yaml_contract_mismatch",
                    "addressed_by": "V620 publishes and audits the same 13 rows in both files.",
                    "retire_if_same_verdict": True,
                }
            ]
        tasks.append(task)
    return {
        "milestone": mod.MILESTONE,
        "milestone_title": "fixture",
        "milestone_doc": str(mod.DESIGN_PATH),
        "tasks": tasks,
    }


def _write_preconditions(root: Path, roadmap: dict[str, Any] | None = None) -> None:
    """Create only the execution inputs used by the audit."""

    values = {
        mod.DESIGN_PATH: MARKDOWN_CONTRACT,
        mod.ACTIVE_ROADMAP_PATH: yaml.safe_dump(roadmap or _roadmap()),
        mod.EXCLUSION_PATH: "retired: []\nretired_extras: []\n",
    }
    for relative, content in values.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")


def test_req_report_7076_spec_defines_thirteen_task_contract() -> None:
    """REQ-REPORT-7076 names the exact count, rows, and required scenarios."""

    spec = (ROOT / mod.SPEC_PATH).read_text(encoding="utf-8")
    section = spec[spec.index("### REQ-REPORT-7076") :]
    assert mod.EXPECTED_TASK_COUNT == len(YAML_TASK_ROWS) == 13
    assert mod.EXPECTED_TASK_IDS == tuple(row[0] for row in YAML_TASK_ROWS)
    for scenario in ("PARITY", "GATES", "DISCIPLINE", "PREFLIGHT", "ARTIFACT"):
        assert f"SCENARIO-REPORT-7076-{scenario}" in section
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert field in section


def test_scenario_report_7076_parity_accepts_independent_rows() -> None:
    """SCENARIO-REPORT-7076-PARITY accepts exactly 13 independent rows."""

    result = mod.evaluate_contract(MARKDOWN_CONTRACT, _roadmap(), set())
    assert result["passed"] is True
    assert len(result["task_contract_rows"]) == 13
    assert len(result["gate_producer_rows"]) == 8
    assert result["markdown_task_rows"] is not result["yaml_task_rows"]
    assert all(row["passed"] for row in result["task_contract_rows"])


def _missing_row(roadmap: dict[str, Any]) -> None:
    roadmap["tasks"].pop(5)


def _reordered_row(roadmap: dict[str, Any]) -> None:
    roadmap["tasks"][5], roadmap["tasks"][6] = roadmap["tasks"][6], roadmap["tasks"][5]


def _changed_title(roadmap: dict[str, Any]) -> None:
    roadmap["tasks"][0]["title"] += " changed"


def _changed_deliverable(roadmap: dict[str, Any]) -> None:
    roadmap["tasks"][0]["deliverable"] = "results/wrong.json"


def _missing_gate_producer_field(roadmap: dict[str, Any]) -> None:
    roadmap["tasks"][2]["prompt"] = roadmap["tasks"][2]["prompt"].replace(
        "gpu_lease_compatibility_ready_score", "different_ready_score"
    )


def _malformed_prior(roadmap: dict[str, Any]) -> None:
    roadmap["tasks"][0]["prior_failures"][0].pop("addressed_by")


def _missing_prompt_tail(roadmap: dict[str, Any]) -> None:
    roadmap["tasks"][0]["prompt"] = roadmap["tasks"][0]["prompt"].replace(mod.PROMPT_TAIL, "")


def _gated_capstone(roadmap: dict[str, Any]) -> None:
    roadmap["tasks"][-1]["gated_on"] = deepcopy(roadmap["tasks"][3]["gated_on"])


@pytest.mark.parametrize(
    ("case", "mutation"),
    (
        ("missing row", _missing_row),
        ("reordered row", _reordered_row),
        ("changed title", _changed_title),
        ("changed deliverable", _changed_deliverable),
        ("missing producer field", _missing_gate_producer_field),
        ("malformed prior", _malformed_prior),
        ("missing prompt tail", _missing_prompt_tail),
        ("gated capstone", _gated_capstone),
    ),
)
def test_req_report_7076_required_mutations_fail(
    case: str, mutation: Callable[[dict[str, Any]], None]
) -> None:
    """REQ-REPORT-7076 rejects every named contract mutation."""

    roadmap = _roadmap()
    mutation(roadmap)
    assert mod.evaluate_contract(MARKDOWN_CONTRACT, roadmap, set())["passed"] is False, case


def test_scenario_report_7076_gates_and_discipline_fail_closed() -> None:
    """SCENARIO-REPORT-7076-GATES rejects identity, model, and prompt defects."""

    mutations: tuple[Callable[[dict[str, Any]], None], ...] = (
        lambda value: value["tasks"][3]["gated_on"][0].__setitem__(
            "artifact_field", "nested.value"
        ),
        lambda value: value["tasks"][3]["gated_on"][0].__setitem__(
            "upstream", value["tasks"][-1]["id"]
        ),
        lambda value: value["tasks"][0].__setitem__("per_unit_rows", False),
        lambda value: value["tasks"][4].__setitem__(
            "prompt", value["tasks"][4]["prompt"].replace("cached_sota_pair()", "direct lookup")
        ),
        lambda value: value["tasks"][4].__setitem__(
            "prompt", value["tasks"][4]["prompt"] + "\nHeadline fallback: Qwen3.5-0.8B."
        ),
        lambda value: value["tasks"][4].__setitem__(
            "prompt",
            value["tasks"][4]["prompt"].replace("unsloth/Qwen3.6-35B-A3B-GGUF", "legacy/model"),
        ),
        lambda value: value["tasks"][4].__setitem__(
            "prompt", value["tasks"][4]["prompt"].replace("MODEL_SPECS", "models")
        ),
        lambda value: value["tasks"][4].__setitem__(
            "prompt",
            value["tasks"][4]["prompt"].replace(
                "No legacy-small headline fallback is allowed.", "Fallback policy is unspecified."
            ),
        ),
        lambda value: value["tasks"][0].__setitem__("model", "sonnet-typo"),
        lambda value: value["tasks"][0].__setitem__(
            "prompt", value["tasks"][0]["prompt"].replace(" | partial", " | deferred")
        ),
        lambda value: value["tasks"][0].__setitem__(
            "prompt", value["tasks"][0]["prompt"].replace("observed value", "observation")
        ),
    )
    for mutation in mutations:
        roadmap = _roadmap()
        mutation(roadmap)
        assert mod.evaluate_contract(MARKDOWN_CONTRACT, roadmap, set())["passed"] is False
    assert mod.evaluate_contract(MARKDOWN_CONTRACT, _roadmap(), {"exp7077"})["passed"] is False


def test_scenario_report_7076_prior_failure_requires_all_four_fields() -> None:
    """SCENARIO-REPORT-7076-DISCIPLINE rejects each incomplete prior record."""

    replacements: tuple[Callable[[dict[str, Any]], None], ...] = (
        lambda prior: prior.__setitem__("experiment_id", ""),
        lambda prior: prior.__setitem__("verdict", ""),
        lambda prior: prior.__setitem__("addressed_by", ""),
        lambda prior: prior.__setitem__("retire_if_same_verdict", False),
    )
    for mutate in replacements:
        roadmap = _roadmap()
        mutate(roadmap["tasks"][0]["prior_failures"][0])
        assert mod.evaluate_contract(MARKDOWN_CONTRACT, roadmap, set())["passed"] is False


def test_scenario_report_7076_prior_failure_identities_match_markdown() -> None:
    """SCENARIO-REPORT-7076-DISCIPLINE compares independently parsed prior IDs."""

    roadmap = _roadmap()
    roadmap["tasks"][0]["prior_failures"] = []
    assert mod.evaluate_contract(MARKDOWN_CONTRACT, roadmap, set())["passed"] is False

    changed_markdown = MARKDOWN_CONTRACT.replace(
        "exp7050-v618-active-contract-preflight", "exp9999-wrong-prior"
    )
    assert mod.evaluate_contract(changed_markdown, _roadmap(), set())["passed"] is False


def test_scenario_report_7076_local_llm_detection_uses_substrate() -> None:
    """SCENARIO-REPORT-7076-DISCIPLINE applies model policy to each local LLM."""

    roadmap = _roadmap()
    roadmap["tasks"][1]["prompt"] = roadmap["tasks"][1]["prompt"].replace(
        mod.INFERENCE_SUBSTRATE, "live_local_sota_gguf_cuda_llamacpp"
    )
    assert mod.evaluate_contract(MARKDOWN_CONTRACT, roadmap, set())["passed"] is False


def test_scenario_report_7076_preflight_builds_all_terminal_classes(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7076-PREFLIGHT separates blocked and disqualified states."""

    _write_preconditions(tmp_path)
    positive = mod.build_artifact(
        tmp_path, "20260906", output_path=tmp_path / "results/positive.json"
    )
    assert positive["verdict_class"] == "positive"
    assert positive["v620_task_contract_conforms_score"] == 1
    assert mod.validate_artifact(positive) == []

    changed = _roadmap()
    _changed_title(changed)
    _write_preconditions(tmp_path, changed)
    disqualified = mod.build_artifact(
        tmp_path, "20260906", output_path=tmp_path / "results/bad.json"
    )
    assert disqualified["verdict_class"] == "disqualified"
    assert disqualified["v620_task_contract_conforms_score"] == 0
    assert mod.validate_artifact(disqualified) == []

    (tmp_path / mod.DESIGN_PATH).unlink()
    blocked = mod.build_artifact(
        tmp_path, "20260906", output_path=tmp_path / "results/blocked.json"
    )
    assert blocked["verdict_class"] == "blocked"
    assert blocked["gate_check_summary"]["failed_check"] == "v620_markdown_readable"
    assert blocked["gate_check_summary"]["expected_value"] == "readable_nonempty_source"
    assert "FileNotFoundError" in blocked["gate_check_summary"]["observed_value"]
    assert mod.validate_artifact(blocked) == []


def test_scenario_report_7076_artifact_rejects_forgery(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7076-ARTIFACT derives the score, verdict, and checksum."""

    _write_preconditions(tmp_path)
    good = mod.build_artifact(tmp_path, "20260906", output_path=tmp_path / "good.json")
    changed = deepcopy(good)
    changed["task_contract_rows"][0]["passed"] = False
    changed["reproducibility_checksum"] = mod.reproducibility_checksum(changed)
    assert "v620_task_contract_conforms_score_invalid" in mod.validate_artifact(changed)
    changed = deepcopy(good)
    changed["reproducibility_checksum"] = "sha256:bad"
    assert "reproducibility_checksum_invalid" in mod.validate_artifact(changed)

    (tmp_path / mod.DESIGN_PATH).unlink()
    blocked = mod.build_artifact(tmp_path, "20260906", output_path=tmp_path / "blocked.json")
    blocked["gate_check_summary"]["failed_check"] = "wrong_prerequisite"
    blocked["reproducibility_checksum"] = mod.reproducibility_checksum(blocked)
    assert "gate_check_summary_not_derived" in mod.validate_artifact(blocked)


def test_req_report_7076_parsers_and_ledgers_are_strict(tmp_path: Path) -> None:
    """REQ-REPORT-7076 rejects malformed sources and normalizes ledger IDs."""

    parsed = mod.parse_markdown_contract(MARKDOWN_CONTRACT)
    assert parsed["tasks"][0]["gates"] == []
    assert mod.task_number("bad") is None
    with pytest.raises(ValueError, match="milestone"):
        mod.parse_markdown_contract("no contract")
    with pytest.raises(ValueError, match="task contract section"):
        mod.parse_markdown_contract(f"**Milestone:** {mod.MILESTONE}")
    with pytest.raises(ValueError, match="task row"):
        mod.parse_markdown_contract(MARKDOWN_CONTRACT.replace("| 1 |", "| 1 | extra |", 1))
    with pytest.raises(ValueError, match="task row"):
        mod.parse_markdown_contract(MARKDOWN_CONTRACT.replace("`exp7076-v620", "`bad-v620", 1))
    with pytest.raises(ValueError, match="task table"):
        mod.parse_markdown_contract(
            f"**Milestone:** {mod.MILESTONE}\n## Exact Task Contract\nno rows\n"
        )
    with pytest.raises(ValueError, match="gate"):
        mod.parse_markdown_contract(MARKDOWN_CONTRACT.replace(" == 1`", " is ready`", 1))
    with pytest.raises(ValueError, match="no producer"):
        mod._parse_markdown_gate_cell("ready_score == 1")
    table_text, section_text = MARKDOWN_CONTRACT.rsplit("gpu_lease_compatibility_ready_score", 1)
    with pytest.raises(ValueError, match="representations disagree"):
        mod.parse_markdown_contract(table_text + "different_ready_score" + section_text)
    table_text, section_text = MARKDOWN_CONTRACT.rsplit(" == 1`", 1)
    with pytest.raises(ValueError, match="structured gate"):
        mod.parse_markdown_contract(table_text + " is ready`" + section_text)
    no_marker = MARKDOWN_CONTRACT + "\n### Exp7076 - no gate marker\nNo structured gate.\n"
    assert len(mod.parse_markdown_contract(no_marker)["tasks"]) == 13
    unknown_section = MARKDOWN_CONTRACT + (
        "\n### Exp9999 - outside table\n**Gate:**\n`exp7076-v620-contract-preflight.ready == 1`\n"
    )
    assert len(mod.parse_markdown_contract(unknown_section)["tasks"]) == 13
    with pytest.raises(ValueError, match="gate block"):
        mod.parse_markdown_contract(
            MARKDOWN_CONTRACT + "\n### Exp7076 - empty gate\n**Gate:**\nno value\n"
        )
    for value, message in (([], "mapping"), ({}, "tasks list"), ({"tasks": [1]}, "mapping")):
        with pytest.raises(ValueError, match=message):
            mod.parse_yaml_contract(value)
    path = tmp_path / "bad.yaml"
    path.write_text("[]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="mapping required"):
        mod.load_yaml(path)
    assert mod.retired_experiment_ids({"retired": [{"experiment_id": 7076}]}) == {"exp7076"}


def test_req_report_7076_precondition_and_parse_error_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-REPORT-7076-PREFLIGHT preserves unavailable input reasons."""

    rows, roadmap, exclusion = mod._preconditions(tmp_path, tmp_path / "results/result.json")
    assert roadmap is exclusion is None
    assert {row["check"] for row in rows if not row["available"]} >= {
        "v620_markdown_readable",
        "v620_active_yaml_readable",
        "exclusion_manifest_readable",
    }

    _write_preconditions(tmp_path)
    (tmp_path / mod.STAGING_ROADMAP_PATH).write_text("not parsed\n", encoding="utf-8")
    rows, roadmap, exclusion = mod._preconditions(tmp_path, tmp_path / "results/result.json")
    assert roadmap and exclusion
    staging = next(row for row in rows if row["check"] == "staging_file_not_required")
    assert staging["observed_value"] == "present_but_not_read"

    (tmp_path / mod.ACTIVE_ROADMAP_PATH).write_text("[]\n", encoding="utf-8")
    rows, roadmap, _exclusion = mod._preconditions(tmp_path, tmp_path / "results/result.json")
    assert roadmap is None
    assert (
        next(row for row in rows if row["check"] == "v620_active_yaml_readable")["available"]
        is False
    )

    _write_preconditions(tmp_path)
    (tmp_path / mod.DESIGN_PATH).write_text("readable but malformed\n", encoding="utf-8")
    artifact = mod.build_artifact(
        tmp_path, "20260906", output_path=tmp_path / "results/malformed.json"
    )
    assert artifact["verdict_class"] == "disqualified"
    assert artifact["gate_check_summary"]["failed_check"] == "contract_parse"
    assert mod.validate_artifact(artifact) == []

    monkeypatch.setattr(
        mod.tempfile,
        "NamedTemporaryFile",
        lambda **_kwargs: (_ for _ in ()).throw(OSError("read only")),
    )
    assert mod._artifact_path_writable(tmp_path / "blocked.json") == (
        False,
        "OSError: read only",
    )


def test_req_report_7076_failure_summaries_name_count_source(tmp_path: Path) -> None:
    """REQ-REPORT-7076 reports Markdown and active-YAML count drift separately."""

    assert (
        mod._contract_failure_summary(
            {"markdown_task_count": 12, "yaml_task_rows": [], "task_contract_rows": []}
        )["failed_check"]
        == "markdown_task_count"
    )
    assert (
        mod._contract_failure_summary(
            {"markdown_task_count": 13, "yaml_task_rows": [], "task_contract_rows": []}
        )["failed_check"]
        == "yaml_task_count"
    )

    _write_preconditions(tmp_path)
    design = tmp_path / mod.DESIGN_PATH
    design.write_text(
        MARKDOWN_CONTRACT.replace("| 13 | `exp7088", "| x | `exp7088"), encoding="utf-8"
    )
    markdown_short = mod.build_artifact(
        tmp_path, "20260906", output_path=tmp_path / "markdown-short.json"
    )
    assert markdown_short["gate_check_summary"]["failed_check"] == "markdown_task_count"
    assert mod.validate_artifact(markdown_short) == []

    _write_preconditions(tmp_path, _roadmap())
    roadmap = _roadmap()
    _missing_row(roadmap)
    _write_preconditions(tmp_path, roadmap)
    yaml_short = mod.build_artifact(tmp_path, "20260906", output_path=tmp_path / "yaml-short.json")
    assert yaml_short["gate_check_summary"]["failed_check"] == "yaml_task_count"
    assert mod.validate_artifact(yaml_short) == []


def test_scenario_report_7076_artifact_validator_covers_each_headline(
    tmp_path: Path,
) -> None:
    """SCENARIO-REPORT-7076-ARTIFACT rejects every forged headline field."""

    _write_preconditions(tmp_path)
    good = mod.build_artifact(tmp_path, "20260906", output_path=tmp_path / "good.json")
    cases = (
        ("field_principles", {}, "field_principles_invalid"),
        ("inference_substrate", "wrong", "inference_substrate_invalid"),
        ("verifier_is_oracle", True, "verifier_is_oracle_invalid"),
        ("expected_task_count", 12, "expected_task_count_invalid"),
        ("expected_id_order", [], "expected_id_order_invalid"),
        ("active_roadmap_path", "research-roadmap-next.yaml", "active_roadmap_path_invalid"),
        ("staging_file_required_at_execution", True, "staging_file_required_invalid"),
        ("random_seed", 0, "random_seed_invalid"),
        ("duration_s", -1, "duration_s_invalid"),
        ("observed_task_count", 0, "observed_task_count_invalid"),
        ("observed_id_order", [], "observed_id_order_invalid"),
        ("rows", [], "rows_not_task_contract_rows"),
        ("verdict_class", "blocked", "verdict_class_not_derived"),
        ("honest_verdict", "complete_blocked_wrong", "honest_verdict_not_derived"),
        ("gate_check_summary", [], "gate_check_summary_invalid"),
    )
    for field, value, expected in cases:
        changed = deepcopy(good)
        changed[field] = value
        changed["reproducibility_checksum"] = mod.reproducibility_checksum(changed)
        assert expected in mod.validate_artifact(changed), field
    missing = deepcopy(good)
    missing.pop("rows")
    assert mod.validate_artifact(missing) == ["missing_required_field:rows"]


def test_req_report_7076_score_rejects_each_evidence_shape(tmp_path: Path) -> None:
    """REQ-REPORT-7076 recomputes conformance from every stored evidence group."""

    _write_preconditions(tmp_path)
    good = mod.build_artifact(tmp_path, "20260906", output_path=tmp_path / "good.json")
    mutations: tuple[Callable[[dict[str, Any]], None], ...] = (
        lambda value: value["markdown_task_rows"][0].__setitem__("id", "exp9999-wrong"),
        lambda value: value["yaml_task_rows"][0].__setitem__("id", "exp9999-wrong"),
        lambda value: value["task_contract_rows"][0].__setitem__("order", 99),
        lambda value: value.__setitem__("title_parity_rows", []),
        lambda value: value["title_parity_rows"][0].__setitem__("passed", False),
        lambda value: value["gate_producer_rows"].pop(),
        lambda value: value.__setitem__("prior_failure_rows", []),
        lambda value: value.__setitem__("prior_failure_rows", value["prior_failure_rows"][:12]),
    )
    for mutation in mutations:
        changed = deepcopy(good)
        mutation(changed)
        assert mod._contract_score_from_artifact(changed) == 0


def test_req_report_7076_cli_validation_and_generation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-REPORT-7076 exposes deterministic validate and generate commands."""

    with pytest.raises(Exception, match="YYYYMMDD"):
        mod._date_argument("2026-09-06")
    with pytest.raises(SystemExit):
        mod.main([])
    assert mod.main(["--validate-artifact", str(tmp_path / "missing.json")]) == 1
    list_path = tmp_path / "list.json"
    list_path.write_text("[]\n", encoding="utf-8")
    assert mod.main(["--validate-artifact", str(list_path)]) == 1
    bad_path = tmp_path / "bad.json"
    bad_path.write_text("{}\n", encoding="utf-8")
    assert mod.main(["--validate-artifact", str(bad_path)]) == 1
    _write_preconditions(tmp_path)
    output = Path("results/result.json")
    assert mod.main(["--date", "20260906", "--root", str(tmp_path), "--output", str(output)]) == 0
    artifact_path = tmp_path / output
    assert json.loads(artifact_path.read_text(encoding="utf-8"))["verdict_class"] == "positive"
    assert mod.main(["--validate-artifact", str(artifact_path)]) == 0
    monkeypatch.setattr(mod, "validate_artifact", lambda _artifact: ["forced failure"])
    assert (
        mod.main(
            [
                "--date",
                "20260906",
                "--root",
                str(tmp_path),
                "--output",
                str(tmp_path / "invalid.json"),
            ]
        )
        == 1
    )
