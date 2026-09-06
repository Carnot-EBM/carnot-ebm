"""Tests for the independent V619 active-roadmap contract preflight."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any, Callable

import pytest
import yaml

from carnot import experiment_7063_v619_contract_preflight as mod


ROOT = Path(__file__).resolve().parents[2]

YAML_TASK_ROWS = (
    (
        "exp7063-v619-contract-preflight",
        "V619 Markdown and YAML task-contract preflight",
        "results/experiment_7063_v619_contract_preflight.json",
    ),
    (
        "exp7064-exact-entrance-constraint-fixture",
        "Exact source-grouped entrance constraint fixture",
        "results/experiment_7064_v619_exact_entrance_fixture.json",
    ),
    (
        "exp7065-three-family-entrance-proposal-bank",
        "Three-family SOTA entrance proposal bank",
        "results/experiment_7065_v619_three_family_entrance_bank.json",
    ),
    (
        "exp7066-entrance-bank-independent-audit",
        "Cold recomputation of entrance-bank support",
        "results/experiment_7066_v619_entrance_bank_audit.json",
    ),
    (
        "exp7067-hopfield-entrance-energy-selection",
        "Hopfield-style entrance energy selection comparison",
        "results/experiment_7067_v619_entrance_energy_selection.json",
    ),
    (
        "exp7068-hierarchical-branch-fidelity-control",
        "Lossless categorical mass-rebalancing evaluation",
        "results/experiment_7068_v619_hierarchical_branch_control.json",
    ),
    (
        "exp7069-context-bound-experience-contract",
        "BCIT use-validate-reject state machine",
        "results/experiment_7069_v619_context_authorization_contract.json",
    ),
    (
        "exp7070-bcit-prospective-self-learning",
        "Prospective context-bound continuous self-learning comparison",
        "results/experiment_7070_v619_bcit_self_learning.json",
    ),
    (
        "exp7071-bcit-drift-rollback-audit",
        "Fresh-process self-learning drift and rollback audit",
        "results/experiment_7071_v619_bcit_drift_audit.json",
    ),
    (
        "exp7072-live-arc-compaction-ab",
        "Claim-grade live ARC compaction generalization A/B",
        "results/experiment_7072_v619_live_arc_compaction_ab.json",
    ),
    (
        "exp7073-entrance-energy-ising-parity",
        "QUBO translation and finite-distribution equivalence",
        "results/experiment_7073_v619_entrance_ising_parity.json",
    ),
    (
        "exp7074-degree16-placement-sampler-audit",
        "Degree-16 placement and finite-sampler audit",
        "results/experiment_7074_v619_degree16_sampler_audit.json",
    ),
    (
        "exp7075-v619-capstone",
        "V619 evidence matrix",
        "results/experiment_7075_v619_capstone.json",
    ),
)

MARKDOWN_CONTRACT = """# Independent V619 design fixture

**Milestone:** 2026.09.619

## Exact Task Contract

| Order | Task ID | Title | Deliverable |
|---:|---|---|---|
| 1 | `exp7063-v619-contract-preflight` | V619 Markdown and YAML task-contract preflight | `results/experiment_7063_v619_contract_preflight.json` |
| 2 | `exp7064-exact-entrance-constraint-fixture` | Exact source-grouped entrance constraint fixture | `results/experiment_7064_v619_exact_entrance_fixture.json` |
| 3 | `exp7065-three-family-entrance-proposal-bank` | Three-family SOTA entrance proposal bank | `results/experiment_7065_v619_three_family_entrance_bank.json` |
| 4 | `exp7066-entrance-bank-independent-audit` | Cold recomputation of entrance-bank support | `results/experiment_7066_v619_entrance_bank_audit.json` |
| 5 | `exp7067-hopfield-entrance-energy-selection` | Hopfield-style entrance energy selection comparison | `results/experiment_7067_v619_entrance_energy_selection.json` |
| 6 | `exp7068-hierarchical-branch-fidelity-control` | Lossless categorical mass-rebalancing evaluation | `results/experiment_7068_v619_hierarchical_branch_control.json` |
| 7 | `exp7069-context-bound-experience-contract` | BCIT use-validate-reject state machine | `results/experiment_7069_v619_context_authorization_contract.json` |
| 8 | `exp7070-bcit-prospective-self-learning` | Prospective context-bound continuous self-learning comparison | `results/experiment_7070_v619_bcit_self_learning.json` |
| 9 | `exp7071-bcit-drift-rollback-audit` | Fresh-process self-learning drift and rollback audit | `results/experiment_7071_v619_bcit_drift_audit.json` |
| 10 | `exp7072-live-arc-compaction-ab` | Claim-grade live ARC compaction generalization A/B | `results/experiment_7072_v619_live_arc_compaction_ab.json` |
| 11 | `exp7073-entrance-energy-ising-parity` | QUBO translation and finite-distribution equivalence | `results/experiment_7073_v619_entrance_ising_parity.json` |
| 12 | `exp7074-degree16-placement-sampler-audit` | Degree-16 placement and finite-sampler audit | `results/experiment_7074_v619_degree16_sampler_audit.json` |
| 13 | `exp7075-v619-capstone` | V619 evidence matrix | `results/experiment_7075_v619_capstone.json` |

### Exp7065 - Three-family SOTA entrance proposal bank
**Gate:**
`exp7064-exact-entrance-constraint-fixture.entrance_fixture_ready_score == 1`

### Exp7066 - Cold recomputation of entrance-bank support
**Gate:**
`exp7065-three-family-entrance-proposal-bank.entrance_proposal_bank_complete_score == 1`

### Exp7067 - Hopfield-style entrance energy selection comparison
**Gates:**
- `exp7066-entrance-bank-independent-audit.entrance_support_audit_ready_score == 1`
- `exp7066-entrance-bank-independent-audit.entrance_selector_headroom_ready_score == 1`

### Exp7068 - Lossless categorical mass-rebalancing evaluation
**Gate:**
`exp7066-entrance-bank-independent-audit.entrance_support_audit_ready_score == 1`

### Exp7070 - Prospective context-bound continuous self-learning comparison
**Gate:**
`exp7069-context-bound-experience-contract.context_authorization_contract_ready_score == 1`

### Exp7071 - Fresh-process self-learning drift and rollback audit
**Gate:**
`exp7070-bcit-prospective-self-learning.bcit_comparison_complete_score == 1`

### Exp7073 - QUBO translation and finite-distribution equivalence
**Gate:**
`exp7067-hopfield-entrance-energy-selection.entrance_energy_comparison_complete_score == 1`

### Exp7074 - Degree-16 placement and finite-sampler audit
**Gate:**
`exp7073-entrance-energy-ising-parity.ising_parity_ready_score == 1`
"""

GATES = {
    7065: [(7064, "entrance_fixture_ready_score")],
    7066: [(7065, "entrance_proposal_bank_complete_score")],
    7067: [
        (7066, "entrance_support_audit_ready_score"),
        (7066, "entrance_selector_headroom_ready_score"),
    ],
    7068: [(7066, "entrance_support_audit_ready_score")],
    7070: [(7069, "context_authorization_contract_ready_score")],
    7071: [(7070, "bcit_comparison_complete_score")],
    7073: [(7067, "entrance_energy_comparison_complete_score")],
    7074: [(7073, "ising_parity_ready_score")],
}
PRODUCED_FIELDS = {
    7064: ("entrance_fixture_ready_score",),
    7065: ("entrance_proposal_bank_complete_score",),
    7066: ("entrance_support_audit_ready_score", "entrance_selector_headroom_ready_score"),
    7067: ("entrance_energy_comparison_complete_score",),
    7069: ("context_authorization_contract_ready_score",),
    7070: ("bcit_comparison_complete_score",),
    7073: ("ising_parity_ready_score",),
}
LIVE_TASKS = {7065, 7072}


def _prompt(number: int) -> str:
    """Create one YAML-only prompt fixture without using Markdown task data."""

    substrate = "live_llm_inference" if number in LIVE_TASKS else mod.INFERENCE_SUBSTRATE
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
        if number == 7063:
            task["prior_failures"] = [
                {
                    "experiment_id": "exp7050-v618-active-contract-preflight",
                    "verdict": "complete_disqualified_v618_markdown_yaml_contract_mismatch",
                    "addressed_by": "V619 publishes and audits the same 13 rows in both files.",
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


def test_req_report_7063_spec_defines_thirteen_task_contract() -> None:
    """REQ-REPORT-7063 names the exact count, rows, and required scenarios."""

    spec = (ROOT / mod.SPEC_PATH).read_text(encoding="utf-8")
    section = spec[spec.index("### REQ-REPORT-7063") :]
    assert mod.EXPECTED_TASK_COUNT == len(YAML_TASK_ROWS) == 13
    assert mod.EXPECTED_TASK_IDS == tuple(row[0] for row in YAML_TASK_ROWS)
    for scenario in ("PARITY", "GATES", "DISCIPLINE", "PREFLIGHT", "ARTIFACT"):
        assert f"SCENARIO-REPORT-7063-{scenario}" in section
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert field in section


def test_scenario_report_7063_parity_accepts_independent_rows() -> None:
    """SCENARIO-REPORT-7063-PARITY accepts exactly 13 independent rows."""

    result = mod.evaluate_contract(MARKDOWN_CONTRACT, _roadmap(), set())
    assert result["passed"] is True
    assert len(result["task_contract_rows"]) == 13
    assert len(result["gate_producer_rows"]) == 9
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
    roadmap["tasks"][1]["prompt"] = roadmap["tasks"][1]["prompt"].replace(
        "entrance_fixture_ready_score", "different_ready_score"
    )


def _malformed_prior(roadmap: dict[str, Any]) -> None:
    roadmap["tasks"][0]["prior_failures"][0].pop("addressed_by")


def _missing_prompt_tail(roadmap: dict[str, Any]) -> None:
    roadmap["tasks"][0]["prompt"] = roadmap["tasks"][0]["prompt"].replace(mod.PROMPT_TAIL, "")


def _gated_capstone(roadmap: dict[str, Any]) -> None:
    roadmap["tasks"][-1]["gated_on"] = deepcopy(roadmap["tasks"][2]["gated_on"])


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
def test_req_report_7063_required_mutations_fail(
    case: str, mutation: Callable[[dict[str, Any]], None]
) -> None:
    """REQ-REPORT-7063 rejects every named contract mutation."""

    roadmap = _roadmap()
    mutation(roadmap)
    assert mod.evaluate_contract(MARKDOWN_CONTRACT, roadmap, set())["passed"] is False, case


def test_scenario_report_7063_gates_and_discipline_fail_closed() -> None:
    """SCENARIO-REPORT-7063-GATES rejects identity, model, and prompt defects."""

    mutations: tuple[Callable[[dict[str, Any]], None], ...] = (
        lambda value: value["tasks"][2]["gated_on"][0].__setitem__(
            "artifact_field", "nested.value"
        ),
        lambda value: value["tasks"][2]["gated_on"][0].__setitem__(
            "upstream", value["tasks"][-1]["id"]
        ),
        lambda value: value["tasks"][0].__setitem__("per_unit_rows", False),
        lambda value: value["tasks"][2].__setitem__(
            "prompt", value["tasks"][2]["prompt"].replace("cached_sota_pair()", "direct lookup")
        ),
        lambda value: value["tasks"][2].__setitem__(
            "prompt", value["tasks"][2]["prompt"] + "\nHeadline fallback: Qwen3.5-0.8B."
        ),
        lambda value: value["tasks"][2].__setitem__(
            "prompt",
            value["tasks"][2]["prompt"].replace("unsloth/Qwen3.6-35B-A3B-GGUF", "legacy/model"),
        ),
        lambda value: value["tasks"][2].__setitem__(
            "prompt", value["tasks"][2]["prompt"].replace("MODEL_SPECS", "models")
        ),
        lambda value: value["tasks"][2].__setitem__(
            "prompt",
            value["tasks"][2]["prompt"].replace(
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
    assert mod.evaluate_contract(MARKDOWN_CONTRACT, _roadmap(), {"exp7064"})["passed"] is False


def test_scenario_report_7063_prior_failure_requires_all_four_fields() -> None:
    """SCENARIO-REPORT-7063-DISCIPLINE rejects each incomplete prior record."""

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


def test_scenario_report_7063_preflight_builds_all_terminal_classes(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7063-PREFLIGHT separates blocked and disqualified states."""

    _write_preconditions(tmp_path)
    positive = mod.build_artifact(
        tmp_path, "20260906", output_path=tmp_path / "results/positive.json"
    )
    assert positive["verdict_class"] == "positive"
    assert positive["v619_task_contract_conforms_score"] == 1
    assert mod.validate_artifact(positive) == []

    changed = _roadmap()
    _changed_title(changed)
    _write_preconditions(tmp_path, changed)
    disqualified = mod.build_artifact(
        tmp_path, "20260906", output_path=tmp_path / "results/bad.json"
    )
    assert disqualified["verdict_class"] == "disqualified"
    assert disqualified["v619_task_contract_conforms_score"] == 0
    assert mod.validate_artifact(disqualified) == []

    (tmp_path / mod.DESIGN_PATH).unlink()
    blocked = mod.build_artifact(
        tmp_path, "20260906", output_path=tmp_path / "results/blocked.json"
    )
    assert blocked["verdict_class"] == "blocked"
    assert blocked["gate_check_summary"]["failed_check"] == "v619_markdown_readable"
    assert blocked["gate_check_summary"]["expected_value"] == "readable_nonempty_source"
    assert "FileNotFoundError" in blocked["gate_check_summary"]["observed_value"]
    assert mod.validate_artifact(blocked) == []


def test_scenario_report_7063_artifact_rejects_forgery(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7063-ARTIFACT derives the score, verdict, and checksum."""

    _write_preconditions(tmp_path)
    good = mod.build_artifact(tmp_path, "20260906", output_path=tmp_path / "good.json")
    changed = deepcopy(good)
    changed["task_contract_rows"][0]["passed"] = False
    changed["reproducibility_checksum"] = mod.reproducibility_checksum(changed)
    assert "v619_task_contract_conforms_score_invalid" in mod.validate_artifact(changed)
    changed = deepcopy(good)
    changed["reproducibility_checksum"] = "sha256:bad"
    assert "reproducibility_checksum_invalid" in mod.validate_artifact(changed)


def test_req_report_7063_parsers_and_ledgers_are_strict(tmp_path: Path) -> None:
    """REQ-REPORT-7063 rejects malformed sources and normalizes ledger IDs."""

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
        mod.parse_markdown_contract(MARKDOWN_CONTRACT.replace("`exp7063-v619", "`bad-v619", 1))
    with pytest.raises(ValueError, match="task table"):
        mod.parse_markdown_contract(
            f"**Milestone:** {mod.MILESTONE}\n## Exact Task Contract\nno rows\n"
        )
    with pytest.raises(ValueError, match="gate"):
        mod.parse_markdown_contract(MARKDOWN_CONTRACT.replace(" == 1`", " is ready`", 1))
    no_marker = MARKDOWN_CONTRACT + "\n### Exp7063 - no gate marker\nNo structured gate.\n"
    assert len(mod.parse_markdown_contract(no_marker)["tasks"]) == 13
    unknown_section = MARKDOWN_CONTRACT + (
        "\n### Exp9999 - outside table\n**Gate:**\n`exp7063-v619-contract-preflight.ready == 1`\n"
    )
    assert len(mod.parse_markdown_contract(unknown_section)["tasks"]) == 13
    with pytest.raises(ValueError, match="gate block"):
        mod.parse_markdown_contract(
            MARKDOWN_CONTRACT + "\n### Exp7063 - empty gate\n**Gate:**\nno value\n"
        )
    for value, message in (([], "mapping"), ({}, "tasks list"), ({"tasks": [1]}, "mapping")):
        with pytest.raises(ValueError, match=message):
            mod.parse_yaml_contract(value)
    path = tmp_path / "bad.yaml"
    path.write_text("[]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="mapping required"):
        mod.load_yaml(path)
    assert mod.retired_experiment_ids({"retired": [{"experiment_id": 7063}]}) == {"exp7063"}


def test_req_report_7063_precondition_and_parse_error_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-REPORT-7063-PREFLIGHT preserves unavailable input reasons."""

    rows, roadmap, exclusion = mod._preconditions(tmp_path, tmp_path / "results/result.json")
    assert roadmap is exclusion is None
    assert {row["check"] for row in rows if not row["available"]} >= {
        "v619_markdown_readable",
        "v619_active_yaml_readable",
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
        next(row for row in rows if row["check"] == "v619_active_yaml_readable")["available"]
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


def test_req_report_7063_failure_summaries_name_count_source() -> None:
    """REQ-REPORT-7063 reports Markdown and active-YAML count drift separately."""

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


def test_scenario_report_7063_artifact_validator_covers_each_headline(
    tmp_path: Path,
) -> None:
    """SCENARIO-REPORT-7063-ARTIFACT rejects every forged headline field."""

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


def test_req_report_7063_score_rejects_each_evidence_shape(tmp_path: Path) -> None:
    """REQ-REPORT-7063 recomputes conformance from every stored evidence group."""

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


def test_req_report_7063_cli_validation_and_generation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-REPORT-7063 exposes deterministic validate and generate commands."""

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
