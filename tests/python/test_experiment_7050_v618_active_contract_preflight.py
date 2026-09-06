"""Tests for the independent V618 active-roadmap contract preflight."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any, Callable

import pytest
import yaml

from carnot import experiment_7050_v618_active_contract_preflight as mod


ROOT = Path(__file__).resolve().parents[2]

YAML_TASK_ROWS = (
    (
        "exp7050-v618-active-contract-preflight",
        "V618 active-roadmap and design-document contract preflight",
        "results/experiment_7050_v618_active_contract_preflight.json",
    ),
    (
        "exp7051-model-report-evidence-requalification",
        "Official-live model report evidence requalification",
        "results/experiment_7051_v618_model_report_requalification.json",
    ),
    (
        "exp7052-typed-identity-bridge-attack-audit",
        "Typed model identity bridge and fresh-process attack audit",
        "results/experiment_7052_v618_typed_identity_attack_audit.json",
    ),
    (
        "exp7053-exact-entrance-family-fixture",
        "Exact entrance-family constraint fixture",
        "results/experiment_7053_v618_exact_entrance_fixture.json",
    ),
    (
        "exp7054-three-family-entrance-proposal-bank",
        "Three-family SOTA entrance proposal and continuation bank",
        "results/experiment_7054_v618_three_family_entrance_bank.json",
    ),
    (
        "exp7055-independent-entrance-support-audit",
        "Independent entrance support and leakage audit",
        "results/experiment_7055_v618_entrance_bank_cold_audit.json",
    ),
    (
        "exp7056-causal-entrance-energy-selection",
        "Causal entrance-energy selection comparison",
        "results/experiment_7056_v618_entrance_energy_selection.json",
    ),
    (
        "exp7057-lossless-hierarchical-branch-control",
        "Lossless hierarchical branch verification control",
        "results/experiment_7057_v618_hierarchical_branch_control.json",
    ),
    (
        "exp7058-context-bound-experience-authorization",
        "Context-bound experience authorization contract",
        "results/experiment_7058_v618_bcit_authorization_contract.json",
    ),
    (
        "exp7059-bcit-prospective-self-learning",
        "Prospective BCIT continuous self-learning comparison",
        "results/experiment_7059_v618_bcit_continuous_self_learning.json",
    ),
    (
        "exp7060-bcit-drift-cold-audit",
        "Fresh-process BCIT drift, poison, and rollback audit",
        "results/experiment_7060_v618_bcit_drift_cold_audit.json",
    ),
    (
        "exp7061-entrance-energy-ising-parity",
        "Entrance energy to sparse Ising parity receipt",
        "results/experiment_7061_v618_entrance_ising_parity.json",
    ),
    (
        "exp7062-v618-capstone",
        "V618 independent capstone and V619 handoff",
        "results/experiment_7062_v618_capstone.json",
    ),
)

MARKDOWN_CONTRACT = """# Independent V618 design fixture

**Milestone:** 2026.09.618

## Exact Task Contract

| Order | Task ID | Title | Deliverable |
|---:|---|---|---|
| 1 | `exp7050-v618-active-contract-preflight` | V618 active-roadmap and design-document contract preflight | `results/experiment_7050_v618_active_contract_preflight.json` |
| 2 | `exp7051-model-report-evidence-requalification` | Official-live model report evidence requalification | `results/experiment_7051_v618_model_report_requalification.json` |
| 3 | `exp7052-typed-identity-bridge-attack-audit` | Typed model identity bridge and fresh-process attack audit | `results/experiment_7052_v618_typed_identity_attack_audit.json` |
| 4 | `exp7053-exact-entrance-family-fixture` | Exact entrance-family constraint fixture | `results/experiment_7053_v618_exact_entrance_fixture.json` |
| 5 | `exp7054-three-family-entrance-proposal-bank` | Three-family SOTA entrance proposal and continuation bank | `results/experiment_7054_v618_three_family_entrance_bank.json` |
| 6 | `exp7055-independent-entrance-support-audit` | Independent entrance support and leakage audit | `results/experiment_7055_v618_entrance_bank_cold_audit.json` |
| 7 | `exp7056-causal-entrance-energy-selection` | Causal entrance-energy selection comparison | `results/experiment_7056_v618_entrance_energy_selection.json` |
| 8 | `exp7057-lossless-hierarchical-branch-control` | Lossless hierarchical branch verification control | `results/experiment_7057_v618_hierarchical_branch_control.json` |
| 9 | `exp7058-context-bound-experience-authorization` | Context-bound experience authorization contract | `results/experiment_7058_v618_bcit_authorization_contract.json` |
| 10 | `exp7059-bcit-prospective-self-learning` | Prospective BCIT continuous self-learning comparison | `results/experiment_7059_v618_bcit_continuous_self_learning.json` |
| 11 | `exp7060-bcit-drift-cold-audit` | Fresh-process BCIT drift, poison, and rollback audit | `results/experiment_7060_v618_bcit_drift_cold_audit.json` |
| 12 | `exp7061-entrance-energy-ising-parity` | Entrance energy to sparse Ising parity receipt | `results/experiment_7061_v618_entrance_ising_parity.json` |
| 13 | `exp7062-v618-capstone` | V618 independent capstone and V619 handoff | `results/experiment_7062_v618_capstone.json` |

### Exp7052 - Typed model identity bridge and fresh-process attack audit
**Gate:**
`exp7051-model-report-evidence-requalification.model_report_evidence_ready_score == 1`

### Exp7054 - Three-family SOTA entrance proposal and continuation bank
**Gate:**
`exp7053-exact-entrance-family-fixture.entrance_fixture_ready_score == 1`

### Exp7055 - Independent entrance support and leakage audit
**Gate:**
`exp7054-three-family-entrance-proposal-bank.entrance_bank_complete_score == 1`

### Exp7056 - Causal entrance-energy selection comparison
**Gates:**
- `exp7055-independent-entrance-support-audit.entrance_audit_ready_score == 1`
- `exp7055-independent-entrance-support-audit.entrance_headroom_ready_score == 1`

### Exp7057 - Lossless hierarchical branch verification control
**Gate:**
`exp7055-independent-entrance-support-audit.entrance_audit_ready_score == 1`

### Exp7059 - Prospective BCIT continuous self-learning comparison
**Gate:**
`exp7058-context-bound-experience-authorization.bcit_contract_ready_score == 1`

### Exp7060 - Fresh-process BCIT drift, poison, and rollback audit
**Gate:**
`exp7059-bcit-prospective-self-learning.bcit_stream_complete_score == 1`

### Exp7061 - Entrance energy to sparse Ising parity receipt
**Gate:**
`exp7056-causal-entrance-energy-selection.entrance_energy_comparison_complete_score == 1`
"""

GATES = {
    7052: [(7051, "model_report_evidence_ready_score")],
    7054: [(7053, "entrance_fixture_ready_score")],
    7055: [(7054, "entrance_bank_complete_score")],
    7056: [(7055, "entrance_audit_ready_score"), (7055, "entrance_headroom_ready_score")],
    7057: [(7055, "entrance_audit_ready_score")],
    7059: [(7058, "bcit_contract_ready_score")],
    7060: [(7059, "bcit_stream_complete_score")],
    7061: [(7056, "entrance_energy_comparison_complete_score")],
}
PRODUCED_FIELDS = {
    7051: ("model_report_evidence_ready_score",),
    7053: ("entrance_fixture_ready_score",),
    7054: ("entrance_bank_complete_score",),
    7055: ("entrance_audit_ready_score", "entrance_headroom_ready_score"),
    7056: ("entrance_energy_comparison_complete_score",),
    7058: ("bcit_contract_ready_score",),
    7059: ("bcit_stream_complete_score",),
}
LIVE_TASKS = {7051, 7054}


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
        lines.extend(
            [
                "MODEL_SPECS resolves through cached_sota_pair().",
                "Use unsloth/Qwen3.6-35B-A3B-GGUF for the headline.",
            ]
        )
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
        if number == 7050:
            task["prior_failures"] = [
                {
                    "experiment_id": "exp7038-v617-active-contract-preflight",
                    "verdict": "complete_blocked_v617_active_contract_preflight_prerequisite_missing",
                    "addressed_by": "V618 creates the missing design and audits 13 rows.",
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
        mod.COMPLETE_PATH: "milestones: []\n",
        mod.SPEC_PATH: "### REQ-REPORT-7050\n",
    }
    for relative, content in values.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")


def test_req_report_7050_spec_defines_thirteen_task_contract() -> None:
    """REQ-REPORT-7050 names the exact count, rows, and required scenarios."""

    spec = (ROOT / mod.SPEC_PATH).read_text(encoding="utf-8")
    section = spec[spec.index("### REQ-REPORT-7050") :]
    assert mod.EXPECTED_TASK_COUNT == len(YAML_TASK_ROWS) == 13
    assert mod.EXPECTED_TASK_IDS == tuple(row[0] for row in YAML_TASK_ROWS)
    for scenario in ("PARITY", "GATES", "DISCIPLINE", "PREFLIGHT", "ARTIFACT"):
        assert f"SCENARIO-REPORT-7050-{scenario}" in section
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert field in section


def test_scenario_report_7050_parity_accepts_independent_rows() -> None:
    """SCENARIO-REPORT-7050-PARITY accepts exactly 13 independent rows."""

    result = mod.evaluate_contract(MARKDOWN_CONTRACT, _roadmap(), set(), set())
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
        "model_report_evidence_ready_score", "different_ready_score"
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
def test_req_report_7050_required_mutations_fail(
    case: str, mutation: Callable[[dict[str, Any]], None]
) -> None:
    """REQ-REPORT-7050 rejects every named contract mutation."""

    roadmap = _roadmap()
    mutation(roadmap)
    assert mod.evaluate_contract(MARKDOWN_CONTRACT, roadmap, set(), set())["passed"] is False, case


def test_scenario_report_7050_gates_and_discipline_fail_closed() -> None:
    """SCENARIO-REPORT-7050-GATES rejects identity, model, and prompt defects."""

    mutations: tuple[Callable[[dict[str, Any]], None], ...] = (
        lambda value: value["tasks"][2]["gated_on"][0].__setitem__(
            "artifact_field", "nested.value"
        ),
        lambda value: value["tasks"][2]["gated_on"][0].__setitem__(
            "upstream", value["tasks"][-1]["id"]
        ),
        lambda value: value["tasks"][0].__setitem__("per_unit_rows", False),
        lambda value: value["tasks"][1].__setitem__(
            "prompt", value["tasks"][1]["prompt"].replace("cached_sota_pair()", "direct lookup")
        ),
        lambda value: value["tasks"][1].__setitem__(
            "prompt", value["tasks"][1]["prompt"] + "\nHeadline fallback: Qwen3.5-0.8B."
        ),
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
        assert mod.evaluate_contract(MARKDOWN_CONTRACT, roadmap, set(), set())["passed"] is False
    assert (
        mod.evaluate_contract(MARKDOWN_CONTRACT, _roadmap(), {"exp7051"}, set())["passed"] is False
    )
    assert (
        mod.evaluate_contract(MARKDOWN_CONTRACT, _roadmap(), set(), {"exp7052"})["passed"] is False
    )


def test_scenario_report_7050_preflight_builds_all_terminal_classes(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7050-PREFLIGHT separates blocked and disqualified states."""

    _write_preconditions(tmp_path)
    positive = mod.build_artifact(
        tmp_path, "20260906", output_path=tmp_path / "results/positive.json"
    )
    assert positive["verdict_class"] == "positive"
    assert positive["v618_task_contract_conforms_score"] == 1
    assert mod.validate_artifact(positive) == []

    changed = _roadmap()
    _changed_title(changed)
    _write_preconditions(tmp_path, changed)
    disqualified = mod.build_artifact(
        tmp_path, "20260906", output_path=tmp_path / "results/bad.json"
    )
    assert disqualified["verdict_class"] == "disqualified"
    assert disqualified["v618_task_contract_conforms_score"] == 0
    assert mod.validate_artifact(disqualified) == []

    (tmp_path / mod.DESIGN_PATH).unlink()
    blocked = mod.build_artifact(
        tmp_path, "20260906", output_path=tmp_path / "results/blocked.json"
    )
    assert blocked["verdict_class"] == "blocked"
    assert blocked["gate_check_summary"]["failed_check"] == "v618_markdown_readable"
    assert blocked["gate_check_summary"]["expected_value"] == "readable_nonempty_source"
    assert "FileNotFoundError" in blocked["gate_check_summary"]["observed_value"]
    assert mod.validate_artifact(blocked) == []


def test_scenario_report_7050_artifact_rejects_forgery(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7050-ARTIFACT derives the score, verdict, and checksum."""

    _write_preconditions(tmp_path)
    good = mod.build_artifact(tmp_path, "20260906", output_path=tmp_path / "good.json")
    changed = deepcopy(good)
    changed["task_contract_rows"][0]["passed"] = False
    changed["reproducibility_checksum"] = mod.reproducibility_checksum(changed)
    assert "v618_task_contract_conforms_score_invalid" in mod.validate_artifact(changed)
    changed = deepcopy(good)
    changed["reproducibility_checksum"] = "sha256:bad"
    assert "reproducibility_checksum_invalid" in mod.validate_artifact(changed)


def test_req_report_7050_parsers_and_ledgers_are_strict(tmp_path: Path) -> None:
    """REQ-REPORT-7050 rejects malformed sources and normalizes ledger IDs."""

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
        mod.parse_markdown_contract(MARKDOWN_CONTRACT.replace("`exp7050-v618", "`bad-v618", 1))
    with pytest.raises(ValueError, match="task table"):
        mod.parse_markdown_contract(
            f"**Milestone:** {mod.MILESTONE}\n## Exact Task Contract\nno rows\n"
        )
    with pytest.raises(ValueError, match="gate"):
        mod.parse_markdown_contract(MARKDOWN_CONTRACT.replace(" == 1`", " is ready`", 1))
    no_marker = MARKDOWN_CONTRACT + "\n### Exp7050 - no gate marker\nNo structured gate.\n"
    assert len(mod.parse_markdown_contract(no_marker)["tasks"]) == 13
    unknown_section = MARKDOWN_CONTRACT + (
        "\n### Exp9999 - outside table\n**Gate:**\n"
        "`exp7050-v618-active-contract-preflight.ready == 1`\n"
    )
    assert len(mod.parse_markdown_contract(unknown_section)["tasks"]) == 13
    with pytest.raises(ValueError, match="gate block"):
        mod.parse_markdown_contract(
            MARKDOWN_CONTRACT + "\n### Exp7050 - empty gate\n**Gate:**\nno value\n"
        )
    for value, message in (([], "mapping"), ({}, "tasks list"), ({"tasks": [1]}, "mapping")):
        with pytest.raises(ValueError, match=message):
            mod.parse_yaml_contract(value)
    path = tmp_path / "bad.yaml"
    path.write_text("[]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="mapping required"):
        mod.load_yaml(path)
    assert mod.retired_experiment_ids({"retired": [{"experiment_id": 7050}]}) == {"exp7050"}
    assert mod.completed_experiment_ids({"milestones": [{"tasks": [{"id": "exp7051-done"}]}]}) == {
        "exp7051"
    }


def test_req_report_7050_precondition_and_parse_error_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-REPORT-7050-PREFLIGHT preserves unavailable input reasons."""

    rows, roadmap, exclusion, complete = mod._preconditions(
        tmp_path, tmp_path / "results/result.json"
    )
    assert roadmap is exclusion is complete is None
    assert {row["check"] for row in rows if not row["available"]} >= {
        "v618_markdown_readable",
        "v618_active_yaml_readable",
        "exclusion_manifest_readable",
        "completed_experiment_ledger_readable",
        "reporting_spec_readable",
    }

    _write_preconditions(tmp_path)
    (tmp_path / mod.STAGING_ROADMAP_PATH).write_text("not parsed\n", encoding="utf-8")
    rows, roadmap, exclusion, complete = mod._preconditions(
        tmp_path, tmp_path / "results/result.json"
    )
    assert roadmap and exclusion and complete
    staging = next(row for row in rows if row["check"] == "staging_file_not_required")
    assert staging["observed_value"] == "present_but_not_read"

    (tmp_path / mod.ACTIVE_ROADMAP_PATH).write_text("[]\n", encoding="utf-8")
    rows, roadmap, _exclusion, _complete = mod._preconditions(
        tmp_path, tmp_path / "results/result.json"
    )
    assert roadmap is None
    assert (
        next(row for row in rows if row["check"] == "v618_active_yaml_readable")["available"]
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


def test_req_report_7050_failure_summaries_name_count_source() -> None:
    """REQ-REPORT-7050 reports Markdown and active-YAML count drift separately."""

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


def test_scenario_report_7050_artifact_validator_covers_each_headline(
    tmp_path: Path,
) -> None:
    """SCENARIO-REPORT-7050-ARTIFACT rejects every forged headline field."""

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


def test_req_report_7050_score_rejects_each_evidence_shape(tmp_path: Path) -> None:
    """REQ-REPORT-7050 recomputes conformance from every stored evidence group."""

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


def test_req_report_7050_cli_validation_and_generation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-REPORT-7050 exposes deterministic validate and generate commands."""

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
