"""Tests for REQ-REPORT-7091, the independent V622 contract preflight."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any, Callable

import pytest
import yaml

from carnot import experiment_7091_v622_contract_preflight as mod


ROOT = Path(__file__).resolve().parents[2]

YAML_TASK_ROWS = (
    (
        "exp7091-v622-contract-preflight",
        "V622 Markdown and YAML task-contract preflight",
        "results/experiment_7091_v622_contract_preflight.json",
    ),
    (
        "exp7092-v622-execution-sota-ingestion",
        "V622 execution-time SOTA ingestion and claim-boundary audit",
        "results/experiment_7092_v622_sota_ingestion.json",
    ),
    (
        "exp7093-recovered-entrance-bank-sufficiency-audit",
        "Recovered cold entrance-bank sufficiency audit",
        "results/experiment_7093_v622_entrance_bank_sufficiency_audit.json",
    ),
    (
        "exp7094-matched-hardness-entrance-diagnostic",
        "Matched-hardness entrance difficulty diagnostic",
        "results/experiment_7094_v622_matched_hardness_diagnostic.json",
    ),
    (
        "exp7095-entrance-energy-matched-controls",
        "Entrance energy versus matched strong controls",
        "results/experiment_7095_v622_entrance_energy_controls.json",
    ),
    (
        "exp7096-cold-entrance-energy-abstention-audit",
        "Cold entrance-energy abstention and shift audit",
        "results/experiment_7096_v622_entrance_energy_abstention_audit.json",
    ),
    (
        "exp7097-z1t-degree16-ising-sampler-receipt",
        "Z1T-style degree-16 Ising and sampler receipt",
        "results/experiment_7097_v622_z1t_degree16_ising_receipt.json",
    ),
    (
        "exp7098-controlled-causal-entrance-stream",
        "Controlled reusable-versus-decoy entrance stream",
        "results/experiment_7098_v622_controlled_causal_stream.json",
    ),
    (
        "exp7099-verifier-signed-continual-memory-ab",
        "Verifier-signed continual constraint-memory A/B",
        "results/experiment_7099_v622_continual_memory_ab.json",
    ),
    (
        "exp7100-continual-memory-cold-audit",
        "Cold continual-memory safety and rollback audit",
        "results/experiment_7100_v622_continual_memory_cold_audit.json",
    ),
    (
        "exp7101-cross-family-entrance-energy-audit",
        "Cross-family entrance-energy shortcut audit",
        "results/experiment_7101_v622_cross_family_energy_audit.json",
    ),
    (
        "exp7102-v622-capstone",
        "V622 independent evidence matrix and branch disposition",
        "results/experiment_7102_v622_capstone.json",
    ),
)

MARKDOWN_CONTRACT = """# Independent V622 design fixture

**Milestone:** `2026.09.622`

## Exact Task Contract

| Order | Task ID | Title | Deliverable | Structured gates |
|---:|---|---|---|---|
| 1 | `exp7091-v622-contract-preflight` | V622 Markdown and YAML task-contract preflight | `results/experiment_7091_v622_contract_preflight.json` | None |
| 2 | `exp7092-v622-execution-sota-ingestion` | V622 execution-time SOTA ingestion and claim-boundary audit | `results/experiment_7092_v622_sota_ingestion.json` | None |
| 3 | `exp7093-recovered-entrance-bank-sufficiency-audit` | Recovered cold entrance-bank sufficiency audit | `results/experiment_7093_v622_entrance_bank_sufficiency_audit.json` | None |
| 4 | `exp7094-matched-hardness-entrance-diagnostic` | Matched-hardness entrance difficulty diagnostic | `results/experiment_7094_v622_matched_hardness_diagnostic.json` | `exp7093-recovered-entrance-bank-sufficiency-audit.entrance_support_audit_ready_score == 1` |
| 5 | `exp7095-entrance-energy-matched-controls` | Entrance energy versus matched strong controls | `results/experiment_7095_v622_entrance_energy_controls.json` | `exp7093-recovered-entrance-bank-sufficiency-audit.entrance_support_audit_ready_score == 1 AND exp7093-recovered-entrance-bank-sufficiency-audit.entrance_selector_headroom_ready_score == 1 AND exp7094-matched-hardness-entrance-diagnostic.matched_difficulty_diagnostic_complete_score == 1` |
| 6 | `exp7096-cold-entrance-energy-abstention-audit` | Cold entrance-energy abstention and shift audit | `results/experiment_7096_v622_entrance_energy_abstention_audit.json` | `exp7095-entrance-energy-matched-controls.entrance_energy_comparison_complete_score == 1` |
| 7 | `exp7097-z1t-degree16-ising-sampler-receipt` | Z1T-style degree-16 Ising and sampler receipt | `results/experiment_7097_v622_z1t_degree16_ising_receipt.json` | `exp7095-entrance-energy-matched-controls.entrance_energy_comparison_complete_score == 1` |
| 8 | `exp7098-controlled-causal-entrance-stream` | Controlled reusable-versus-decoy entrance stream | `results/experiment_7098_v622_controlled_causal_stream.json` | None |
| 9 | `exp7099-verifier-signed-continual-memory-ab` | Verifier-signed continual constraint-memory A/B | `results/experiment_7099_v622_continual_memory_ab.json` | `exp7098-controlled-causal-entrance-stream.controlled_causal_stream_ready_score == 1` |
| 10 | `exp7100-continual-memory-cold-audit` | Cold continual-memory safety and rollback audit | `results/experiment_7100_v622_continual_memory_cold_audit.json` | `exp7099-verifier-signed-continual-memory-ab.continual_memory_comparison_complete_score == 1` |
| 11 | `exp7101-cross-family-entrance-energy-audit` | Cross-family entrance-energy shortcut audit | `results/experiment_7101_v622_cross_family_energy_audit.json` | `exp7095-entrance-energy-matched-controls.entrance_energy_comparison_complete_score == 1` |
| 12 | `exp7102-v622-capstone` | V622 independent evidence matrix and branch disposition | `results/experiment_7102_v622_capstone.json` | None |

### Exp7091 - preflight
**Prior failures:** Exp7076 and Exp7084 both found mismatches.

### Exp7092 - ingestion
**Prior failures:** None.

### Exp7093 - recovered audit
**Prior failures:** Exp7081 and Exp7087 did not produce usable current evidence.

### Exp7094 - hardness
**Prior failures:** None.

### Exp7095 - energy
**Prior failures:** Exp1006, Exp7067, Exp7082, and Exp7088 lost their chains.

### Exp7096 - abstention
**Prior failures:** Exp533 and Exp7089 did not establish this scope.

### Exp7097 - Ising
**Prior failures:** Exp7073, Exp7074, Exp7083, and Exp7090 were blocked.

### Exp7098 - stream
**Prior failures:** Exp7070 had insufficient evidence.

### Exp7099 - memory
**Prior failures:** Exp6978 and Exp7070 did not show value.

### Exp7100 - cold audit
**Prior failures:** Exp6979 and Exp7071 did not complete this chain.

### Exp7101 - family audit
**Prior failures:** Exp6808 was blocked.

### Exp7102 - capstone
**Prior failures:** None.
"""

GATES = {
    7094: [(7093, "entrance_support_audit_ready_score")],
    7095: [
        (7093, "entrance_support_audit_ready_score"),
        (7093, "entrance_selector_headroom_ready_score"),
        (7094, "matched_difficulty_diagnostic_complete_score"),
    ],
    7096: [(7095, "entrance_energy_comparison_complete_score")],
    7097: [(7095, "entrance_energy_comparison_complete_score")],
    7099: [(7098, "controlled_causal_stream_ready_score")],
    7100: [(7099, "continual_memory_comparison_complete_score")],
    7101: [(7095, "entrance_energy_comparison_complete_score")],
}
PRODUCED_FIELDS = {
    7093: ("entrance_support_audit_ready_score", "entrance_selector_headroom_ready_score"),
    7094: ("matched_difficulty_diagnostic_complete_score",),
    7095: ("entrance_energy_comparison_complete_score",),
    7098: ("controlled_causal_stream_ready_score",),
    7099: ("continual_memory_comparison_complete_score",),
}
PRIOR_IDS = {
    7091: ("exp7076-v620-contract-preflight", "exp7084-v621-contract-preflight"),
    7093: (
        "exp7081-entrance-bank-set-sufficiency-audit",
        "exp7087-cold-entrance-bank-sufficiency-audit",
    ),
    7095: (
        "exp1006-energy-selection-ssd",
        "exp7067-hopfield-entrance-energy-selection",
        "exp7082-entrance-energy-likelihood-controls",
        "exp7088-entrance-energy-strong-controls",
    ),
    7096: ("exp533-cold-decoding-energy-guidance", "exp7089-cold-entrance-energy-abstention-audit"),
    7097: (
        "exp7073-entrance-energy-ising-parity",
        "exp7074-degree16-placement-sampler-audit",
        "exp7083-entrance-ising-degree16-parity",
        "exp7090-entrance-ising-degree16-sampling-receipt",
    ),
    7098: ("exp7070-bcit-prospective-self-learning",),
    7099: (
        "exp6978-transactional-constraint-self-learning",
        "exp7070-bcit-prospective-self-learning",
    ),
    7100: ("exp6979-self-learning-cold-audit", "exp7071-bcit-drift-rollback-audit"),
    7101: ("exp6808-route-program-portability",),
}


def _prompt(number: int) -> str:
    """Build a valid YAML-only prompt for one synthetic task."""

    substrate_class = "aggregation" if number in {7091, 7102} else "no_model_load"
    fields = [
        "field_principles",
        "preconditions_checked",
        f"inference_substrate (fixture substrate {number})",
        f"inference_substrate_class ({substrate_class}, or blocked_no_run on a blocked precondition)",
        "execution_venue (host)",
        "duration_s",
        "source_artifact_hashes",
        "rows",
        *PRODUCED_FIELDS.get(number, ()),
        "random_seed",
        "reproducibility_checksum",
        "gate_check_summary with failed check, expected value, and observed value for any blocked verdict",
        "verifier_is_oracle (false)",
        "verdict_class (positive | circular_positive | null | blocked | disqualified | partial)",
        "honest_verdict with a terminal prefix consistent with verdict_class",
    ]
    lines = ["Compare every per-unit row under matched controls."]
    if number in mod.EXP7086_CONSUMERS:
        fields.append("source_model_specs")
        lines.append("Consume Exp7086 and preserve immutable source provenance.")
        lines.extend(f"Preserve {model}." for model in mod.MANDATED_SOTA_GGUFS)
        lines.append("Never regenerate or substitute a legacy-small model.")
    lines.extend(
        [
            "REQUIRED ARTIFACT FIELDS: " + "; ".join(fields) + ".",
            "Run command: fixture",
            mod.PROMPT_TAIL,
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
            "agent_type": "claude" if number in {7091, 7093} else "codex",
            "model": "opus" if number in {7091, 7093} else "gpt-5.6-sol",
            "requires_gpu": False,
            "max_turns": 50,
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
        if number in PRIOR_IDS:
            task["prior_failures"] = [
                {
                    "experiment_id": prior,
                    "verdict": "blocked_prior",
                    "addressed_by": "V622 changes the failed prerequisite and tests the replacement.",
                    "retire_if_same_verdict": True,
                }
                for prior in PRIOR_IDS[number]
            ]
        tasks.append(task)
    return {
        "milestone": mod.MILESTONE,
        "milestone_title": "fixture",
        "milestone_doc": str(mod.DESIGN_PATH),
        "tasks": tasks,
    }


def _write_preconditions(root: Path, roadmap: dict[str, Any] | None = None) -> None:
    """Create only the runtime files that the audit may read."""

    values = {
        mod.DESIGN_PATH: MARKDOWN_CONTRACT,
        mod.ACTIVE_ROADMAP_PATH: yaml.safe_dump(roadmap or _roadmap()),
        mod.EXCLUSION_PATH: "retired: []\nretired_extras: []\n",
    }
    for relative, content in values.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")


def test_req_report_7091_spec_defines_exact_contract() -> None:
    """REQ-REPORT-7091 names all rows, evidence fields, and scenarios."""

    spec = (ROOT / mod.SPEC_PATH).read_text(encoding="utf-8")
    section = spec[spec.index("### REQ-REPORT-7091") :]
    assert mod.EXPECTED_TASK_COUNT == len(YAML_TASK_ROWS) == 12
    assert mod.EXPECTED_TASK_IDS == tuple(row[0] for row in YAML_TASK_ROWS)
    for scenario in ("PARITY", "GATES", "DISCIPLINE", "PREFLIGHT", "ARTIFACT"):
        assert f"SCENARIO-REPORT-7091-{scenario}" in section
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert field in section


def test_scenario_report_7091_parity_accepts_twelve_independent_rows() -> None:
    """SCENARIO-REPORT-7091-PARITY accepts exact independent sources."""

    result = mod.evaluate_contract(MARKDOWN_CONTRACT, _roadmap(), set())
    assert result["passed"] is True
    assert len(result["task_contract_rows"]) == 12
    assert len(result["gate_producer_rows"]) == 9
    assert result["markdown_task_rows"] is not result["yaml_task_rows"]
    assert all(row["passed"] for row in result["task_contract_rows"])


@pytest.mark.parametrize("case", ("missing", "extra", "reordered", "title", "deliverable"))
def test_req_report_7091_row_mutations_fail(case: str) -> None:
    """REQ-REPORT-7091 rejects missing, extra, reordered, or changed YAML rows."""

    roadmap = _roadmap()
    if case == "missing":
        roadmap["tasks"].pop(5)
    elif case == "extra":
        roadmap["tasks"].append(deepcopy(roadmap["tasks"][-1]))
    elif case == "reordered":
        roadmap["tasks"][5], roadmap["tasks"][6] = roadmap["tasks"][6], roadmap["tasks"][5]
    elif case == "title":
        roadmap["tasks"][0]["title"] += " changed"
    else:
        roadmap["tasks"][0]["deliverable"] = "results/wrong.json"
    assert mod.evaluate_contract(MARKDOWN_CONTRACT, roadmap, set())["passed"] is False


def test_scenario_report_7091_markdown_and_gate_syntax_are_strict() -> None:
    """SCENARIO-REPORT-7091-GATES rejects malformed Markdown and YAML gates."""

    with pytest.raises(ValueError, match="task row"):
        mod.parse_markdown_contract(MARKDOWN_CONTRACT.replace("| 1 |", "| 1 | extra |", 1))
    with pytest.raises(ValueError, match="structured gate"):
        mod.parse_markdown_contract(MARKDOWN_CONTRACT.replace(" == 1`", " is ready`", 1))
    with pytest.raises(ValueError, match="section is missing"):
        mod.parse_markdown_contract(MARKDOWN_CONTRACT.replace("## Exact Task Contract", "## Other"))
    with pytest.raises(ValueError, match="task row"):
        mod.parse_markdown_contract(MARKDOWN_CONTRACT.replace("exp7091-v622", "not-an-exp", 1))
    with pytest.raises(ValueError, match="table is missing"):
        mod.parse_markdown_contract("**Milestone:** `2026.09.622`\n\n## Exact Task Contract\n")
    without_one_prior_marker = MARKDOWN_CONTRACT.replace(
        "**Prior failures:** None.", "**History:** None.", 1
    )
    assert len(mod.parse_markdown_contract(without_one_prior_marker)["tasks"]) == 12
    roadmap = _roadmap()
    roadmap["tasks"][3]["gated_on"][0]["op"] = None
    assert mod.evaluate_contract(MARKDOWN_CONTRACT, roadmap, set())["passed"] is False


def test_scenario_report_7091_gates_require_earlier_producer_bare_owned_fields() -> None:
    """SCENARIO-REPORT-7091-GATES rejects bad producers and a gated capstone."""

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
    )
    for mutate in mutations:
        roadmap = _roadmap()
        mutate(roadmap)
        assert mod.evaluate_contract(MARKDOWN_CONTRACT, roadmap, set())["passed"] is False


def test_scenario_report_7091_prior_failures_require_all_four_fields() -> None:
    """SCENARIO-REPORT-7091-DISCIPLINE rejects incomplete or missing priors."""

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


def test_scenario_report_7091_execution_rules_fail_closed() -> None:
    """SCENARIO-REPORT-7091-DISCIPLINE checks routing, rows, classes, venue, and tails."""

    mutations: tuple[Callable[[dict[str, Any]], None], ...] = (
        lambda value: value["tasks"][4].__setitem__("per_unit_rows", False),
        lambda value: value["tasks"][0].__setitem__("model", "wrong"),
        lambda value: value["tasks"][0].__setitem__("agent_type", "gemini"),
        lambda value: value["tasks"][0].__setitem__("requires_gpu", True),
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
            "prompt", value["tasks"][0]["prompt"].replace(mod.PROMPT_TAIL, "")
        ),
    )
    for mutate in mutations:
        roadmap = _roadmap()
        mutate(roadmap)
        assert mod.evaluate_contract(MARKDOWN_CONTRACT, roadmap, set())["passed"] is False
    assert mod.evaluate_contract(MARKDOWN_CONTRACT, _roadmap(), {"exp7095"})["passed"] is False


def test_scenario_report_7091_model_policy_forbids_load_and_provenance_loss() -> None:
    """SCENARIO-REPORT-7091-DISCIPLINE preserves Exp7086 models without regeneration."""

    mutations: tuple[Callable[[dict[str, Any]], None], ...] = (
        lambda value: value["tasks"][2].__setitem__("requires_gpu", True),
        lambda value: value["tasks"][2].__setitem__(
            "prompt",
            value["tasks"][2]["prompt"].replace("no_model_load, or", "model_full_generation, or"),
        ),
        lambda value: value["tasks"][2].__setitem__(
            "prompt", value["tasks"][2]["prompt"].replace("source_model_specs", "models")
        ),
        lambda value: value["tasks"][2].__setitem__(
            "prompt",
            value["tasks"][2]["prompt"].replace(mod.MANDATED_SOTA_GGUFS[0], "legacy/model"),
        ),
        lambda value: value["tasks"][2].__setitem__(
            "prompt",
            value["tasks"][2]["prompt"].replace(
                "Never regenerate or substitute a legacy-small model.", "Fallback is allowed."
            ),
        ),
    )
    for mutate in mutations:
        roadmap = _roadmap()
        mutate(roadmap)
        assert mod.evaluate_contract(MARKDOWN_CONTRACT, roadmap, set())["passed"] is False


def test_scenario_report_7091_preflight_separates_terminal_classes(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7091-PREFLIGHT separates positive, mismatch, and no-run."""

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
    assert disqualified["v622_task_contract_conforms_score"] == 0
    assert mod.validate_artifact(disqualified) == []

    (tmp_path / mod.DESIGN_PATH).unlink()
    blocked = mod.build_artifact(tmp_path, "20260907", output_path=tmp_path / "blocked.json")
    assert blocked["verdict_class"] == "blocked"
    assert blocked["inference_substrate_class"] == "blocked_no_run"
    assert blocked["gate_check_summary"]["failed_check"] == "v622_markdown_readable"
    assert mod.validate_artifact(blocked) == []


def test_scenario_report_7091_artifact_validator_rejects_forgery(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7091-ARTIFACT recomputes evidence and headlines."""

    _write_preconditions(tmp_path)
    good = mod.build_artifact(tmp_path, "20260907", output_path=tmp_path / "good.json")
    cases = (
        ("field_principles", {}, "field_principles_invalid"),
        ("inference_substrate", "wrong", "inference_substrate_invalid"),
        ("inference_substrate_class", "blocked_no_run", "inference_substrate_class_invalid"),
        ("execution_venue", "container", "execution_venue_invalid"),
        ("v622_task_contract_conforms_score", 0, "v622_task_contract_conforms_score_invalid"),
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
    assert "v622_task_contract_conforms_score_invalid" in mod.validate_artifact(changed)
    changed = deepcopy(good)
    changed["reproducibility_checksum"] = "sha256:bad"
    assert "reproducibility_checksum_invalid" in mod.validate_artifact(changed)


def test_req_report_7091_cli_parse_and_writable_error_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-REPORT-7091 covers command, parse-error, and writable-path behavior."""

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


def test_req_report_7091_validator_rejects_shapes_and_derives_summaries(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7091-ARTIFACT derives all count and row diagnostics."""

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
    markdown_short["v622_task_contract_conforms_score"] = 0
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


def test_req_report_7091_preconditions_report_missing_yaml_inputs(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7091-PREFLIGHT preserves each missing YAML prerequisite."""

    rows, roadmap, exclusion = mod._preconditions(tmp_path, tmp_path / "result.json")
    assert roadmap is exclusion is None
    assert {row["check"] for row in rows if not row["available"]} >= {
        "v622_markdown_readable",
        "v622_active_yaml_readable",
        "exclusion_manifest_readable",
    }


def test_req_report_7091_real_active_roadmap_emits_stable_disqualification(tmp_path: Path) -> None:
    """REQ-REPORT-7091 preserves the observed six-row active-YAML failure."""

    artifact = mod.build_artifact(ROOT, "20260907", output_path=tmp_path / "real.json")
    assert artifact["verdict_class"] == "disqualified"
    assert artifact["expected_task_count"] == 12
    assert artifact["observed_task_count"] == 6
    assert artifact["gate_check_summary"] == {
        "failed_check": "yaml_task_count",
        "expected_value": 12,
        "observed_value": 6,
        "passed": False,
    }
    assert mod.validate_artifact(artifact) == []


def test_req_report_7091_cli_rejects_invalid_artifacts_and_generated_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-REPORT-7091 returns failure for invalid stored or generated data."""

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
