"""Tests for REQ-REPORT-7038 and its V617 contract scenarios."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any, Callable

import pytest
import yaml

from carnot import experiment_7038_v617_active_contract_preflight as mod


ROOT = Path(__file__).resolve().parents[2]

EXPECTED_ROWS = (
    (
        "exp7038-v617-active-contract-preflight",
        "V617 active-roadmap and design-document contract preflight",
        "results/experiment_7038_v617_active_contract_preflight.json",
    ),
    (
        "exp7039-live-model-report-channel-forensics",
        "Official-live llama.cpp model report-channel forensic capture",
        "results/experiment_7039_v617_model_report_forensics.json",
    ),
    (
        "exp7040-typed-model-identity-report-bridge",
        "Typed raw-to-canonical ARC model identity report bridge",
        "results/experiment_7040_v617_typed_identity_bridge.json",
    ),
    (
        "exp7041-identity-report-channel-cold-audit",
        "Fresh-process ARC identity report-channel attack audit",
        "results/experiment_7041_v617_identity_attack_audit.json",
    ),
    (
        "exp7042-typed-identity-belief-shadow-trace",
        "Typed-identity official-live belief shadow trace",
        "results/experiment_7042_v617_belief_shadow_trace.json",
    ),
    (
        "exp7043-belief-shadow-trace-cold-audit",
        "Fresh-process official-live belief shadow trace audit",
        "results/experiment_7043_v617_belief_shadow_cold_audit.json",
    ),
    (
        "exp7044-uniform-belief-two-model-live-ab",
        "Two-model uniform belief influence live A/B",
        "results/experiment_7044_v617_uniform_belief_live_ab.json",
    ),
    (
        "exp7045-uniform-belief-value-cold-audit",
        "Independent uniform belief live-value cold audit",
        "results/experiment_7045_v617_uniform_belief_cold_audit.json",
    ),
    (
        "exp7046-frontier-stratified-exact-outcome-curriculum",
        "Frozen frontier-stratified exact-outcome curriculum",
        "results/experiment_7046_v617_frontier_stratified_curriculum.json",
    ),
    (
        "exp7047-selective-belief-exact-advantage-csl",
        "Exact-advantage selective belief continuous self-learning",
        "results/experiment_7047_v617_selective_belief_csl.json",
    ),
    (
        "exp7048-selective-belief-two-model-live-ab",
        "Two-model selective belief policy live A/B",
        "results/experiment_7048_v617_selective_belief_live_ab.json",
    ),
    (
        "exp7049-v617-evidence-disposition-capstone",
        "V617 evidence synthesis and release-or-retire disposition",
        "results/experiment_7049_v617_capstone_disposition.json",
    ),
)

GATES: dict[int, list[dict[str, Any]]] = {
    7040: [
        {
            "upstream": EXPECTED_ROWS[1][0],
            "artifact_field": "arc_report_channel_forensics_ready_score",
            "op": "==",
            "value": 1,
        }
    ],
    7041: [
        {
            "upstream": EXPECTED_ROWS[2][0],
            "artifact_field": "arc_typed_identity_bridge_ready_score",
            "op": "==",
            "value": 1,
        }
    ],
    7042: [
        {
            "upstream": EXPECTED_ROWS[2][0],
            "artifact_field": "arc_typed_identity_bridge_ready_score",
            "op": "==",
            "value": 1,
        },
        {
            "upstream": EXPECTED_ROWS[3][0],
            "artifact_field": "arc_identity_report_attack_audit_ready_score",
            "op": "==",
            "value": 1,
        },
    ],
    7043: [
        {
            "upstream": EXPECTED_ROWS[4][0],
            "artifact_field": "belief_shadow_transport_ready_score",
            "op": "==",
            "value": 1,
        }
    ],
    7044: [
        {
            "upstream": EXPECTED_ROWS[4][0],
            "artifact_field": "belief_shadow_transport_ready_score",
            "op": "==",
            "value": 1,
        },
        {
            "upstream": EXPECTED_ROWS[5][0],
            "artifact_field": "belief_shadow_trace_audit_ready_score",
            "op": "==",
            "value": 1,
        },
    ],
    7045: [
        {
            "upstream": EXPECTED_ROWS[6][0],
            "artifact_field": "uniform_belief_live_ab_complete_score",
            "op": "==",
            "value": 1,
        }
    ],
    7046: [
        {
            "upstream": EXPECTED_ROWS[7][0],
            "artifact_field": "uniform_belief_value_audit_complete_score",
            "op": "==",
            "value": 1,
        }
    ],
    7047: [
        {
            "upstream": EXPECTED_ROWS[8][0],
            "artifact_field": "belief_frontier_curriculum_ready_score",
            "op": "==",
            "value": 1,
        }
    ],
    7048: [
        {
            "upstream": EXPECTED_ROWS[9][0],
            "artifact_field": "selective_belief_policy_safety_score",
            "op": "==",
            "value": 1,
        },
        {
            "upstream": EXPECTED_ROWS[9][0],
            "artifact_field": "selective_belief_policy_nontrivial_score",
            "op": "==",
            "value": 1,
        },
    ],
}

PRODUCED_FIELDS = {
    7038: "v617_task_contract_conforms_score",
    7039: "arc_report_channel_forensics_ready_score",
    7040: "arc_typed_identity_bridge_ready_score",
    7041: "arc_identity_report_attack_audit_ready_score",
    7042: "belief_shadow_transport_ready_score",
    7043: "belief_shadow_trace_audit_ready_score",
    7044: "uniform_belief_live_ab_complete_score",
    7045: "uniform_belief_value_audit_complete_score",
    7046: "belief_frontier_curriculum_ready_score",
    7047: "selective_belief_policy_safety_score; selective_belief_policy_nontrivial_score",
    7048: "selective_belief_live_ab_complete_score",
    7049: "v617_capstone_complete_score",
}

LIVE_TASKS = frozenset({7039, 7042, 7044, 7048})
PRIORS = {
    number: [("exp7028-v616-active-contract-preflight", "complete_disqualified")]
    for number in range(7038, 7050)
}


def _prompt(number: int, substrate: str | None = None) -> str:
    """Build a complete task prompt without using either production parser."""

    live = number in LIVE_TASKS
    selected = substrate or ("live_llm_inference" if live else mod.INFERENCE_SUBSTRATE)
    model = (
        f"MODEL_SPECS use cached_sota_pair() with {mod.MANDATED_SOTA_GGUFS[0]}; " if live else ""
    )
    return (
        "TASK: Run a paired comparison with controls.\n"
        "REQUIRED ARTIFACT FIELDS: field_principles; preconditions_checked; "
        f"inference_substrate ({selected}); duration_s; {model}rows; "
        f"{PRODUCED_FIELDS[number]}; random_seed; reproducibility_checksum; "
        "gate_check_summary with failed check, expected value, and observed value; "
        "verifier_is_oracle (false); verdict_class "
        "(positive | circular_positive | null | blocked | disqualified | partial); "
        "honest_verdict.\n"
        "Run command: fixture only\n"
        "Do NOT push. Do NOT modify scripts/research_conductor.py.\n"
    )


def _roadmap() -> dict[str, Any]:
    """Build the active-YAML fixture independently from the Markdown text."""

    tasks = []
    for order, (task_id, title, deliverable) in enumerate(EXPECTED_ROWS, start=1):
        number = 7037 + order
        tasks.append(
            {
                "id": task_id,
                "title": title,
                "track": "fixture",
                "priority": "critical",
                "agent_type": "codex",
                "model": "gpt-5.6-sol",
                "requires_gpu": number in LIVE_TASKS,
                "max_turns": 50,
                "estimated_wall_time_min": 10,
                "per_unit_rows": True,
                "milestone": mod.MILESTONE,
                "deliverable": deliverable,
                "gated_on": deepcopy(GATES.get(number, [])),
                "prior_failures": [
                    {
                        "experiment_id": experiment_id,
                        "verdict": verdict,
                        "addressed_by": "The new task changes the failed contract mechanism.",
                        "retire_if_same_verdict": True,
                    }
                    for experiment_id, verdict in PRIORS[number]
                ],
                "prompt": _prompt(number),
            }
        )
    return {"milestone": mod.MILESTONE, "tasks": tasks}


def _markdown(rows: tuple[tuple[str, str, str], ...] = EXPECTED_ROWS) -> str:
    """Build a Markdown table from test constants, not from active-YAML tasks."""

    lines = [
        f"**Milestone:** `{mod.MILESTONE}`",
        "## 7. Phases and Exact Task Contract",
        "| # | Task ID | Title | Deliverable | Structured prerequisites |",
        "|---:|---|---|---|---|",
    ]
    ids_by_number = {7038 + index: row[0] for index, row in enumerate(rows)}
    for order, (task_id, title, deliverable) in enumerate(rows, start=1):
        number = mod.task_number(task_id)
        gates = GATES.get(number or -1, [])
        prerequisites = "; ".join(
            "Exp{} `{} {} {}`".format(
                mod.task_number(gate["upstream"]),
                gate["artifact_field"],
                gate["op"],
                gate["value"],
            )
            for gate in gates
            if mod.task_number(gate["upstream"]) in ids_by_number
        )
        lines.append(
            f"| {order} | `{task_id}` | {title} | `{deliverable}` | "
            f"{prerequisites or 'None; structurally ungated'} |"
        )
    lines.append("### Phase I")
    return "\n".join(lines) + "\n"


def _write_preconditions(root: Path, roadmap: dict[str, Any] | None = None) -> None:
    """Create every required input in an isolated directory."""

    for relative, content in (
        (mod.DESIGN_PATH, _markdown()),
        (mod.EXP7028_PATH, json.dumps({"experiment_id": 7028, "status": "complete"})),
        (mod.EXCLUSION_PATH, "retired_rows: []\n"),
        (mod.COMPLETE_PATH, "milestones: []\n"),
        (mod.SPEC_PATH, (ROOT / mod.SPEC_PATH).read_text(encoding="utf-8")),
    ):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    (root / mod.ACTIVE_ROADMAP_PATH).write_text(
        yaml.safe_dump(roadmap or _roadmap()), encoding="utf-8"
    )


def test_req_report_7038_spec_defines_exact_v617_contract() -> None:
    """REQ-REPORT-7038 defines the 12-row schema and all named scenarios."""

    spec = (ROOT / mod.SPEC_PATH).read_text(encoding="utf-8")
    section = spec[spec.index("### REQ-REPORT-7038") :]
    for scenario in ("PARITY", "GATES", "DISCIPLINE", "RESERVED", "PREFLIGHT", "ARTIFACT"):
        assert f"SCENARIO-REPORT-7038-{scenario}" in section
    assert mod.EXPECTED_TASK_COUNT == len(EXPECTED_ROWS) == 12
    assert mod.EXPECTED_TASK_IDS == tuple(row[0] for row in EXPECTED_ROWS)
    assert mod.RESERVED_EXPERIMENT_IDS == frozenset(f"exp{number}" for number in range(7033, 7038))
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert field in section


def test_scenario_report_7038_parity_accepts_twelve_independent_rows() -> None:
    """SCENARIO-REPORT-7038-PARITY accepts the exact independent contracts."""

    result = mod.evaluate_contract(_markdown(), _roadmap(), set(), set())
    assert result["passed"] is True
    assert len(result["task_contract_rows"]) == 12
    assert len(result["gate_producer_rows"]) == 12
    assert result["observed_id_order"] == [row[0] for row in EXPECTED_ROWS]
    assert result["markdown_task_rows"] is not result["yaml_task_rows"]
    for field in mod.CONTRACT_ROW_FIELDS:
        assert result[field] and all(row["passed"] for row in result[field]), field


def _missing_row(roadmap: dict[str, Any]) -> None:
    roadmap["tasks"].pop(5)


def _reordered_row(roadmap: dict[str, Any]) -> None:
    roadmap["tasks"][4], roadmap["tasks"][5] = roadmap["tasks"][5], roadmap["tasks"][4]


def _changed_title(roadmap: dict[str, Any]) -> None:
    roadmap["tasks"][0]["title"] += " changed"


def _changed_deliverable(roadmap: dict[str, Any]) -> None:
    roadmap["tasks"][0]["deliverable"] = "results/wrong.json"


def _changed_gate_field(roadmap: dict[str, Any]) -> None:
    roadmap["tasks"][2]["gated_on"][0]["artifact_field"] += "_changed"


def _missing_prompt_tail(roadmap: dict[str, Any]) -> None:
    roadmap["tasks"][0]["prompt"] = roadmap["tasks"][0]["prompt"].replace(mod.PROMPT_TAIL, "")


@pytest.mark.parametrize(
    ("name", "mutation"),
    (
        ("one_missing_row", _missing_row),
        ("one_reordered_row", _reordered_row),
        ("changed_title", _changed_title),
        ("changed_deliverable", _changed_deliverable),
        ("one_changed_gate_field", _changed_gate_field),
        ("one_missing_prompt_tail", _missing_prompt_tail),
    ),
)
def test_req_report_7038_required_contract_mutations_fail(
    name: str, mutation: Callable[[dict[str, Any]], None]
) -> None:
    """REQ-REPORT-7038 rejects every required row and prompt mutation."""

    roadmap = _roadmap()
    mutation(roadmap)
    assert mod.evaluate_contract(_markdown(), roadmap, set(), set())["passed"] is False, name


def test_scenario_report_7038_reserved_ids_cannot_be_reused() -> None:
    """SCENARIO-REPORT-7038-RESERVED rejects Exp7033 through Exp7037 reuse."""

    for reserved in sorted(mod.RESERVED_EXPERIMENT_IDS):
        roadmap = _roadmap()
        replacement = f"{reserved}-attempted-v617-reuse"
        roadmap["tasks"][0]["id"] = replacement
        markdown = _markdown(
            ((replacement, EXPECTED_ROWS[0][1], EXPECTED_ROWS[0][2]), *EXPECTED_ROWS[1:])
        )
        result = mod.evaluate_contract(markdown, roadmap, set(), set())
        assert result["passed"] is False
        assert result["reserved_id_rows"][0]["passed"] is False


@pytest.mark.parametrize(
    "mutation",
    (
        lambda roadmap: roadmap["tasks"][2]["gated_on"][0].__setitem__(
            "upstream", "exp7049-v617-evidence-disposition-capstone"
        ),
        lambda roadmap: roadmap["tasks"][2]["gated_on"][0].__setitem__(
            "artifact_field", "nested.value"
        ),
        lambda roadmap: roadmap["tasks"][1]["prompt"].replace(
            "cached_sota_pair()", "direct lookup"
        ),
    ),
)
def test_scenario_report_7038_gate_and_model_contracts_fail_closed(
    mutation: Callable[[dict[str, Any]], Any],
) -> None:
    """SCENARIO-REPORT-7038-GATES rejects invalid producers, fields, and models."""

    roadmap = _roadmap()
    returned = mutation(roadmap)
    if isinstance(returned, str):
        roadmap["tasks"][1]["prompt"] = returned
    assert mod.evaluate_contract(_markdown(), roadmap, set(), set())["passed"] is False


def test_scenario_report_7038_discipline_checks_every_yaml_rule() -> None:
    """SCENARIO-REPORT-7038-DISCIPLINE rejects all execution-rule violations."""

    mutations: tuple[Callable[[dict[str, Any]], None], ...] = (
        lambda roadmap: roadmap["tasks"][0]["prior_failures"][0].pop("addressed_by"),
        lambda roadmap: roadmap["tasks"][0].__setitem__("per_unit_rows", False),
        lambda roadmap: roadmap["tasks"][0].__setitem__(
            "prompt", roadmap["tasks"][0]["prompt"].replace("failed check", "failure")
        ),
        lambda roadmap: roadmap["tasks"][0].__setitem__(
            "prompt", roadmap["tasks"][0]["prompt"].replace(" | partial", " | deferred")
        ),
        lambda roadmap: roadmap["tasks"][0].__setitem__(
            "prompt", roadmap["tasks"][0]["prompt"].replace(mod.INFERENCE_SUBSTRATE, "unknown")
        ),
        lambda roadmap: roadmap["tasks"][1].__setitem__(
            "prompt", roadmap["tasks"][1]["prompt"] + "Headline fallback: Qwen3.5-0.8B.\n"
        ),
        lambda roadmap: roadmap["tasks"][-1].__setitem__("gated_on", deepcopy(GATES[7040])),
    )
    for mutation in mutations:
        roadmap = _roadmap()
        mutation(roadmap)
        assert mod.evaluate_contract(_markdown(), roadmap, set(), set())["passed"] is False

    assert mod.evaluate_contract(_markdown(), _roadmap(), {"exp7040"}, set())["passed"] is False
    assert mod.evaluate_contract(_markdown(), _roadmap(), set(), {"exp7041"})["passed"] is False


def test_scenario_report_7038_preflight_writes_complete_block(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7038-PREFLIGHT reports the absent Markdown prerequisite."""

    _write_preconditions(tmp_path)
    (tmp_path / mod.DESIGN_PATH).unlink()
    artifact = mod.build_artifact(
        tmp_path, "20260905", output_path=tmp_path / "results/blocked.json"
    )
    assert artifact["verdict_class"] == "blocked"
    assert artifact["honest_verdict"].startswith("complete_blocked_")
    assert artifact["gate_check_summary"]["failed_check"] == "v617_markdown_readable"
    assert artifact["gate_check_summary"]["expected_value"] == "readable_nonempty_source"
    assert "FileNotFoundError" in artifact["gate_check_summary"]["observed_value"]
    assert artifact["v617_task_contract_conforms_score"] == 0
    assert set(mod.REQUIRED_ARTIFACT_FIELDS) == artifact["field_principles"].keys()
    assert mod.validate_artifact(artifact) == []


def test_scenario_report_7038_artifact_derives_positive_and_disqualified(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7038-ARTIFACT recomputes terminal contract states."""

    _write_preconditions(tmp_path)
    positive = mod.build_artifact(
        tmp_path, "20260905", output_path=tmp_path / "results/positive.json"
    )
    assert positive["verdict_class"] == "positive"
    assert positive["v617_task_contract_conforms_score"] == 1
    assert mod.validate_artifact(positive) == []

    roadmap = _roadmap()
    _changed_gate_field(roadmap)
    (tmp_path / mod.ACTIVE_ROADMAP_PATH).write_text(yaml.safe_dump(roadmap), encoding="utf-8")
    disqualified = mod.build_artifact(
        tmp_path, "20260905", output_path=tmp_path / "results/disqualified.json"
    )
    assert disqualified["verdict_class"] == "disqualified"
    assert disqualified["gate_check_summary"]["expected_value"] is True
    assert disqualified["gate_check_summary"]["observed_value"] is False
    assert mod.validate_artifact(disqualified) == []

    forged = deepcopy(positive)
    forged["v617_task_contract_conforms_score"] = 0
    forged["reproducibility_checksum"] = mod.reproducibility_checksum(forged)
    assert "v617_task_contract_conforms_score_invalid" in mod.validate_artifact(forged)


def test_req_report_7038_current_checkout_is_blocked_without_design(tmp_path: Path) -> None:
    """REQ-REPORT-7038 preserves the real missing-design prerequisite as blocked."""

    artifact = mod.build_artifact(ROOT, "20260905", output_path=tmp_path / "experiment_7038.json")
    assert artifact["verdict_class"] == "blocked"
    assert artifact["gate_check_summary"]["failed_check"] == "v617_markdown_readable"
    assert artifact["staging_file_required_at_execution"] is False
    assert mod.validate_artifact(artifact) == []


def test_req_report_7038_parsers_reject_malformed_sources(tmp_path: Path) -> None:
    """REQ-REPORT-7038 keeps the two source parsers strict and independent."""

    markdown = _markdown()
    assert mod.task_number("bad") is None
    assert mod._normal_exp_id(7038) == "exp7038"
    assert mod._normal_exp_id("bad") is None
    with pytest.raises(ValueError, match="must be scalar"):
        mod._parse_scalar("{}")
    with pytest.raises(ValueError, match="milestone"):
        mod.parse_markdown_contract("no milestone")
    with pytest.raises(ValueError, match="section"):
        mod.parse_markdown_contract(f"**Milestone:** `{mod.MILESTONE}`")
    with pytest.raises(ValueError, match="task row"):
        mod.parse_markdown_contract(markdown.replace("| 1 |", "| 1 | extra |", 1))
    with pytest.raises(ValueError, match="task row"):
        mod.parse_markdown_contract(markdown.replace("`exp7038-", "`invalid-", 1))
    empty = (
        f"**Milestone:** `{mod.MILESTONE}`\n"
        "## 7. Phases and Exact Task Contract\n"
        "| no task rows |\n"
        "### Phase I\n"
    )
    with pytest.raises(ValueError, match="task table"):
        mod.parse_markdown_contract(empty)
    with pytest.raises(ValueError, match="structured gate"):
        mod.parse_markdown_contract(
            markdown.replace(
                "Exp7039 `arc_report_channel_forensics_ready_score == 1`",
                "Exp7039 malformed gate",
                1,
            )
        )

    for document, message in (
        ([], "mapping"),
        ({}, "tasks list"),
        ({"tasks": [1]}, "must be a mapping"),
        ({"tasks": [{"gated_on": "bad"}]}, "malformed list"),
        ({"tasks": [{"gated_on": ["bad"]}]}, "malformed gate"),
    ):
        with pytest.raises(ValueError, match=message):
            mod.parse_yaml_contract(document)
    path = tmp_path / "list.yaml"
    path.write_text("[]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="mapping required"):
        mod.load_yaml(path)


def test_req_report_7038_id_ledgers_and_prompt_helpers_are_defensive() -> None:
    """REQ-REPORT-7038 handles malformed ledger rows without inventing IDs."""

    assert mod._required_block("no field block") == ""
    assert mod._substrate("no substrate") is None
    assert mod._comparison_prompt("single result") is False
    assert mod._legacy_small_headline_path("Qwen3.5-0.8B smoke model") is False
    assert mod._legacy_small_headline_path("No headline fallback: Qwen3.5-0.8B") is False
    assert mod.retired_experiment_ids([]) == set()
    assert mod.completed_experiment_ids([]) == set()
    manifest = {
        "not_retired": [{"experiment_id": 7038}],
        "retired_rows": [
            "bad",
            {"experiment_id": 7038},
            {
                "experiment_ids": ["exp7039-task", "bad"],
                "un_retired_experiment_ids": [7038],
            },
        ],
    }
    assert mod.retired_experiment_ids(manifest) == {"exp7039"}
    assert mod.completed_experiment_ids({"milestones": "bad"}) == set()
    complete = {
        "milestones": [
            "bad",
            {"tasks": "bad"},
            {"tasks": ["bad", {"id": "bad"}, {"id": "exp7040-complete"}]},
        ]
    }
    assert mod.completed_experiment_ids(complete) == {"exp7040"}
    task = mod.parse_yaml_contract(_roadmap())["tasks"][0]
    task["prior_failures"] = []
    assert mod._prior_rows(task)[0]["passed"] is True


def test_req_report_7038_precondition_error_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-REPORT-7038-PREFLIGHT preserves every missing-input reason."""

    rows, roadmap, exclusion, complete = mod._preconditions(
        tmp_path, tmp_path / "results/result.json"
    )
    assert roadmap is exclusion is complete is None
    unavailable = {row["check"] for row in rows if not row["available"]}
    assert unavailable >= {
        "v617_markdown_readable",
        "v617_active_yaml_readable",
        "exp7028_evidence_readable",
        "exclusion_manifest_readable",
        "completed_experiment_ledger_readable",
        "reporting_spec_readable",
    }

    _write_preconditions(tmp_path)
    (tmp_path / mod.STAGING_ROADMAP_PATH).write_text("not read\n", encoding="utf-8")
    rows, roadmap, exclusion, complete = mod._preconditions(
        tmp_path, tmp_path / "results/result.json"
    )
    assert roadmap and exclusion and complete
    staging = next(row for row in rows if row["check"] == "staging_file_not_required")
    assert staging["observed_value"] == "present_but_not_read"

    monkeypatch.setattr(
        mod.tempfile,
        "NamedTemporaryFile",
        lambda **_kwargs: (_ for _ in ()).throw(OSError("read only")),
    )
    writable, observed = mod._artifact_path_writable(tmp_path / "blocked.json")
    assert writable is False and "read only" in observed


def test_req_report_7038_parse_failure_is_terminal_disqualified(tmp_path: Path) -> None:
    """REQ-REPORT-7038 treats malformed available contracts as disqualified."""

    _write_preconditions(tmp_path)
    (tmp_path / mod.DESIGN_PATH).write_text("readable but malformed\n", encoding="utf-8")
    artifact = mod.build_artifact(
        tmp_path, "20260905", output_path=tmp_path / "results/malformed.json"
    )
    assert artifact["verdict_class"] == "disqualified"
    assert artifact["gate_check_summary"]["failed_check"] == "contract_parse"
    assert mod.validate_artifact(artifact) == []

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
                "yaml_task_rows": [],
                "task_contract_rows": [],
            }
        )["failed_check"]
        == "observed_task_count"
    )


def test_req_report_7038_validator_rejects_forged_artifacts(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7038-ARTIFACT rejects each forged headline field."""

    _write_preconditions(tmp_path)
    good = mod.build_artifact(tmp_path, "20260905", output_path=tmp_path / "good.json")
    cases = (
        ("field_principles", {}, "field_principles_invalid"),
        ("inference_substrate", "wrong", "inference_substrate_invalid"),
        ("verifier_is_oracle", True, "verifier_is_oracle_invalid"),
        ("expected_task_count", 11, "expected_task_count_invalid"),
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
    for field, value, expected_error in cases:
        changed = deepcopy(good)
        changed[field] = value
        changed["reproducibility_checksum"] = mod.reproducibility_checksum(changed)
        assert expected_error in mod.validate_artifact(changed), field

    missing = deepcopy(good)
    missing.pop("rows")
    assert mod.validate_artifact(missing) == ["missing_required_field:rows"]
    checksum = deepcopy(good)
    checksum["reproducibility_checksum"] = "sha256:bad"
    assert "reproducibility_checksum_invalid" in mod.validate_artifact(checksum)
    summary = deepcopy(good)
    summary["gate_check_summary"]["passed"] = False
    summary["reproducibility_checksum"] = mod.reproducibility_checksum(summary)
    assert "gate_check_summary_not_derived" in mod.validate_artifact(summary)


def test_req_report_7038_score_rejects_forged_evidence_rows(tmp_path: Path) -> None:
    """REQ-REPORT-7038 recomputes conformance from all preserved evidence rows."""

    _write_preconditions(tmp_path)
    good = mod.build_artifact(tmp_path, "20260905", output_path=tmp_path / "good.json")
    mutations: tuple[Callable[[dict[str, Any]], None], ...] = (
        lambda value: value["markdown_task_rows"][0].__setitem__("id", "exp9999-wrong"),
        lambda value: value["yaml_task_rows"][0].__setitem__("id", "exp9999-wrong"),
        lambda value: value["task_contract_rows"][0].__setitem__("passed", False),
        lambda value: value["task_contract_rows"][0].__setitem__("order", 99),
        lambda value: value.__setitem__("title_parity_rows", []),
        lambda value: value["title_parity_rows"][0].__setitem__("passed", False),
        lambda value: value["gate_producer_rows"].pop(),
        lambda value: value.__setitem__("prior_failure_rows", []),
        lambda value: value.__setitem__("prior_failure_rows", value["prior_failure_rows"][:1]),
    )
    for mutation in mutations:
        changed = deepcopy(good)
        mutation(changed)
        assert mod._contract_score_from_artifact(changed) == 0


def test_req_report_7038_cli_validation_and_generation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-REPORT-7038 exposes deterministic validation and generation exits."""

    assert mod._date_argument("20260905") == "20260905"
    with pytest.raises(Exception, match="YYYYMMDD"):
        mod._date_argument("2026-09-05")
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
    good = mod.build_artifact(tmp_path, "20260905", output_path=tmp_path / "good.json")
    good_path = tmp_path / "good.json"
    good_path.write_text(json.dumps(good), encoding="utf-8")
    assert mod.main(["--validate-artifact", str(good_path)]) == 0

    output = Path("results/relative.json")
    assert mod.main(["--date", "20260905", "--root", str(tmp_path), "--output", str(output)]) == 0
    assert (
        json.loads((tmp_path / output).read_text(encoding="utf-8"))["verdict_class"] == "positive"
    )

    monkeypatch.setattr(mod, "validate_artifact", lambda _artifact: ["forced failure"])
    assert (
        mod.main(
            [
                "--date",
                "20260905",
                "--root",
                str(tmp_path),
                "--output",
                str(tmp_path / "invalid.json"),
            ]
        )
        == 1
    )
