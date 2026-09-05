"""Tests for REQ-REPORT-7028 and its V616 contract scenarios."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import subprocess
from typing import Any, Callable

import pytest
import yaml

from carnot import experiment_7028_v616_active_contract_preflight as mod


ROOT = Path(__file__).resolve().parents[2]

EXPECTED_ROWS = (
    (
        "exp7028-v616-active-contract-preflight",
        "V616 active-roadmap and design-document contract preflight",
        "results/experiment_7028_v616_active_contract_preflight.json",
    ),
    (
        "exp7029-v616-sota-scope-audit",
        "V616 post-marker source delta and experiment-scope audit",
        "results/experiment_7029_v616_sota_scope_audit.json",
    ),
    (
        "exp7030-arc-gguf-model-identity-bridge",
        "ARC GGUF snapshot-to-blob model identity bridge",
        "results/experiment_7030_arc_gguf_model_identity_bridge.json",
    ),
    (
        "exp7031-arc-model-identity-cold-audit",
        "Fresh-process ARC model identity and alias-confusion audit",
        "results/experiment_7031_arc_model_identity_cold_audit.json",
    ),
    (
        "exp7032-repaired-belief-shadow-live-trace",
        "Repaired provenance-complete live belief shadow trace",
        "results/experiment_7032_repaired_belief_shadow_live_trace.json",
    ),
    (
        "exp7033-uniform-belief-live-ab",
        "Held-mechanic uniform belief live A/B and retirement test",
        "results/experiment_7033_uniform_belief_live_ab.json",
    ),
    (
        "exp7034-live-belief-cold-audit",
        "Independent live belief value and provenance cold audit",
        "results/experiment_7034_live_belief_cold_audit.json",
    ),
    (
        "exp7035-selective-belief-csl",
        "Prospective selective belief-use continuous self-learning A/B",
        "results/experiment_7035_selective_belief_csl.json",
    ),
    (
        "exp7036-belief-release-or-retire",
        "ARC belief policy release-or-retire lifecycle closure",
        "results/experiment_7036_belief_release_or_retire.json",
    ),
    (
        "exp7037-v616-capstone",
        "V616 independent evidence capstone and V617 handoff",
        "results/experiment_7037_v616_capstone.json",
    ),
)

GATES: dict[int, list[dict[str, Any]]] = {
    7031: [
        {
            "upstream": EXPECTED_ROWS[2][0],
            "artifact_field": "arc_model_identity_bridge_ready_score",
            "op": "==",
            "value": 1,
        }
    ],
    7032: [
        {
            "upstream": EXPECTED_ROWS[2][0],
            "artifact_field": "arc_model_identity_bridge_ready_score",
            "op": "==",
            "value": 1,
        },
        {
            "upstream": EXPECTED_ROWS[3][0],
            "artifact_field": "arc_model_identity_audit_ready_score",
            "op": "==",
            "value": 1,
        },
    ],
    7033: [
        {
            "upstream": EXPECTED_ROWS[4][0],
            "artifact_field": "belief_shadow_trace_ready_score",
            "op": "==",
            "value": 1,
        }
    ],
    7034: [
        {
            "upstream": EXPECTED_ROWS[5][0],
            "artifact_field": "belief_live_comparison_complete_score",
            "op": "==",
            "value": 1,
        }
    ],
    7035: [
        {
            "upstream": EXPECTED_ROWS[6][0],
            "artifact_field": "belief_live_value_audit_complete_score",
            "op": "==",
            "value": 1,
        }
    ],
    7036: [
        {
            "upstream": EXPECTED_ROWS[6][0],
            "artifact_field": "belief_live_value_audit_complete_score",
            "op": "==",
            "value": 1,
        },
        {
            "upstream": EXPECTED_ROWS[7][0],
            "artifact_field": "selective_belief_policy_complete_score",
            "op": "==",
            "value": 1,
        },
    ],
}

PRODUCED_FIELDS = {
    7028: "v616_task_contract_conforms_score",
    7029: "v616_sota_scope_complete_score",
    7030: "arc_model_identity_bridge_ready_score",
    7031: "arc_model_identity_audit_ready_score",
    7032: "belief_shadow_trace_ready_score",
    7033: "belief_live_comparison_complete_score",
    7034: "belief_live_value_audit_complete_score",
    7035: "selective_belief_policy_complete_score",
    7036: "belief_lifecycle_closed_score",
    7037: "v617_handoff",
}

LIVE_TASKS = frozenset({7032, 7033, 7035})
COMPARATIVE_TASKS = frozenset({7032, 7033, 7034, 7035})
PRIORS = {
    7028: [("exp7016-v615-source-contract-preflight", "blocked_v615_contract_preflight")],
    7030: [("exp7025-belief-shadow-live-trace", "blocked_belief_shadow_live_trace")],
    7031: [("exp7025-belief-shadow-live-trace", "blocked_belief_shadow_live_trace")],
    7032: [("exp7025-belief-shadow-live-trace", "blocked_belief_shadow_live_trace")],
}


def _prompt(number: int, substrate: str | None = None) -> str:
    """Build a complete test prompt from facts independent of production parsing."""

    live = number in LIVE_TASKS
    chosen = substrate or ("live_llm_inference" if live else "aggregation_from_upstream_artifacts")
    comparison = "paired A/B comparison with controls; " if number in COMPARATIVE_TASKS else ""
    model = f"MODEL_SPECS: [{mod.MANDATED_SOTA_GGUFS[0]}]; " if live else ""
    return (
        f"TASK: Run the {comparison}V616 task.\n"
        "REQUIRED ARTIFACT FIELDS: field_principles; preconditions_checked; "
        f"inference_substrate ({chosen}); duration_s; {model}rows; "
        f"{PRODUCED_FIELDS[number]}; random_seed; reproducibility_checksum; "
        "gate_check_summary with failed check, expected value, and observed value; "
        "verifier_is_oracle (false); verdict_class "
        "(positive | circular_positive | null | blocked | disqualified | partial); "
        "honest_verdict.\n"
        "Run command: fixture only\n"
        "Do NOT push. Do NOT modify scripts/research_conductor.py.\n"
    )


def _roadmap() -> dict[str, Any]:
    """Build the active YAML fixture without reading the Markdown source."""

    tasks = []
    for order, (task_id, title, deliverable) in enumerate(EXPECTED_ROWS, start=1):
        number = 7027 + order
        tasks.append(
            {
                "id": task_id,
                "title": title,
                "deliverable": deliverable,
                "milestone": mod.MILESTONE,
                "requires_gpu": number in LIVE_TASKS,
                "per_unit_rows": True,
                "gated_on": deepcopy(GATES.get(number, [])),
                "prior_failures": [
                    {
                        "experiment_id": experiment_id,
                        "verdict": verdict,
                        "addressed_by": "The task changes the failed mechanism and keeps a falsifiable gate.",
                        "retire_if_same_verdict": True,
                    }
                    for experiment_id, verdict in PRIORS.get(number, [])
                ],
                "prompt": _prompt(number),
            }
        )
    return {"milestone": mod.MILESTONE, "tasks": tasks}


def _copy_input(root: Path, relative: Path) -> None:
    """Copy one required read-only input into an isolated fixture."""

    target = root / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes((ROOT / relative).read_bytes())


def _write_preconditions(root: Path, roadmap: dict[str, Any] | None = None) -> None:
    """Create every required input except the staging roadmap."""

    for relative in (
        mod.DESIGN_PATH,
        mod.V615_PREFLIGHT_PATH,
        mod.V615_CAPSTONE_PATH,
        mod.EXCLUSION_PATH,
        mod.SPEC_PATH,
    ):
        _copy_input(root, relative)
    (root / mod.ACTIVE_ROADMAP_PATH).write_text(
        yaml.safe_dump(roadmap or _roadmap()), encoding="utf-8"
    )


def test_req_report_7028_spec_and_markdown_define_ten_rows() -> None:
    """REQ-REPORT-7028 and SCENARIO-REPORT-7028-PARITY define all rows."""

    spec = (ROOT / mod.SPEC_PATH).read_text(encoding="utf-8")
    section = spec[spec.index("### REQ-REPORT-7028") :]
    for scenario in ("ACTIVE", "PREFLIGHT", "PARITY", "GATES", "DISCIPLINE", "ARTIFACT"):
        assert f"SCENARIO-REPORT-7028-{scenario}" in section
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert field in section

    parsed = mod.parse_markdown_contract((ROOT / mod.DESIGN_PATH).read_text(encoding="utf-8"))
    assert parsed["milestone"] == mod.MILESTONE
    assert [(row["id"], row["title"], row["deliverable"]) for row in parsed["tasks"]] == list(
        EXPECTED_ROWS
    )
    assert sum(len(row["gates"]) for row in parsed["tasks"]) == 8


def test_scenario_report_7028_active_uses_no_staging_file(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7028-ACTIVE proves the active file is sufficient."""

    _write_preconditions(tmp_path)
    assert not (tmp_path / mod.STAGING_ROADMAP_PATH).exists()
    artifact = mod.build_artifact(
        tmp_path, "20260905", output_path=tmp_path / "results/active.json"
    )

    assert artifact["verdict_class"] == "positive"
    assert artifact["staging_file_required_at_execution"] is False
    staging = next(
        row for row in artifact["preconditions_checked"] if row["check"] == "staging_file_not_required"
    )
    assert staging["available"] is True
    assert staging["observed_value"] == "absent_and_not_read"
    assert mod.validate_artifact(artifact) == []


def test_scenario_report_7028_conforming_fixture_checks_every_row() -> None:
    """SCENARIO-REPORT-7028-PARITY uses independent Markdown and YAML rows."""

    result = mod.evaluate_contract(
        (ROOT / mod.DESIGN_PATH).read_text(encoding="utf-8"), _roadmap(), set()
    )

    assert result["passed"] is True
    assert result["observed_id_order"] == [row[0] for row in EXPECTED_ROWS]
    assert len(result["task_contract_rows"]) == 10
    assert len(result["gate_producer_rows"]) == 8
    for name in mod.CONTRACT_ROW_FIELDS:
        assert all(row["passed"] for row in result[name]), name


def _mutate_task_count(roadmap: dict[str, Any]) -> None:
    roadmap["tasks"].pop()


def _mutate_order(roadmap: dict[str, Any]) -> None:
    roadmap["tasks"][0], roadmap["tasks"][1] = roadmap["tasks"][1], roadmap["tasks"][0]


def _mutate_title(roadmap: dict[str, Any]) -> None:
    roadmap["tasks"][0]["title"] += " changed"


def _mutate_deliverable(roadmap: dict[str, Any]) -> None:
    roadmap["tasks"][0]["deliverable"] = "results/wrong.json"


def _mutate_milestone(roadmap: dict[str, Any]) -> None:
    roadmap["tasks"][2]["milestone"] = "2026.09.999"


def _mutate_gate(roadmap: dict[str, Any]) -> None:
    roadmap["tasks"][3]["gated_on"][0]["op"] = ">="


def _mutate_producer_alias(roadmap: dict[str, Any]) -> None:
    roadmap["tasks"][2]["prompt"] = roadmap["tasks"][2]["prompt"].replace(
        "arc_model_identity_bridge_ready_score;", "arc_model_identity_bridge_ready_score.alias;"
    )


def _mutate_producer_missing(roadmap: dict[str, Any]) -> None:
    roadmap["tasks"][3]["gated_on"][0]["upstream"] = "exp9999-missing"


def _mutate_prior(roadmap: dict[str, Any], field: str) -> None:
    roadmap["tasks"][0]["prior_failures"][0][field] = (
        False if field == "retire_if_same_verdict" else ""
    )


def _mutate_override(roadmap: dict[str, Any]) -> None:
    roadmap["tasks"][0]["operator_override"] = "because"


def _mutate_model(roadmap: dict[str, Any]) -> None:
    roadmap["tasks"][4]["prompt"] = roadmap["tasks"][4]["prompt"].replace(
        mod.MANDATED_SOTA_GGUFS[0], "old/model-GGUF"
    )


def _mutate_legacy_headline(roadmap: dict[str, Any]) -> None:
    roadmap["tasks"][4]["prompt"] += "Headline fallback: Qwen3.5-0.8B.\n"


def _mutate_substrate(roadmap: dict[str, Any]) -> None:
    roadmap["tasks"][0]["prompt"] = roadmap["tasks"][0]["prompt"].replace(
        "aggregation_from_upstream_artifacts", "deterministic_contract_no_llm"
    )


def _mutate_artifact_field(roadmap: dict[str, Any]) -> None:
    roadmap["tasks"][4]["prompt"] = roadmap["tasks"][4]["prompt"].replace(
        "verdict_class", "result_class"
    )


def _mutate_comparison_rows(roadmap: dict[str, Any]) -> None:
    roadmap["tasks"][4]["per_unit_rows"] = False


def _mutate_tail(roadmap: dict[str, Any]) -> None:
    roadmap["tasks"][0]["prompt"] += "Trailing text.\n"


def _mutate_capstone_gate(roadmap: dict[str, Any]) -> None:
    roadmap["tasks"][-1]["gated_on"] = deepcopy(GATES[7031])


def _mutate_advisory_gate(roadmap: dict[str, Any]) -> None:
    roadmap["tasks"][0]["gated_on"] = deepcopy(GATES[7031])


MUTATIONS: tuple[tuple[str, Callable[[dict[str, Any]], None]], ...] = (
    ("task_count", _mutate_task_count),
    ("order", _mutate_order),
    ("title", _mutate_title),
    ("deliverable", _mutate_deliverable),
    ("milestone", _mutate_milestone),
    ("gate", _mutate_gate),
    ("producer_alias", _mutate_producer_alias),
    ("producer_missing", _mutate_producer_missing),
    ("prior_experiment_id", lambda value: _mutate_prior(value, "experiment_id")),
    ("prior_verdict", lambda value: _mutate_prior(value, "verdict")),
    ("prior_addressed_by", lambda value: _mutate_prior(value, "addressed_by")),
    ("prior_retire", lambda value: _mutate_prior(value, "retire_if_same_verdict")),
    ("operator_override", _mutate_override),
    ("model", _mutate_model),
    ("legacy_headline", _mutate_legacy_headline),
    ("substrate", _mutate_substrate),
    ("artifact_field", _mutate_artifact_field),
    ("comparison_rows", _mutate_comparison_rows),
    ("tail", _mutate_tail),
    ("capstone_gate", _mutate_capstone_gate),
    ("advisory_gate", _mutate_advisory_gate),
)


@pytest.mark.parametrize(("name", "mutation"), MUTATIONS)
def test_scenario_report_7028_required_mutations_fail(
    name: str, mutation: Callable[[dict[str, Any]], None]
) -> None:
    """SCENARIO-REPORT-7028-DISCIPLINE rejects every contract mutation."""

    roadmap = _roadmap()
    mutation(roadmap)
    result = mod.evaluate_contract(
        (ROOT / mod.DESIGN_PATH).read_text(encoding="utf-8"), roadmap, set()
    )
    assert result["passed"] is False, name


@pytest.mark.parametrize(
    ("mutation", "retired"),
    (
        (lambda _roadmap: None, {"exp7030"}),
        (lambda roadmap: roadmap["tasks"][0].__setitem__("id", "exp2091-retired"), {"exp2091"}),
        (
            lambda roadmap: roadmap["tasks"][3]["gated_on"][0].__setitem__(
                "artifact_field", "arc_model_identity_bridge_ready_score.value"
            ),
            set(),
        ),
    ),
)
def test_scenario_report_7028_gate_and_retirement_checks_fail_closed(
    mutation: Callable[[dict[str, Any]], None], retired: set[str]
) -> None:
    """SCENARIO-REPORT-7028-GATES rejects retired IDs and non-bare fields."""

    roadmap = _roadmap()
    mutation(roadmap)
    result = mod.evaluate_contract(
        (ROOT / mod.DESIGN_PATH).read_text(encoding="utf-8"), roadmap, retired
    )
    assert result["passed"] is False


def test_req_report_7028_legal_substrates_are_exact() -> None:
    """REQ-REPORT-7028 keeps the six-value CLAUDE.md substrate set closed."""

    assert mod.LEGAL_INFERENCE_SUBSTRATES == {
        "live_llm_inference",
        "verifier_ensemble_against_cached_candidates",
        "aggregation_from_upstream_artifacts",
        "hardware_smoke",
        "offline_arcade_live_agent_runtime_self_discovery_no_llm",
        "live_llm_embedding_extraction",
    }


def test_scenario_report_7028_preflight_blocks_missing_active_yaml(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7028-PREFLIGHT emits a complete active-file block."""

    for relative in (
        mod.DESIGN_PATH,
        mod.V615_PREFLIGHT_PATH,
        mod.V615_CAPSTONE_PATH,
        mod.EXCLUSION_PATH,
        mod.SPEC_PATH,
    ):
        _copy_input(tmp_path, relative)

    artifact = mod.build_artifact(
        tmp_path, "20260905", output_path=tmp_path / "results/blocked.json"
    )
    assert artifact["verdict_class"] == "blocked"
    assert artifact["gate_check_summary"]["failed_check"] == "v616_active_yaml_readable"
    assert artifact["gate_check_summary"]["expected_value"] == "readable_nonempty_yaml_mapping"
    assert "FileNotFoundError" in artifact["gate_check_summary"]["observed_value"]
    assert artifact["v616_task_contract_conforms_score"] == 0
    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= artifact.keys()
    assert set(mod.REQUIRED_ARTIFACT_FIELDS) == artifact["field_principles"].keys()
    assert mod.validate_artifact(artifact) == []


def test_scenario_report_7028_artifact_derives_positive_and_disqualified(
    tmp_path: Path,
) -> None:
    """SCENARIO-REPORT-7028-ARTIFACT derives both contract terminal states."""

    _write_preconditions(tmp_path)
    positive = mod.build_artifact(
        tmp_path, "20260905", output_path=tmp_path / "results/positive.json"
    )
    assert positive["verdict_class"] == "positive"
    assert positive["v616_task_contract_conforms_score"] == 1
    assert mod.validate_artifact(positive) == []

    changed = _roadmap()
    changed["tasks"][0]["title"] += " changed"
    (tmp_path / mod.ACTIVE_ROADMAP_PATH).write_text(yaml.safe_dump(changed), encoding="utf-8")
    disqualified = mod.build_artifact(
        tmp_path, "20260905", output_path=tmp_path / "results/disqualified.json"
    )
    assert disqualified["verdict_class"] == "disqualified"
    assert disqualified["title_parity_rows"][0]["observed"].endswith("changed")
    assert mod.validate_artifact(disqualified) == []


def test_req_report_7028_current_active_roadmap_is_reported_without_repair(tmp_path: Path) -> None:
    """REQ-REPORT-7028 preserves the current five-row active-YAML mismatch."""

    artifact = mod.build_artifact(
        ROOT, "20260905", output_path=tmp_path / "experiment_7028.json"
    )
    assert artifact["verdict_class"] == "disqualified"
    assert artifact["observed_task_count"] == 5
    assert artifact["observed_id_order"][-1].startswith("exp7032-")
    assert artifact["gate_check_summary"] == {
        "failed_check": "observed_task_count",
        "expected_value": 10,
        "observed_value": 5,
        "passed": False,
    }
    assert mod.validate_artifact(artifact) == []


def test_req_report_7028_parsers_reject_malformed_contracts(tmp_path: Path) -> None:
    """REQ-REPORT-7028 rejects malformed Markdown and YAML independently."""

    markdown = (ROOT / mod.DESIGN_PATH).read_text(encoding="utf-8")
    with pytest.raises(ValueError, match="must be scalar"):
        mod._parse_scalar("{}")
    with pytest.raises(ValueError, match="milestone"):
        mod.parse_markdown_contract("no milestone")
    with pytest.raises(ValueError, match="section"):
        mod.parse_markdown_contract(f"**Milestone:** `{mod.MILESTONE}`")
    with pytest.raises(ValueError, match="task row"):
        mod.parse_markdown_contract(markdown.replace("| 1 | `exp7028", "| 1 | extra | `exp7028", 1))
    with pytest.raises(ValueError, match="task row"):
        mod.parse_markdown_contract(markdown.replace("`exp7028-v616", "`invalid-v616", 1))
    empty = (
        f"**Milestone:** `{mod.MILESTONE}`\n"
        "## 7. Phases and Exact Task Contract\n"
        "| no task rows |\n"
        "### Phase I"
    )
    with pytest.raises(ValueError, match="task table"):
        mod.parse_markdown_contract(empty)
    with pytest.raises(ValueError, match="structured gate"):
        mod.parse_markdown_contract(
            markdown.replace(
                "Exp7030 `arc_model_identity_bridge_ready_score == 1`", "Exp7030 bad gate", 1
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


def test_req_report_7028_manifest_and_precondition_defensive_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-REPORT-7028 keeps restorations and missing inputs explicit."""

    assert mod._normal_exp_id("bad") is None
    assert mod._required_block("no required block") == ""
    assert mod._documented_override(7) is False
    assert mod._legacy_small_headline_path("Qwen3.5-0.8B is a smoke model.") is False
    assert mod.retired_experiment_ids([]) == set()
    manifest = {
        "not_retired": [{"experiment_id": 7028}],
        "retired_rows": [
            "bad",
            {"experiment_id": 7028},
            {
                "experiment_ids": ["exp7029-task", "bad"],
                "un_retired_experiment_ids": [7028],
            },
        ],
    }
    assert mod.retired_experiment_ids(manifest) == {"exp7029"}

    rows, roadmap, exclusion = mod._preconditions(tmp_path, tmp_path / "result.json")
    assert roadmap is None and exclusion is None
    assert {row["check"] for row in rows if not row["available"]} >= {
        "v616_markdown_readable",
        "v616_active_yaml_readable",
        "v615_preflight_evidence_readable",
        "v615_capstone_evidence_readable",
        "exclusion_manifest_readable",
        "reporting_spec_readable",
    }

    monkeypatch.setattr(
        mod.tempfile,
        "NamedTemporaryFile",
        lambda **_kwargs: (_ for _ in ()).throw(OSError("read only")),
    )
    assert mod._artifact_path_writable(tmp_path / "blocked.json")[0] is False


def test_req_report_7028_parse_and_command_blocks_are_terminal(tmp_path: Path) -> None:
    """REQ-REPORT-7028 distinguishes malformed contracts from command blocks."""

    _write_preconditions(tmp_path)
    (tmp_path / mod.DESIGN_PATH).write_text("readable but malformed\n", encoding="utf-8")
    malformed = mod.build_artifact(
        tmp_path, "20260905", output_path=tmp_path / "malformed.json"
    )
    assert malformed["verdict_class"] == "disqualified"
    assert malformed["gate_check_summary"]["failed_check"] == "contract_parse"
    assert mod.validate_artifact(malformed) == []

    _copy_input(tmp_path, mod.DESIGN_PATH)
    failed_receipt = {
        "name": "focused_tests_100pct_coverage",
        "outcome": "failed",
        "terminal": True,
        "passed": False,
    }
    blocked = mod.build_artifact(
        tmp_path,
        "20260905",
        output_path=tmp_path / "command.json",
        command_rows=[failed_receipt],
    )
    assert blocked["verdict_class"] == "blocked"
    assert blocked["gate_check_summary"]["failed_check"] == "focused_tests_100pct_coverage"
    assert mod.validate_artifact(blocked) == []

    summary = mod._contract_failure_summary(
        {"markdown_task_count": 9, "yaml_task_rows": [], "task_contract_rows": []}
    )
    assert summary["failed_check"] == "markdown_task_count"


def test_req_report_7028_validator_rejects_forged_fields(tmp_path: Path) -> None:
    """REQ-REPORT-7028 recomputes every headline artifact field."""

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
        ("verdict_class", "blocked", "verdict_class_not_derived"),
        ("gate_check_summary", [], "gate_check_summary_invalid"),
    )
    for field, value, expected_error in cases:
        changed = deepcopy(good)
        changed[field] = value
        changed["reproducibility_checksum"] = mod.reproducibility_checksum(changed)
        assert expected_error in mod.validate_artifact(changed), field

    missing = dict(good)
    missing.pop("rows")
    assert mod.validate_artifact(missing) == ["missing_required_field:rows"]

    row_forgery = deepcopy(good)
    row_forgery["artifact_field_rows"][0]["passed"] = False
    row_forgery["reproducibility_checksum"] = mod.reproducibility_checksum(row_forgery)
    assert "v616_task_contract_conforms_score_invalid" in mod.validate_artifact(row_forgery)

    missing_gate_row = deepcopy(good)
    missing_gate_row["gate_producer_rows"].pop()
    assert mod._contract_score_from_artifact(missing_gate_row) == 0

    missing_prior_row = deepcopy(good)
    missing_prior_row["prior_failure_rows"] = []
    assert mod._contract_score_from_artifact(missing_prior_row) == 0

    checksum_forgery = deepcopy(good)
    checksum_forgery["reproducibility_checksum"] = "sha256:bad"
    assert "reproducibility_checksum_invalid" in mod.validate_artifact(checksum_forgery)

    summary_forgery = deepcopy(good)
    summary_forgery["gate_check_summary"]["passed"] = False
    summary_forgery["reproducibility_checksum"] = mod.reproducibility_checksum(summary_forgery)
    assert "gate_check_summary_not_derived" in mod.validate_artifact(summary_forgery)


def test_req_report_7028_command_receipts_cover_all_outcomes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-REPORT-7028 records pass, failure, timeout, and unavailable tools."""

    passed = mod._run_command(tmp_path, "pass", ["/bin/sh", "-c", "printf clean"])
    failed = mod._run_command(tmp_path, "fail", ["/bin/sh", "-c", "exit 3"])
    assert passed["passed"] is True and passed["stdout"] == "clean"
    assert failed["passed"] is False and failed["exit_code"] == 3

    monkeypatch.setattr(
        mod.subprocess,
        "run",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            subprocess.TimeoutExpired("slow", 1, output="partial", stderr="late")
        ),
    )
    assert mod._run_command(tmp_path, "slow", ["slow"])["outcome"] == "timeout"
    monkeypatch.setattr(
        mod.subprocess,
        "run",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("missing")),
    )
    assert mod._run_command(tmp_path, "missing", ["missing"])["outcome"] == "tool_error"

    names: list[str] = []

    def record(_root: Path, name: str, _argv: list[str]) -> dict[str, Any]:
        names.append(name)
        return {"name": name, "terminal": True, "passed": True}

    monkeypatch.setattr(mod, "_run_command", record)
    rows = mod.run_validation_commands(ROOT, tmp_path / "artifact.json")
    assert len(rows) == 9
    assert names == [row["name"] for row in rows]


def test_req_report_7028_cli_validation_and_generation_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-REPORT-7028 exposes deterministic validation and generation exits."""

    assert mod._date_argument("20260905") == "20260905"
    with pytest.raises(Exception, match="YYYYMMDD"):
        mod._date_argument("2026-09-05")
    with pytest.raises(SystemExit):
        mod.main([])

    unreadable = tmp_path / "unreadable.json"
    assert mod.main(["--validate-artifact", str(unreadable)]) == 1
    list_path = tmp_path / "list.json"
    list_path.write_text("[]\n", encoding="utf-8")
    assert mod.main(["--validate-artifact", str(list_path)]) == 1

    _write_preconditions(tmp_path)
    good = mod.build_artifact(tmp_path, "20260905", output_path=tmp_path / "good.json")
    good_path = tmp_path / "good.json"
    good_path.write_text(json.dumps(good), encoding="utf-8")
    assert mod.main(["--validate-artifact", str(good_path)]) == 0
    bad_path = tmp_path / "bad.json"
    bad_path.write_text("{}\n", encoding="utf-8")
    assert mod.main(["--validate-artifact", str(bad_path)]) == 1

    monkeypatch.setattr(mod, "run_validation_commands", lambda *_args: [])
    output = tmp_path / "nested/positive.json"
    assert mod.main(
        ["--date", "20260905", "--root", str(tmp_path), "--output", str(output)]
    ) == 0
    assert json.loads(output.read_text(encoding="utf-8"))["verdict_class"] == "positive"

    blocked_output = tmp_path / "blocked.json"
    (tmp_path / mod.ACTIVE_ROADMAP_PATH).unlink()
    assert mod.main(
        ["--date", "20260905", "--root", str(tmp_path), "--output", str(blocked_output)]
    ) == 0
    assert json.loads(blocked_output.read_text(encoding="utf-8"))["verdict_class"] == "blocked"

    monkeypatch.setattr(mod, "validate_artifact", lambda _artifact: ["forced failure"])
    assert mod.main(
        ["--date", "20260905", "--root", str(ROOT), "--output", str(tmp_path / "bad-final.json")]
    ) == 1
