"""Focused tests for the V603 executable manifest contract.

Spec refs: REQ-REPORT-6885 and SCENARIO-REPORT-6885-*.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from carnot import experiment_6885_v603_executable_manifest_branch_contract as mod


ROOT = Path(__file__).resolve().parents[2]
DESIGN_TEXT = (ROOT / "openspec/change-proposals/research-roadmap-vNEXT.md").read_text(
    encoding="utf-8"
)
SCHEMA_TEXT = (ROOT / "scripts/roadmap_schema.py").read_text(encoding="utf-8")
GATE_TEXT = (ROOT / "scripts/conductor_gates.py").read_text(encoding="utf-8")


def _prompt(task: dict[str, Any], produced_fields: set[str]) -> str:
    stem = Path(task["deliverable"]).stem
    fields = "; ".join(sorted(produced_fields | {"honest_verdict"}))
    return (
        "CONTEXT:\nA deterministic fixture.\n\n"
        "EXISTING CODE TO READ FIRST:\n- CODEX.md\n\n"
        "TASK:\nValidate the fixture.\n\n"
        "CONCRETE STEPS:\n1. Validate it.\n\n"
        f"REQUIRED ARTIFACT FIELDS: {fields}\n\n"
        f"Run command: cd {{project_root}} && .venv/bin/python scripts/experiments/{stem}.py --date {{date}}\n"
        "Do NOT push. Do NOT modify scripts/research_conductor.py.\n"
    )


def _valid_inputs() -> tuple[str, dict[str, Any], dict[str, Any], dict[str, Any]]:
    document = mod.parse_design(DESIGN_TEXT)
    producer_fields: dict[str, set[str]] = {row["task_id"]: set() for row in document["tasks"]}
    for row in document["tasks"]:
        for gate in row["gates"]:
            producer_fields[gate["upstream"]].add(gate["artifact_field"])
    tasks = []
    for row in document["tasks"]:
        task = {
            "id": row["task_id"],
            "title": row["title"],
            "milestone": mod.V603_MILESTONE,
            "deliverable": row["deliverable"],
            "gated_on": deepcopy(row["gates"]),
            "prior_failures": [],
        }
        task["prompt"] = _prompt(task, producer_fields[task["id"]])
        tasks.append(task)
    roadmap = {
        "milestone": mod.V603_MILESTONE,
        "milestone_title": "fixture",
        "milestone_doc": mod.DESIGN_PATH.as_posix(),
        "tasks": tasks,
    }
    return DESIGN_TEXT, roadmap, {"retired": []}, {}


def _evaluate(
    roadmap: dict[str, Any],
    *,
    design: str = DESIGN_TEXT,
    exclusion: dict[str, Any] | None = None,
    priors: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return mod.evaluate_contract(
        design,
        roadmap,
        exclusion or {"retired": []},
        priors or {},
        SCHEMA_TEXT,
        GATE_TEXT,
    )


def test_req_report_6885_spec_owns_fields_and_scenarios() -> None:
    """REQ-REPORT-6885 declares the complete contract before implementation."""

    text = (ROOT / mod.SPEC_PATH).read_text(encoding="utf-8")
    section = text[text.index("### REQ-REPORT-6885") :]
    for scenario in (
        "SCENARIO-REPORT-6885-DOCUMENT-YAML",
        "SCENARIO-REPORT-6885-IDENTITY",
        "SCENARIO-REPORT-6885-PROMPT-END",
        "SCENARIO-REPORT-6885-GATE",
        "SCENARIO-REPORT-6885-PRIOR",
        "SCENARIO-REPORT-6885-RETIRED-ID",
        "SCENARIO-REPORT-6885-ROOT-FANOUT",
        "SCENARIO-REPORT-6885-UNGATED-TAIL",
    ):
        assert scenario in section
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert f"`{field}`" in section or field == "field_principles"


def test_req_report_6885_valid_contract_passes_all_checks() -> None:
    """REQ-REPORT-6885 accepts the exact 13-task isolated branch graph."""

    _, roadmap, exclusion, priors = _valid_inputs()
    result = _evaluate(roadmap, exclusion=exclusion, priors=priors)

    assert result["passed"] is True
    assert len(result["document_task_rows"]) == 13
    assert len(result["yaml_task_rows"]) == 13
    assert all(row["passed"] for row in result["document_yaml_parity_rows"])
    assert all(row["passed"] for row in result["gate_contract_rows"])
    assert all(row["passed"] for row in result["prompt_ending_rows"])
    assert all(row["passed"] for row in result["branch_root_rows"])
    assert all(row["passed"] for row in result["ungated_tail_rows"])
    assert result["warnings"] == []
    assert result["hard_failures"] == []


@pytest.mark.parametrize("mismatch", ["count", "order", "id", "deliverable", "milestone"])
def test_scenario_report_6885_document_yaml_identity_mismatches(mismatch: str) -> None:
    """SCENARIO-REPORT-6885-DOCUMENT-YAML and -IDENTITY fail each exact field."""

    _, roadmap, _, _ = _valid_inputs()
    tasks = roadmap["tasks"]
    if mismatch == "count":
        tasks.pop()
    elif mismatch == "order":
        tasks[2], tasks[3] = tasks[3], tasks[2]
    elif mismatch == "id":
        tasks[4]["id"] = "exp6889-wrong-id"
    elif mismatch == "deliverable":
        tasks[4]["deliverable"] = "results/experiment_6889_wrong.json"
    else:
        tasks[4]["milestone"] = "2026.09.602"

    result = _evaluate(roadmap)

    assert result["passed"] is False
    assert any(not row["passed"] for row in result["document_yaml_parity_rows"])


@pytest.mark.parametrize("mismatch", ["missing_section", "run_command", "final_sentence"])
def test_scenario_report_6885_prompt_endings_are_exact(mismatch: str) -> None:
    """SCENARIO-REPORT-6885-PROMPT-END rejects missing sections and changed endings."""

    _, roadmap, _, _ = _valid_inputs()
    prompt = roadmap["tasks"][0]["prompt"]
    if mismatch == "missing_section":
        prompt = prompt.replace("TASK:\n", "WORK:\n")
    elif mismatch == "run_command":
        prompt = prompt.replace("experiment_6885_", "experiment_9999_", 1)
    else:
        prompt = prompt.replace("Do NOT push.", "Do not push.")
    roadmap["tasks"][0]["prompt"] = prompt

    result = _evaluate(roadmap)
    row = result["prompt_ending_rows"][0]

    assert row["passed"] is False
    assert result["passed"] is False


@pytest.mark.parametrize("mismatch", ["field", "upstream", "operator", "value"])
def test_scenario_report_6885_gate_mismatches_fail_closed(mismatch: str) -> None:
    """SCENARIO-REPORT-6885-GATE checks fields, upstreams, operators, and values."""

    _, roadmap, _, _ = _valid_inputs()
    gate = roadmap["tasks"][2]["gated_on"][0]
    if mismatch == "field":
        gate["artifact_field"] = "relation_fixture_ready_typo"
    elif mismatch == "upstream":
        gate["upstream"] = "exp6999-missing-upstream"
    elif mismatch == "operator":
        gate["op"] = "==>"
    else:
        gate["value"] = 0

    result = _evaluate(roadmap)
    row = next(row for row in result["gate_contract_rows"] if row["downstream_number"] == 6887)

    assert row["passed"] is False
    assert result["passed"] is False


@pytest.mark.parametrize(
    ("missing_field", "replacement"),
    [
        ("experiment_id", None),
        ("verdict", None),
        ("addressed_by", ""),
        ("retire_if_same_verdict", False),
    ],
)
def test_scenario_report_6885_prior_failure_subfields(missing_field: str, replacement: Any) -> None:
    """SCENARIO-REPORT-6885-PRIOR checks all four required subfields."""

    _, roadmap, _, _ = _valid_inputs()
    prior_id = "exp6874-v602-evidence-substrate-manifest-contract"
    prior = {
        "experiment_id": prior_id,
        "verdict": "complete_prior",
        "addressed_by": "A changed branch shape addresses the failure.",
        "retire_if_same_verdict": True,
    }
    if replacement is None:
        prior.pop(missing_field)
    else:
        prior[missing_field] = replacement
    roadmap["tasks"][0]["prior_failures"] = [prior]

    result = _evaluate(
        roadmap,
        priors={prior_id: {"honest_verdict": "complete_prior"}},
    )
    row = result["prior_failure_contract_rows"][0]

    assert row["passed"] is False
    assert missing_field in row["failed_subfields"]
    assert result["passed"] is False


def test_scenario_report_6885_prior_verdict_uses_primary_artifact() -> None:
    """SCENARIO-REPORT-6885-PRIOR rejects a changed primary verdict."""

    _, roadmap, _, _ = _valid_inputs()
    prior_id = "exp6874-v602-evidence-substrate-manifest-contract"
    roadmap["tasks"][0]["prior_failures"] = [
        {
            "experiment_id": prior_id,
            "verdict": "wrong_verdict",
            "addressed_by": "The execution graph changed.",
            "retire_if_same_verdict": True,
        }
    ]

    result = _evaluate(roadmap, priors={prior_id: {"honest_verdict": "complete_prior"}})

    assert result["prior_failure_contract_rows"][0]["failed_subfields"] == ["verdict"]


def test_scenario_report_6885_retired_ids_and_upstreams_are_separate() -> None:
    """SCENARIO-REPORT-6885-RETIRED-ID preserves hard failures and allowed warnings."""

    _, roadmap, _, _ = _valid_inputs()
    prior_id = "exp5909-sota-constraint-synthesis-ab"
    roadmap["tasks"][0]["prior_failures"] = [
        {
            "experiment_id": prior_id,
            "verdict": "complete_prior",
            "addressed_by": "The new audit does not rerun the retired method.",
            "retire_if_same_verdict": True,
        }
    ]
    exclusion = {
        "retired_experiments": [
            {"experiment_id": 6886},
            {"experiment_ids": ["exp5909"]},
        ]
    }

    result = _evaluate(
        roadmap,
        exclusion=exclusion,
        priors={prior_id: {"honest_verdict": "complete_prior"}},
    )

    retired_task = next(row for row in result["retired_id_rows"] if row["number"] == 6886)
    retired_edge = next(
        row for row in result["retired_upstream_rows"] if row["upstream_number"] == 6886
    )
    assert retired_task["passed"] is False
    assert retired_edge["passed"] is False
    assert result["warnings"][0]["warning"] == "retired_prior_failure_reference_allowed"
    assert result["hard_failures"]


def test_req_report_6885_unretirement_entries_are_honored() -> None:
    """REQ-REPORT-6885 applies append-only un-retirement corrections."""

    _, roadmap, _, _ = _valid_inputs()
    exclusion = {
        "retired_extras": [
            {
                "experiment_ids": ["exp6886"],
                "un_retired_experiment_ids": ["exp6886"],
            }
        ]
    }

    result = _evaluate(roadmap, exclusion=exclusion)

    assert result["passed"] is True
    assert all(row["passed"] for row in result["retired_id_rows"])


def test_scenario_report_6885_root_fanout_is_forbidden() -> None:
    """SCENARIO-REPORT-6885-ROOT-FANOUT keeps advisory readiness out of science."""

    _, roadmap, _, _ = _valid_inputs()
    roadmap["tasks"][7]["gated_on"] = [
        {
            "upstream": roadmap["tasks"][0]["id"],
            "artifact_field": "v603_manifest_contract_ready_score",
            "op": "==",
            "value": 1,
        }
    ]

    result = _evaluate(roadmap)
    advisory = next(
        row for row in result["branch_root_rows"] if row["kind"] == "advisory_manifest_fanout"
    )
    arc_root = next(row for row in result["branch_root_rows"] if row.get("number") == 6892)

    assert advisory["passed"] is False
    assert arc_root["passed"] is False
    assert result["passed"] is False


@pytest.mark.parametrize("mismatch", ["gated_audit", "missing_capstone"])
def test_scenario_report_6885_ungated_terminal_tail(mismatch: str) -> None:
    """SCENARIO-REPORT-6885-UNGATED-TAIL requires present ungated terminal tasks."""

    _, roadmap, _, _ = _valid_inputs()
    if mismatch == "gated_audit":
        roadmap["tasks"][11]["gated_on"] = [
            {
                "upstream": roadmap["tasks"][10]["id"],
                "artifact_field": "tool_loop_promotion_ready_score",
                "op": "==",
                "value": 1,
            }
        ]
    else:
        roadmap["tasks"].pop()

    result = _evaluate(roadmap)

    assert any(not row["passed"] for row in result["ungated_tail_rows"])
    assert result["passed"] is False


def test_req_report_6885_active_primary_files_fail_closed() -> None:
    """REQ-REPORT-6885 preserves the real active four-versus-thirteen mismatch."""

    artifact = mod.build_artifact(ROOT, "20260902")

    assert artifact["task_count"] == 13
    assert artifact["experiment_range"] == [6885, 6897]
    assert len(artifact["document_task_rows"]) == 13
    assert len(artifact["yaml_task_rows"]) == 4
    assert artifact["v603_manifest_contract_ready_score"] == 0
    assert artifact["verdict_class"] == "blocked"
    assert artifact["honest_verdict"] == (
        "complete_blocked_v603_executable_manifest_branch_contract"
    )
    assert artifact["gate_check_summary"]["failed_check"] == "document_yaml_task_contract"
    assert mod.validate_artifact(artifact) == []


def test_req_report_6885_missing_preconditions_still_emit_complete_block(tmp_path: Path) -> None:
    """REQ-REPORT-6885 writes diagnostic expected and observed values when sources fail."""

    artifact = mod.build_artifact(tmp_path, "20260902")

    assert artifact["status"] == "complete_blocked"
    assert artifact["preconditions_checked"]["passed"] is False
    assert artifact["gate_check_summary"]["failed_check"] == "preconditions"
    assert artifact["gate_check_summary"]["expected"] == "all required sources readable"
    assert artifact["gate_check_summary"]["observed"]
    assert mod.validate_artifact(artifact) == []


def test_req_report_6885_validation_and_checksum_failures_are_explicit() -> None:
    """REQ-REPORT-6885 validates required fields, principles, readiness, and checksum."""

    artifact = mod.build_artifact(ROOT, "20260902")
    broken = deepcopy(artifact)
    broken.pop("rows")
    broken["field_principles"].pop("task_count")
    broken["inference_substrate"] = "live_llm_inference"
    broken["verifier_is_oracle"] = True
    broken["verdict_class"] = "unknown"
    broken["honest_verdict"] = "blocked"
    broken["v603_manifest_contract_ready_score"] = 1
    broken["reproducibility_checksum"] = "sha256:bad"

    errors = mod.validate_artifact(broken)

    assert "missing_required_fields:rows" in errors
    assert "field_principles_missing" in errors
    assert "invalid_inference_substrate" in errors
    assert "verifier_is_oracle_must_be_false" in errors
    assert "invalid_verdict_class" in errors
    assert "honest_verdict_not_terminal" in errors
    assert "readiness_recomputation_mismatch" in errors
    assert "reproducibility_checksum_mismatch" in errors


def test_req_report_6885_writer_and_cli_date_validation(tmp_path: Path) -> None:
    """REQ-REPORT-6885 writes only the requested path and rejects ambiguous dates."""

    payload = {"z": 1, "a": 2}
    output = tmp_path / "receipt.json"
    mod.write_json_atomic(output, payload)
    assert json.loads(output.read_text(encoding="utf-8")) == payload
    assert mod.main(["--date", "bad", "--output", str(output)]) == 2


def test_req_report_6885_cli_surfaces_internal_validation_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-REPORT-6885 returns failure after preserving an invalid receipt."""

    output = tmp_path / "invalid.json"
    monkeypatch.setattr(mod, "build_artifact", lambda _root, _date: {"value": 1})
    monkeypatch.setattr(mod, "validate_artifact", lambda _artifact: ["invalid"])

    assert mod.main(["--date", "20260902", "--output", str(output)]) == 1
    assert json.loads(output.read_text(encoding="utf-8")) == {"value": 1}


def test_req_report_6885_cli_writes_a_valid_blocked_receipt(tmp_path: Path) -> None:
    """REQ-REPORT-6885 CLI persists the current honest blocked contract."""

    output = tmp_path / "receipt.json"
    assert mod.main(["--date", "20260902", "--output", str(output)]) == 0
    artifact = json.loads(output.read_text(encoding="utf-8"))
    assert artifact["v603_manifest_contract_ready_score"] == 0
    assert mod.validate_artifact(artifact) == []


@pytest.mark.parametrize(
    "document",
    [
        "not a V603 design",
        "**Milestone:** `2026.09.603`\n### Exp6885: Missing deliverable\nBody\n",
    ],
)
def test_req_report_6885_design_parse_errors_are_explicit(document: str) -> None:
    """REQ-REPORT-6885 refuses malformed primary design records."""

    with pytest.raises(ValueError):
        mod.parse_design(document)


def test_req_report_6885_design_ignores_out_of_range_rows_and_gates() -> None:
    """REQ-REPORT-6885 keeps unrelated design sections outside the V603 inventory."""

    extra = (
        "### Exp1: Unrelated task\n"
        "**Deliverable:** `results/experiment_1_unrelated_task.json`\n\n"
        "| Exp1 | `exp2.ready` | `contains 1` |\n\n"
    )
    document = mod.parse_design(DESIGN_TEXT + "\n" + extra)

    assert len(document["tasks"]) == 13
    assert mod._parse_condition("contains 1") == ("contains 1", None)


@pytest.mark.parametrize(
    "roadmap",
    [
        [],
        {"tasks": "bad"},
        {"tasks": ["bad"]},
        {"tasks": [{"gated_on": "bad"}]},
        {"tasks": [{"gated_on": ["bad"]}]},
    ],
)
def test_req_report_6885_roadmap_parse_errors_are_explicit(roadmap: Any) -> None:
    """REQ-REPORT-6885 refuses malformed executable task shapes."""

    with pytest.raises(ValueError):
        mod.parse_roadmap(roadmap)


def test_req_report_6885_operator_extractors_fail_closed() -> None:
    """REQ-REPORT-6885 reads current schema and conductor operators from code."""

    assert {"==", ">=", "exists"}.issubset(mod.schema_gate_operators(SCHEMA_TEXT))
    assert {"==", ">=", "not_contains"}.issubset(mod.conductor_gate_operators(GATE_TEXT))
    with pytest.raises(ValueError):
        mod.schema_gate_operators("GateOp = str")
    with pytest.raises(ValueError):
        mod.conductor_gate_operators("def other(): pass")
    with pytest.raises(ValueError):
        mod.conductor_gate_operators("def _eval_op(): return True")


def test_req_report_6885_malformed_retirement_sections_fail_quiet() -> None:
    """REQ-REPORT-6885 ignores non-row exclusion data without inventing IDs."""

    manifest = {"retired": "bad", "retired_experiments": ["bad"]}
    assert mod.retired_experiment_ids(manifest) == set()


def test_req_report_6885_checksum_ignores_only_runtime_fields() -> None:
    """REQ-REPORT-6885 checksum stays stable across elapsed-time changes."""

    first = {"duration_s": 1.0, "reproducibility_checksum": "", "value": 2}
    second = {"duration_s": 9.0, "reproducibility_checksum": "bad", "value": 2}
    assert mod.reproducibility_checksum(first) == mod.reproducibility_checksum(second)


def test_req_report_6885_yaml_loader_rejects_non_mapping(tmp_path: Path) -> None:
    """REQ-REPORT-6885 source loading reports malformed YAML without raising."""

    path = tmp_path / "bad.yaml"
    path.write_text("- item\n", encoding="utf-8")
    value, error = mod.read_yaml_mapping(path)
    assert value is None
    assert error == "yaml_mapping_required"
    path.write_text("[", encoding="utf-8")
    value, error = mod.read_yaml_mapping(path)
    assert value is None
    assert error.startswith("invalid_yaml:")
    assert mod.read_yaml_mapping(tmp_path / "missing.yaml") == (None, "missing")


def test_req_report_6885_primary_prior_loader_preserves_missing_and_invalid(tmp_path: Path) -> None:
    """REQ-REPORT-6885 prior artifacts stay keyed to their exact declared IDs."""

    tasks = [
        {
            "prior_failures": [
                {"experiment_id": "exp1-good"},
                {"experiment_id": "exp2-missing"},
                {"experiment_id": "exp3-bad"},
                {"experiment_id": "exp4-invalid"},
                {},
            ]
        }
    ]
    results = tmp_path / "results"
    results.mkdir()
    (results / "experiment_1_good.json").write_text(
        json.dumps({"honest_verdict": "complete_good"}), encoding="utf-8"
    )
    (results / "experiment_3_bad.json").write_text("[]", encoding="utf-8")
    (results / "experiment_4_invalid.json").write_text("{", encoding="utf-8")

    artifacts, rows = mod.load_prior_artifacts(tmp_path, tasks)

    assert artifacts["exp1-good"]["honest_verdict"] == "complete_good"
    assert [row["error"] for row in rows] == [
        None,
        "missing",
        "json_object_required",
        "invalid_json:JSONDecodeError",
        "invalid_id",
    ]


def test_req_report_6885_empty_required_source_is_a_precondition_failure(tmp_path: Path) -> None:
    """REQ-REPORT-6885 distinguishes an empty source from a missing source."""

    path = tmp_path / mod.DESIGN_PATH
    path.parent.mkdir(parents=True)
    path.write_text("", encoding="utf-8")

    artifact = mod.build_artifact(tmp_path, "20260902")

    observed = artifact["preconditions_checked"]["observed"]
    assert {"source": "v603_design", "error": "empty"} in observed


def test_req_report_6885_ready_terminal_shape_is_recomputed() -> None:
    """REQ-REPORT-6885 rejects a positive reduction with a blocked terminal shape."""

    artifact = mod.build_artifact(ROOT, "20260902")
    for check in artifact["gate_check_summary"]["checks"]:
        check["passed"] = True
    artifact["v603_manifest_contract_ready_score"] = 1
    artifact["reproducibility_checksum"] = mod.reproducibility_checksum(artifact)

    assert "ready_terminal_shape_mismatch" in mod.validate_artifact(artifact)


def test_req_report_6885_yaml_round_trip_fixture_is_structural() -> None:
    """REQ-REPORT-6885 synthetic fixtures use the same YAML types as the active file."""

    _, roadmap, _, _ = _valid_inputs()
    loaded = yaml.safe_load(yaml.safe_dump(roadmap, sort_keys=False))
    assert _evaluate(loaded)["passed"] is True
