"""Focused tests for the V588 handoff binding contract audit.

Spec refs: REQ-REPORT-6742, REQ-HARNESS-008,
SCENARIO-REPORT-6742-CONTRACT, SCENARIO-REPORT-6742-BLOCKED, and
SCENARIO-REPORT-6742-ATOMIC.
"""

from __future__ import annotations

from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import types

import pytest
import yaml


REPO = Path(__file__).resolve().parents[2]
MODULE_PATH = REPO / "python/carnot/experiment_6742_v588_handoff_contract_audit.py"
SPEC = importlib.util.spec_from_file_location("exp6742_under_test", MODULE_PATH)
assert SPEC and SPEC.loader
exp = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = exp
SPEC.loader.exec_module(exp)


def _active_manifest() -> dict[str, object]:
    return yaml.safe_load((REPO / exp.ACTIVE_ROADMAP_PATH).read_text(encoding="utf-8"))


def _expanded_manifest() -> dict[str, object]:
    manifest = _active_manifest()
    for task in manifest["tasks"]:
        task["prompt"] = (
            task["prompt"]
            .replace("{project_root}", exp.PROJECT_ROOT_LITERAL)
            .replace("{date}", exp.PLANNING_DATE)
        )
    return manifest


def _audit(candidate: dict[str, object], retired_ids: set[str] | None = None) -> dict[str, object]:
    design = exp.parse_design_contract((REPO / exp.DESIGN_PATH).read_text(encoding="utf-8"))
    receipts = exp.collect_source_receipts(REPO, design)
    return exp.audit_contract(
        REPO,
        design,
        candidate,
        receipts,
        retired_ids=retired_ids or set(),
        validator_rows=[],
    )


def test_req_report_6742_spec_and_design_own_the_contract() -> None:
    """REQ-REPORT-6742: the reporting spec owns the durable V588 contract."""

    reporting = (REPO / exp.SPEC_PATH).read_text(encoding="utf-8")
    harness = (REPO / exp.HARNESS_SPEC_PATH).read_text(encoding="utf-8")
    section = reporting.split("REQ-REPORT-6742", 1)[1]
    for field in exp.REQUIRED_ARTIFACT_FIELDS:
        assert field in section
    for marker in (
        "SCENARIO-REPORT-6742-CONTRACT",
        "SCENARIO-REPORT-6742-BLOCKED",
        "SCENARIO-REPORT-6742-ATOMIC",
        exp.INFERENCE_SUBSTRATE,
        exp.RESULT_PATH.as_posix(),
    ):
        assert marker in section
    assert "REQ-HARNESS-008" in harness

    design = exp.parse_design_contract((REPO / exp.DESIGN_PATH).read_text(encoding="utf-8"))
    assert design["milestone"] == exp.MILESTONE
    assert [row["task_id"] for row in design["tasks"]] == list(exp.EXPECTED_TASK_IDS)
    assert sorted(design["phases"]) == ["Phase 1", "Phase 2", "Phase 3", "Phase 4"]
    assert design["tasks"][-1]["task_id"] == exp.CAPSTONE_TASK_ID


def test_scenario_report_6742_current_manifest_blocks_on_prompt_literals() -> None:
    """SCENARIO-REPORT-6742-BLOCKED: placeholders are not normalized."""

    artifact = exp.build_artifact(REPO, duration_s=1.0)
    assert artifact["task_count"] == 13
    assert len(artifact["rows"]) == 20
    assert artifact["rows"] == artifact["binding_contract_rows"]
    assert artifact["handoff_contract_preserved"] is False
    assert artifact["science_branches_independent_of_handoff_audit"] is True
    assert artifact["verdict_class"] == "blocked"
    assert artifact["honest_verdict"].startswith("complete_blocked_handoff_contract:")
    assert exp.validate_artifact(artifact) == []

    failures = {(row["check"], row["observed_value"]) for row in artifact["gate_check_summary"]}
    assert ("prompt.project_root_literal", "{project_root}") in failures
    assert ("prompt.planning_date_literal", "{date}") in failures

    validators = {row["validator"]: row for row in artifact["validator_rows"]}
    assert validators["roadmap_schema"]["passed"] is True
    assert validators["gate_contract"]["passed"] is True
    assert validators["prompt_contract"]["passed"] is False
    assert validators["exclusion_manifest_lint"]["passed"] is True
    assert validators["audit_roadmap_gates_legacy"]["compatibility_accepted"] is True
    assert validators["audit_roadmap_gates_legacy"]["model_only_findings"] is True


def test_scenario_report_6742_expanded_manifest_passes_contract() -> None:
    """SCENARIO-REPORT-6742-CONTRACT: expanded V588 prompts preserve rows."""

    artifact = exp.build_artifact(
        REPO,
        duration_s=2.0,
        active_payload=_expanded_manifest(),
        run_external_validators=False,
    )
    assert artifact["handoff_contract_preserved"] is True
    assert artifact["verdict_class"] == "null"
    assert artifact["honest_verdict"].startswith("complete_null:")
    assert artifact["gate_check_summary"] == []
    assert len(artifact["binding_contract_rows"]) == 20
    assert all(row["operationally_preserved"] for row in artifact["binding_contract_rows"])
    assert exp.validate_artifact(artifact) == []


def test_scenario_report_6742_fail_closed_mutations_are_named() -> None:
    """SCENARIO-REPORT-6742-BLOCKED: exact handoff defects remain visible."""

    base = _expanded_manifest()
    cases: list[tuple[dict[str, object], set[str], set[str]]] = []

    duplicate = deepcopy(base)
    duplicate["tasks"][1]["deliverable"] = duplicate["tasks"][0]["deliverable"]
    cases.append((duplicate, set(), {"manifest.deliverables_unique"}))

    renamed_gate = deepcopy(base)
    renamed_gate["tasks"][3]["gated_on"][0]["artifact_field"] = "renamed_ready"
    cases.append((renamed_gate, set(), {"gate.producer_field", "gate.matches_design_gate"}))

    incomplete_prior = deepcopy(base)
    incomplete_prior["tasks"][4]["prior_failures"][0]["retire_if_same_verdict"] = False
    cases.append((incomplete_prior, set(), {"prior.failure_contract"}))

    no_rows = deepcopy(base)
    no_rows["tasks"][12]["per_unit_rows"] = False
    cases.append((no_rows, set(), {"task.per_unit_rows"}))

    capstone_gate = deepcopy(base)
    capstone_gate["tasks"][12]["gated_on"] = [
        {
            "upstream": "exp6742-v588-handoff-contract-audit",
            "artifact_field": "handoff_contract_preserved",
            "op": "==",
            "value": True,
        }
    ]
    cases.append((capstone_gate, set(), {"task.capstone_ungated"}))

    missing_model = deepcopy(base)
    for model in exp.MANDATED_MODELS:
        missing_model["tasks"][3]["prompt"] = missing_model["tasks"][3]["prompt"].replace(
            model, "wrong/model-GGUF"
        )
    cases.append((missing_model, set(), {"model.policy"}))

    retired_reuse = deepcopy(base)
    cases.append((retired_reuse, {"exp6744-hardness-controlled-certificate-stream"}, {"task.task_id_not_retired"}))

    for candidate, retired_ids, expected_checks in cases:
        audit = _audit(candidate, retired_ids)
        observed = {row["check"] for row in audit["failures"]}
        assert expected_checks <= observed
        assert audit["passed"] is False


def test_scenario_report_6742_precondition_block_and_atomic_cli(tmp_path: Path) -> None:
    """SCENARIO-REPORT-6742-ATOMIC: blocked input writes still validate."""

    artifact = exp.build_artifact(tmp_path, duration_s=0.5)
    assert artifact["honest_verdict"].startswith("complete_blocked_handoff_input:")
    assert artifact["verdict_class"] == "blocked"
    assert artifact["rows"] == []
    assert any(row["check"] == "precondition.local_file" for row in artifact["gate_check_summary"])
    assert exp.validate_artifact(artifact) == []

    target = tmp_path / "nested" / "artifact.json"
    exp.write_json_atomic(target, artifact)
    assert json.loads(target.read_text(encoding="utf-8")) == artifact
    assert not list(target.parent.glob("*.tmp"))

    real_replace = exp.os.replace
    try:
        exp.os.replace = lambda *_args: (_ for _ in ()).throw(OSError("replace failed"))
        with pytest.raises(OSError, match="replace failed"):
            exp.write_json_atomic(tmp_path / "failed.json", artifact)
    finally:
        exp.os.replace = real_replace
    assert not list(tmp_path.glob("*.tmp"))

    broken = deepcopy(artifact)
    broken["handoff_contract_preserved"] = True
    assert "reproducibility_checksum_mismatch" in exp.validate_artifact(broken)

    bad_principles = deepcopy(artifact)
    bad_principles["field_principles"] = {}
    bad_principles["reproducibility_checksum"] = exp.reproducibility_checksum(bad_principles)
    assert "field_principles_missing" in exp.validate_artifact(bad_principles)

    bad_block = deepcopy(artifact)
    bad_block["gate_check_summary"] = []
    bad_block["reproducibility_checksum"] = exp.reproducibility_checksum(bad_block)
    assert "blocked_gate_summary_missing" in exp.validate_artifact(bad_block)

    assert exp.main(["--repo-root", str(REPO), "--output", str(tmp_path / "cli.json")]) == 0
    assert exp.main(["--validate", "--output", str(tmp_path / "cli.json")]) == 0
    (tmp_path / "cli.json").write_text(json.dumps({"bad": "payload"}), encoding="utf-8")
    assert exp.main(["--validate", "--output", str(tmp_path / "cli.json")]) == 1

    wrapper_path = REPO / "scripts/experiments/experiment_6742_v588_handoff_contract_audit.py"
    saved_path = list(sys.path)
    saved_carnot = sys.modules.get("carnot")
    saved_module = sys.modules.get("carnot.experiment_6742_v588_handoff_contract_audit")
    fake_carnot = types.ModuleType("carnot")
    fake_carnot.__path__ = []
    fake_module = types.ModuleType("carnot.experiment_6742_v588_handoff_contract_audit")
    fake_module.main = exp.main
    sys.modules["carnot"] = fake_carnot
    sys.modules["carnot.experiment_6742_v588_handoff_contract_audit"] = fake_module
    for path in (REPO, REPO / "python"):
        while str(path) in sys.path:
            sys.path.remove(str(path))
    spec = importlib.util.spec_from_file_location("exp6742_wrapper", wrapper_path)
    assert spec and spec.loader
    wrapper = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(wrapper)
    finally:
        sys.path = saved_path
        if saved_carnot is None:
            sys.modules.pop("carnot", None)
        else:
            sys.modules["carnot"] = saved_carnot
        if saved_module is None:
            sys.modules.pop("carnot.experiment_6742_v588_handoff_contract_audit", None)
        else:
            sys.modules["carnot.experiment_6742_v588_handoff_contract_audit"] = saved_module
    assert wrapper.main(["--repo-root", str(REPO), "--output", str(tmp_path / "wrapper.json")]) == 0


def test_scenario_report_6742_defensive_helper_edges(tmp_path: Path) -> None:
    """SCENARIO-REPORT-6742-BLOCKED: parser edges fail closed."""

    assert exp.extract_required_artifact_fields("no field block") == []
    assert exp.extract_model_specs("no model block") == []
    assert exp._slug("Unmapped A/B Title!") == "unmapped-a-b-title"
    with pytest.raises(ValueError, match="invalid task id"):
        exp._task_number("not-an-exp")
    assert exp._as_list(None) == []
    assert exp._as_list("exp1") == ["exp1"]
    assert exp._as_list(["exp1", 2]) == ["exp1", "2"]
    assert exp._observed_literal("uses {date}", "20260829", "{date}") == "{date}"
    assert exp._observed_literal("literal 20260829", "20260829", "{date}") == "20260829"
    assert exp._observed_literal("missing", "20260829", "{date}") == "missing"
    assert exp._source_block("no matching source", "2608.27311") == ""
    assert exp._source_block("2608.27311 only source", "2608.27311") == "2608.27311 only source"
    assert exp._block_after("no marker here", "0. PRECONDITIONS:", ("1.",)) == ""
    assert exp._first_sentence_with("No matching words.", ("authority",)) == ""
    assert exp._binding_task_fields(
        {"agent_type": "codex", "model": "gpt-5.6-sol", "prompt": "plain prompt"}
    )["execution_consequence"]["value"] == "ungated task executes under its own preconditions"

    with pytest.raises(ValueError, match="task deliverable missing"):
        exp._next_deliverable(["### Exp 6742: Missing", "### Exp 6743: Next"], 0)
    with pytest.raises(ValueError, match="V588 planner refresh section missing"):
        exp._v588_section("## V587 Planner Refresh - 2026-08-29")
    with pytest.raises(ValueError, match="V588 design milestone missing"):
        exp.parse_design_contract("# no milestone")

    (tmp_path / "ops").mkdir()
    (tmp_path / exp.EXCLUSION_PATH).write_text(
        "retired:\n- bad\n- experiment_id: 6742\nretired_extras:\n- id: exp6743-old\n",
        encoding="utf-8",
    )
    assert exp.retired_task_ids(tmp_path) == {
        "exp6742-v588-handoff-contract-audit",
        "exp6743-old",
    }

    missing, missing_row = exp._read_yaml_mapping(tmp_path / "absent.yaml")
    assert missing is None
    assert missing_row["reason"] == "file_missing"
    invalid_yaml = tmp_path / "invalid.yaml"
    invalid_yaml.write_text("a: [", encoding="utf-8")
    invalid, invalid_row = exp._read_yaml_mapping(invalid_yaml)
    assert invalid is None
    assert invalid_row["reason"] == "yaml_error"
    list_yaml = tmp_path / "list.yaml"
    list_yaml.write_text("- one\n", encoding="utf-8")
    top_level, top_level_row = exp._read_yaml_mapping(list_yaml)
    assert top_level is None
    assert top_level_row["reason"] == "top_level_not_mapping"

    real_spec_from_file = exp.importlib.util.spec_from_file_location
    try:
        exp.importlib.util.spec_from_file_location = lambda *_args, **_kwargs: None
        with pytest.raises(ImportError, match="cannot load"):
            exp._load_schema_module(REPO)
    finally:
        exp.importlib.util.spec_from_file_location = real_spec_from_file

    gate_rows, gate_failures = exp.build_gate_binding_rows(
        {"tasks": [{"id": "exp6745-x", "gated_on": ["bad"]}]}
    )
    assert gate_rows == []
    assert gate_failures == []
    prior_rows, prior_failures = exp.build_prior_failure_rows({"tasks": ["bad"]})
    assert prior_rows == []
    assert prior_failures == []

    bad_design = {
        "milestone": "wrong",
        "tasks": [
            {"task_id": "exp1-a", "deliverable": "results/a.json"},
            {"task_id": "exp1-a", "deliverable": "results/a.json"},
        ],
        "phases": ["Phase 1"],
    }
    assert {
        "design.milestone",
        "design.task_count",
        "design.task_ids_unique",
        "design.deliverables_unique",
        "design.phase_count",
    } == {row["check"] for row in exp._design_failures(bad_design)}

    handoff_gated = _expanded_manifest()
    handoff_gated["tasks"][2]["gated_on"] = [
        {
            "upstream": "exp6742-v588-handoff-contract-audit",
            "artifact_field": "handoff_contract_preserved",
            "op": "==",
            "value": True,
        }
    ]
    assert exp.science_branches_independent(handoff_gated) is False

    real_run = exp._run_command
    try:
        exp._run_command = lambda _root, args: subprocess.CompletedProcess(
            args,
            0 if args[0] == exp.EXCLUSION_LINT_PATH.as_posix() else 1,
            "clean" if args[0] == exp.EXCLUSION_LINT_PATH.as_posix() else "not json",
            "legacy stderr",
        )
        validators = exp.build_validator_rows(REPO, _expanded_manifest(), [], [], True)
    finally:
        exp._run_command = real_run
    legacy = {row["validator"]: row for row in validators}["audit_roadmap_gates_legacy"]
    assert legacy["model_only_findings"] is False
    assert legacy["compatibility_accepted"] is False

    bad_rows = exp.build_artifact(tmp_path, duration_s=0.5)
    bad_rows["rows"] = [{"different": True}]
    bad_rows["reproducibility_checksum"] = exp.reproducibility_checksum(bad_rows)
    assert "rows_binding_contract_rows_mismatch" in exp.validate_artifact(bad_rows)

    assert exp.main(["--validate", "--output", str(tmp_path / "missing.json")]) == 1
    real_validate = exp.validate_artifact
    try:
        exp.validate_artifact = lambda _payload: ["forced-invalid"]
        assert exp.main(["--repo-root", str(REPO), "--output", str(tmp_path / "invalid.json")]) == 1
    finally:
        exp.validate_artifact = real_validate
