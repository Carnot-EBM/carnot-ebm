"""Focused tests for the V588 branch disposition capstone.

Spec refs: REQ-REPORT-6754, SCENARIO-REPORT-6754-MISSING-BRANCH,
SCENARIO-REPORT-6754-ROW-RECOMPUTATION,
SCENARIO-REPORT-6754-VERDICT-PROPAGATION, and
SCENARIO-REPORT-6754-NO-POOLED-CLAIM.
"""

from __future__ import annotations

from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys
import types

import pytest
import yaml


REPO = Path(__file__).resolve().parents[2]
MODULE_PATH = REPO / "python/carnot/experiment_6754_v588_branch_disposition.py"
SPEC = importlib.util.spec_from_file_location("exp6754_under_test", MODULE_PATH)
assert SPEC and SPEC.loader
exp = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = exp
SPEC.loader.exec_module(exp)


@pytest.fixture(scope="module")
def current_inputs() -> tuple[list[dict], dict[str, dict]]:
    planned = exp.load_planned_tasks(REPO)
    sources = exp.load_source_artifacts(REPO, planned)
    return planned, sources


def _task_rows(
    current_inputs: tuple[list[dict], dict[str, dict]],
) -> list[dict]:
    planned, sources = current_inputs
    return exp.build_task_rows(REPO, planned, sources)


def test_req_report_6754_spec_precedes_implementation() -> None:
    """REQ-REPORT-6754: the reporting spec owns the capstone contract."""

    text = (REPO / exp.REPORT_SPEC_PATH).read_text(encoding="utf-8")
    section = text.split("REQ-REPORT-6754", 1)[1]
    anchors = set(exp.spec_anchors(section))

    assert {
        "REQ-REPORT-6754",
        "SCENARIO-REPORT-6754-MISSING-BRANCH",
        "SCENARIO-REPORT-6754-ROW-RECOMPUTATION",
        "SCENARIO-REPORT-6754-VERDICT-PROPAGATION",
        "SCENARIO-REPORT-6754-NO-POOLED-CLAIM",
    } <= anchors
    for field in exp.REQUIRED_ARTIFACT_FIELDS:
        assert field in section
    assert exp.INFERENCE_SUBSTRATE in section
    assert exp.RESULT_PATH.as_posix() in section


def test_scenario_report_6754_presence_matrix_keeps_all_tasks(
    current_inputs: tuple[list[dict], dict[str, dict]],
) -> None:
    """SCENARIO-REPORT-6754-MISSING-BRANCH: all expected tasks get one row."""

    planned, sources = current_inputs
    matrix = exp.build_artifact_presence_matrix(REPO, planned, sources)

    assert [row["task_id"] for row in planned] == list(exp.EXPECTED_TASK_IDS)
    assert len(matrix) == 13
    assert sum(row["artifact_state"] == "present" for row in matrix) == 12
    assert matrix[-1]["task_id"] == exp.CAPSTONE_TASK_ID
    assert matrix[-1]["artifact_state"] == "current_synthesis"
    assert all(row["branch"] for row in matrix)


def test_scenario_report_6754_missing_branch_is_preserved(
    current_inputs: tuple[list[dict], dict[str, dict]],
) -> None:
    """SCENARIO-REPORT-6754-MISSING-BRANCH: absent branch artifacts do not block the capstone."""

    planned, sources = current_inputs
    missing_sources = deepcopy(sources)
    for task_id in exp.BRANCH_TASKS["fr12_diagnostics_repair"]:
        missing_sources[task_id] = exp.missing_source_record(task_id, "synthetic missing")

    task_rows = exp.build_task_rows(REPO, planned, missing_sources)
    headlines, mismatches = exp.recompute_headlines(missing_sources)
    branch_rows = exp.build_branch_rows(task_rows, headlines, mismatches, [])
    artifact = exp.build_artifact(
        REPO,
        duration_s=0.25,
        planned=planned,
        sources=missing_sources,
        validator_findings=[],
    )

    missing = [row for row in task_rows if row["branch"] == "fr12_diagnostics_repair"]
    assert {row["verdict_class"] for row in missing} == {"missing"}
    assert {row["artifact_state"] for row in missing} == {"missing"}
    assert {row["branch"]: row["verdict_class"] for row in branch_rows}[
        "fr12_diagnostics_repair"
    ] == "missing"
    assert artifact["artifact_presence_matrix"][2]["artifact_state"] == "missing"
    assert artifact["verdict_class"] == "partial"
    assert exp.validate_artifact(artifact) == []


def test_scenario_report_6754_recomputes_current_headlines_from_rows(
    current_inputs: tuple[list[dict], dict[str, dict]],
) -> None:
    """SCENARIO-REPORT-6754-ROW-RECOMPUTATION: headline values come from rows."""

    _planned, sources = current_inputs
    headlines, mismatches = exp.recompute_headlines(sources)

    activity = headlines["activity"]
    assert activity["first_token_reached"] == {"numerator": 3, "denominator": 3, "rate": 1.0}
    assert activity["teardown_completed"] == {"numerator": 3, "denominator": 3, "rate": 1.0}

    fr12 = headlines["fr12_diagnostics_repair"]
    assert fr12["certificate_rows"]["denominator"] == 72
    assert fr12["proposal_exact_valid"] == {"numerator": 0, "denominator": 216, "rate": 0.0}
    assert fr12["diagnosis_counts"] == {
        "abstention": 0,
        "exact_valid": 0,
        "malformed_certificate": 216,
        "reasoning_error": 0,
        "translation_disagreement": 0,
    }
    assert fr12["heldout_reasoning_error_auroc"] is None

    fr11 = headlines["fr11_continuous_self_learning"]
    assert fr11["prequential_exact_yield_delta_by_order"]["mean_delta"] == 0.0
    assert fr11["order_level_ci95"]["lower"] == 0.0
    assert fr11["commit_activity"]["commits"] == 0

    stochastic = headlines["stochastic_portability"]
    assert stochastic["trajectory_tv_by_arm"]["independent_factor"]["denominator"] == 64
    assert stochastic["trajectory_tv_by_arm"]["context_matched"]["value"] == pytest.approx(
        0.259491478137594
    )
    assert stochastic["trajectory_tv_by_arm"]["trajectory_refinement"]["value"] == pytest.approx(
        0.257737889413545
    )
    assert stochastic["hardware_used"] is False
    assert stochastic["simulator_used"] is True

    arc = headlines["arc_transport_object_table_quality"]
    assert arc["preflight_parse_dispatch_bounded"]["numerator"] == 2
    assert arc["preflight_parse_dispatch_bounded"]["denominator"] == 2
    assert arc["object_table_science_pairs"]["denominator"] == 0
    assert arc["object_table_science_pairs"]["value"] is None
    assert arc["solve_claim"] is False

    assert mismatches == []


def test_scenario_report_6754_mismatch_stays_visible(
    current_inputs: tuple[list[dict], dict[str, dict]],
) -> None:
    """SCENARIO-REPORT-6754-ROW-RECOMPUTATION: contradictory headlines are retained."""

    _planned, sources = current_inputs
    changed = deepcopy(sources)
    payload = deepcopy(changed["exp6751"]["payload"])
    payload["positive_result_gate"]["context_matched_mean_trajectory_tv"] = 9.0
    changed["exp6751"]["payload"] = payload

    headlines, mismatches = exp.recompute_headlines(changed)

    assert headlines["stochastic_portability"]["trajectory_tv_by_arm"]["context_matched"][
        "value"
    ] == pytest.approx(0.259491478137594)
    assert mismatches == [
        {
            "artifact": "results/experiment_6751_thermalizer_factor_trajectory_fidelity.json",
            "field": "positive_result_gate.context_matched_mean_trajectory_tv",
            "artifact_value": 9.0,
            "recomputed_value": pytest.approx(0.259491478137594),
            "reason": "headline_mismatch",
        }
    ]


def test_scenario_report_6754_verdict_classes_remain_branch_local(
    current_inputs: tuple[list[dict], dict[str, dict]],
) -> None:
    """SCENARIO-REPORT-6754-VERDICT-PROPAGATION: source classes stay local."""

    planned, sources = current_inputs
    task_rows = exp.build_task_rows(REPO, planned, sources)
    headlines, mismatches = exp.recompute_headlines(sources)
    validators = [
        {
            "artifact": "results/experiment_6751_thermalizer_factor_trajectory_fidelity.json",
            "validator": "verdict_row_consistency_lint",
            "exit_code": 1,
            "severity": "blocked",
            "findings": ["ALL_ROWS_NULL: synthetic validator fixture"],
            "report_hash": "sha256:fixture",
        }
    ]
    branch_rows = exp.build_branch_rows(task_rows, headlines, mismatches, validators)

    by_task = {row["task_id"]: row for row in task_rows}
    assert by_task["exp6748"]["verdict_class"] == "circular_positive"
    assert by_task["exp6749"]["verdict_class"] == "null"
    assert by_task["exp6753"]["verdict_class"] == "blocked"

    assert {row["branch"]: row["verdict_class"] for row in branch_rows} == {
        "handoff": "blocked",
        "activity": "positive",
        "fr12_diagnostics_repair": "blocked",
        "fr11_continuous_self_learning": "null",
        "stochastic_portability": "positive",
        "arc_transport_object_table_quality": "partial",
    }
    stochastic = next(row for row in branch_rows if row["branch"] == "stochastic_portability")
    assert stochastic["validator_blocking_findings"] == 1


def test_scenario_report_6754_complete_artifact_has_no_pooled_claim(
    current_inputs: tuple[list[dict], dict[str, dict]],
) -> None:
    """SCENARIO-REPORT-6754-NO-POOLED-CLAIM: branches are not averaged together."""

    planned, sources = current_inputs
    artifact = exp.build_artifact(
        REPO,
        duration_s=0.25,
        planned=planned,
        sources=sources,
        validator_findings=[],
    )

    assert exp.validate_artifact(artifact) == []
    assert set(exp.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert set(artifact) == set(artifact["field_principles"])
    assert artifact["status"] == "complete_terminal_partial"
    assert artifact["honest_verdict"].startswith("complete_partial:")
    assert artifact["verdict_class"] == "partial"
    assert artifact["inference_substrate"] == exp.INFERENCE_SUBSTRATE
    assert artifact["recomputed_headlines"]["pooled_milestone_success_score"] is None
    assert artifact["recomputed_headlines"]["pooled_success_claim_emitted"] is False
    assert not any("pooled_success" in row for row in artifact["rows"])
    self_rows = [row for row in artifact["artifact_presence_matrix"] if row["task_id"] == "exp6754"]
    assert self_rows == [
        {
            "task_id": "exp6754",
            "manifest_task_id": "exp6754-v588-branch-disposition",
            "branch": "capstone",
            "path": exp.RESULT_PATH.as_posix(),
            "artifact_state": "current_synthesis",
            "valid_json": True,
            "artifact_sha256": None,
            "row_count": 0,
            "verdict_class": "partial",
            "error": None,
        }
    ]
    assert artifact["branch_verdicts"] == {
        "handoff": "blocked",
        "activity": "positive",
        "fr12_diagnostics_repair": "blocked",
        "fr11_continuous_self_learning": "null",
        "stochastic_portability": "positive",
        "arc_transport_object_table_quality": "partial",
    }
    assert len(artifact["prd_gap_disposition"]) == 3
    assert [row["branch"] for row in artifact["next_licensed_actions"]] == list(exp.BRANCH_ORDER)
    assert any(
        row["task_id"] == "exp6747" and row["same_verdict_condition_fired"]
        for row in artifact["prior_failure_retirements"]
    )


@pytest.mark.parametrize(
    ("field", "bad"),
    [
        ("verdict_class", "positive"),
        ("inference_substrate", "live_llm_inference"),
        ("rows", []),
        ("branch_verdicts", {}),
        ("recomputed_headlines", {"pooled_success_claim_emitted": True}),
        ("field_principles", {}),
        ("reproducibility_checksum", "sha256:bad"),
    ],
)
def test_req_report_6754_validator_fails_closed(
    current_inputs: tuple[list[dict], dict[str, dict]],
    field: str,
    bad: object,
) -> None:
    """REQ-REPORT-6754: invalid capstone mutations are rejected."""

    planned, sources = current_inputs
    artifact = exp.build_artifact(
        REPO,
        duration_s=0.25,
        planned=planned,
        sources=sources,
        validator_findings=[],
    )
    changed = deepcopy(artifact)
    changed[field] = bad
    if field != "reproducibility_checksum":
        changed["reproducibility_checksum"] = exp.reproducibility_checksum(changed)
    assert exp.validate_artifact(changed)


def test_req_report_6754_atomic_write_and_wrapper(tmp_path: Path) -> None:
    """REQ-REPORT-6754: CLI and wrapper publish one valid JSON document."""

    target = tmp_path / "nested" / "artifact.json"
    assert exp.main(["--repo-root", str(REPO), "--output", str(target), "--skip-validators"]) == 0
    artifact = json.loads(target.read_text(encoding="utf-8"))
    assert exp.validate_artifact(artifact) == []
    assert exp.main(["--validate", "--output", str(target)]) == 0
    target.write_text(json.dumps({"bad": "payload"}), encoding="utf-8")
    assert exp.main(["--validate", "--output", str(target)]) == 1

    wrapper_path = REPO / "scripts/experiments/experiment_6754_v588_branch_disposition.py"
    saved_path = list(sys.path)
    saved_carnot = sys.modules.get("carnot")
    saved_module = sys.modules.get("carnot.experiment_6754_v588_branch_disposition")
    fake_carnot = types.ModuleType("carnot")
    fake_carnot.__path__ = []
    fake_module = types.ModuleType("carnot.experiment_6754_v588_branch_disposition")
    fake_module.main = exp.main
    sys.modules["carnot"] = fake_carnot
    sys.modules["carnot.experiment_6754_v588_branch_disposition"] = fake_module
    for path in (REPO, REPO / "python"):
        while str(path) in sys.path:
            sys.path.remove(str(path))
    spec = importlib.util.spec_from_file_location("exp6754_wrapper", wrapper_path)
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
            sys.modules.pop("carnot.experiment_6754_v588_branch_disposition", None)
        else:
            sys.modules["carnot.experiment_6754_v588_branch_disposition"] = saved_module
    assert (
        wrapper.main(
            [
                "--repo-root",
                str(REPO),
                "--output",
                str(tmp_path / "wrapper.json"),
                "--skip-validators",
            ]
        )
        == 0
    )


def _design_text(
    ids: tuple[str, ...] = exp.EXPECTED_TASK_IDS, milestone: str = exp.MILESTONE
) -> str:
    lines = [f"**Milestone:** `{milestone}`"]
    for task_id in ids:
        lines.extend(
            [
                f"### Exp {task_id.removeprefix('exp')}: Synthetic {task_id}",
                f"**Deliverable:** `{exp.TASK_PATHS[task_id]}`",
            ]
        )
    return "\n".join(lines)


def _manifest(deliverable_override: tuple[str, str] | None = None) -> dict[str, object]:
    tasks = []
    for full_id in exp.FULL_TASK_IDS:
        task_id = exp.short_task_id(full_id)
        deliverable = exp.TASK_PATHS[task_id]
        if deliverable_override and deliverable_override[0] == task_id:
            deliverable = deliverable_override[1]
        tasks.append(
            {
                "id": full_id,
                "title": f"Synthetic {task_id}",
                "deliverable": deliverable,
                "prior_failures": [],
            }
        )
    return {"milestone": exp.MILESTONE, "tasks": tasks}


def _write_plan_root(root: Path, manifest: object, design: str) -> None:
    (root / exp.ACTIVE_ROADMAP_PATH).write_text(yaml.safe_dump(manifest), encoding="utf-8")
    (root / exp.DESIGN_PATH.parent).mkdir(parents=True, exist_ok=True)
    (root / exp.DESIGN_PATH).write_text(design, encoding="utf-8")


def test_req_report_6754_defensive_edges(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """REQ-REPORT-6754: malformed local inputs fail closed or stay explicit."""

    with pytest.raises(ValueError, match="invalid V588 task id"):
        exp.short_task_id("bad-task")
    with pytest.raises(ValueError, match="deliverable missing"):
        exp._next_deliverable(["### Exp 6754: Missing", "### Exp 6755: Next"], 0)
    with pytest.raises(ValueError, match="milestone missing"):
        exp.parse_design_tasks("# no milestone")
    assert exp.sha256_file(tmp_path / "absent.json") is None

    _write_plan_root(tmp_path, [], _design_text())
    with pytest.raises(ValueError, match="mapping with tasks"):
        exp.load_planned_tasks(tmp_path)
    _write_plan_root(tmp_path, _manifest(), _design_text(milestone="2026.08.999"))
    with pytest.raises(ValueError, match="expected V588 design"):
        exp.load_planned_tasks(tmp_path)
    _write_plan_root(tmp_path, _manifest(), _design_text(exp.EXPECTED_TASK_IDS[:-1]))
    with pytest.raises(ValueError, match="Exp6742 through Exp6754"):
        exp.load_planned_tasks(tmp_path)
    bad_manifest = _manifest()
    bad_manifest["tasks"] = bad_manifest["tasks"][:-1]
    _write_plan_root(tmp_path, bad_manifest, _design_text())
    with pytest.raises(ValueError, match="exact 13 tasks"):
        exp.load_planned_tasks(tmp_path)
    _write_plan_root(tmp_path, _manifest(("exp6742", "results/wrong.json")), _design_text())
    with pytest.raises(ValueError, match="deliverable mismatch"):
        exp.load_planned_tasks(tmp_path)

    planned = [
        {"task_id": "exp6742", "path": "missing.json"},
        {"task_id": "exp6743", "path": "bad.json"},
        {"task_id": exp.CAPSTONE_TASK_ID, "path": exp.RESULT_PATH.as_posix()},
    ]
    (tmp_path / "bad.json").write_text("[]", encoding="utf-8")
    sources = exp.load_source_artifacts(tmp_path, planned)
    assert sources["exp6742"]["artifact_state"] == "missing"
    assert sources["exp6743"]["artifact_state"] == "invalid"
    assert sources[exp.CAPSTONE_TASK_ID]["artifact_state"] == "current_synthesis"

    assert exp._record_class({"artifact_state": "invalid"}) == "disqualified"
    assert exp._record_class({"artifact_state": "present", "payload": []}) == "missing"
    assert (
        exp._record_class(
            {"artifact_state": "present", "payload": {"honest_verdict": "circular_positive"}}
        )
        == "circular_positive"
    )
    assert (
        exp._record_class({"artifact_state": "present", "payload": {"honest_verdict": "partial"}})
        == "partial"
    )
    assert (
        exp._record_class({"artifact_state": "present", "payload": {"honest_verdict": "success"}})
        == "positive"
    )
    assert (
        exp._record_class({"artifact_state": "present", "payload": {"honest_verdict": "complete"}})
        == "null"
    )
    assert (
        exp._record_class({"artifact_state": "present", "payload": {"honest_verdict": "unknown"}})
        == "disqualified"
    )
    assert exp._get_path({"a": {}}, "a.b") is None
    assert exp._arm_metric([{"arm": "a", "ok": True}], "ok")["a"]["value"] == 1.0
    assert exp.gate_failures("exp1", 42) == []
    assert exp._authority_boundary(None)["models_used"] is None

    minimal_headlines = {
        branch: {
            "handoff_contract_preserved": False,
            "accelerator_receipt_ready": False,
            "object_table_ab_completed": False,
            "adoption_gate_passed": False,
            "preflight_parse_dispatch_bounded": {"rate": 0.0},
            "compiler_fidelity_completed": False,
            "context_reduced_vs_independent": False,
            "trajectory_reduced_vs_independent": False,
        }
        for branch in exp.BRANCH_ORDER
    }
    assert (
        exp._class_for_branch(
            "handoff", ["blocked"], minimal_headlines, [{"artifact": exp.TASK_PATHS["exp6742"]}]
        )
        == "disqualified"
    )
    assert (
        exp._class_for_branch(
            "arc_transport_object_table_quality", ["positive"], minimal_headlines, []
        )
        == "blocked"
    )
    minimal_headlines["arc_transport_object_table_quality"]["object_table_ab_completed"] = True
    assert (
        exp._class_for_branch(
            "arc_transport_object_table_quality", ["positive"], minimal_headlines, []
        )
        == "null"
    )
    minimal_headlines["arc_transport_object_table_quality"]["adoption_gate_passed"] = True
    assert (
        exp._class_for_branch(
            "arc_transport_object_table_quality", ["positive"], minimal_headlines, []
        )
        == "positive"
    )
    with pytest.raises(ValueError, match="unknown branch"):
        exp._class_for_branch("bad", [], minimal_headlines, [])
    try:
        exp._class_for_branch("also_bad", [], minimal_headlines, [])
    except ValueError as exc:
        assert "unknown branch also_bad" in str(exc)
    else:
        pytest.fail("unknown branch did not fail closed")

    prior_rows = exp.build_prior_failure_retirements(
        [{"task_id": "exp6742", "prior_failures": ["bad"]}],
        [{"task_id": "exp6742", "branch": "handoff", "honest_verdict": "blocked"}],
    )
    assert prior_rows == []

    command_calls = []

    def fake_run(args: list[str], _root: Path) -> tuple[int, str]:
        command_calls.append(args)
        if str(exp.ADVERSARIAL_SCRIPT) in args:
            return (
                0,
                json.dumps(
                    {
                        "reports": [
                            {
                                "artifact": "results/experiment_6742_v588_handoff_contract_audit.json",
                                "flags": [{"kind": "FLAG", "severity": 2}],
                            }
                        ]
                    }
                ),
            )
        return 1, "  [BLOCK] ALL_ROWS_NULL: fixture\nsummary"

    validator_sources = {
        task_id: exp.missing_source_record(task_id, "fixture_missing")
        for task_id in exp.EXPECTED_TASK_IDS
    }
    validator_sources["exp6742"] = {
        "task_id": "exp6742",
        "path": "results/experiment_6742_v588_handoff_contract_audit.json",
        "artifact_state": "present",
        "valid_json": True,
        "payload": {},
        "sha256": "sha256:fixture",
        "error": None,
    }
    monkeypatch.setattr(exp, "_run_command", fake_run)
    findings = exp.run_validator_findings(REPO, validator_sources)
    by_validator = {
        row["validator"]: row
        for row in findings
        if row["artifact"] == validator_sources["exp6742"]["path"]
    }
    assert by_validator["adversarial_verify"]["flag_count"] == 1
    assert by_validator["verdict_row_consistency_lint"]["findings"] == [
        "[BLOCK] ALL_ROWS_NULL: fixture"
    ]
    assert command_calls

    monkeypatch.setattr(exp, "_run_command", lambda *_args: (1, "not-json"))
    findings = exp.run_validator_findings(REPO, validator_sources)
    assert any(
        row["validator"] == "adversarial_verify" and row["exit_code"] == 1 for row in findings
    )

    planned_real = exp.load_planned_tasks(REPO)
    sources_real = exp.load_source_artifacts(REPO, planned_real)
    artifact = exp.build_artifact(
        REPO,
        duration_s=0.25,
        planned=planned_real,
        sources=sources_real,
        validator_findings=[],
    )
    bad_closed = deepcopy(artifact)
    bad_closed["branch_verdicts"]["handoff"] = "bad"
    bad_closed["reproducibility_checksum"] = exp.reproducibility_checksum(bad_closed)
    assert "branch_verdicts_closed_class" in exp.validate_artifact(bad_closed)
    bad_score = deepcopy(artifact)
    bad_score["recomputed_headlines"]["pooled_milestone_success_score"] = 1.0
    bad_score["reproducibility_checksum"] = exp.reproducibility_checksum(bad_score)
    assert "pooled_milestone_success_score" in exp.validate_artifact(bad_score)
    with pytest.raises(ValueError, match="invalid Exp6754 artifact"):
        exp.write_json_atomic(tmp_path / "bad-artifact.json", bad_score)
    bad_root = tmp_path / "bad-root.json"
    bad_root.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="JSON object"):
        exp._load_artifact(bad_root)
    assert exp.main(["--validate", "--output", str(tmp_path / "missing-output.json")]) == 1

    target = tmp_path / "replace-fails.json"
    real_replace = exp.os.replace
    try:
        exp.os.replace = lambda *_args: (_ for _ in ()).throw(OSError("replace failed"))
        with pytest.raises(OSError, match="replace failed"):
            exp.write_json_atomic(target, artifact)
    finally:
        exp.os.replace = real_replace
    assert not list(tmp_path.glob("*.tmp"))

    real_validate = exp.validate_artifact
    try:
        exp.validate_artifact = lambda _payload: ["forced-invalid"]
        assert (
            exp.main(
                [
                    "--repo-root",
                    str(REPO),
                    "--output",
                    str(tmp_path / "forced.json"),
                    "--skip-validators",
                ]
            )
            == 1
        )
    finally:
        exp.validate_artifact = real_validate
