"""Focused tests for the V587 activation evidence contract.

Spec refs: REQ-REPORT-6729, REQ-HARNESS-002, REQ-HARNESS-008,
SCENARIO-REPORT-6729-READY, SCENARIO-REPORT-6729-BLOCKED, and
SCENARIO-REPORT-6729-ATOMIC.
"""

from __future__ import annotations

from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys

import pytest

from carnot import experiment_6729_v587_activation_evidence_contract as exp


REPO = Path(__file__).resolve().parents[2]


def _mutated_audit(manifest: dict[str, object], **kwargs: object) -> dict[str, object]:
    design = exp.parse_design_contract((REPO / exp.DESIGN_PATH).read_text(encoding="utf-8"))
    receipts = exp.collect_source_receipts(REPO, design)
    return exp.audit_contract(
        REPO,
        design,
        manifest,
        deepcopy(manifest),
        receipts,
        retired_ids=set(),
        **kwargs,
    )


def test_req_report_6729_spec_and_design_own_the_contract() -> None:
    """REQ-REPORT-6729: the spec names the durable V587 receipt contract."""

    reporting = (REPO / exp.SPEC_PATH).read_text(encoding="utf-8")
    harness = (REPO / exp.HARNESS_SPEC_PATH).read_text(encoding="utf-8")
    section = reporting.split("REQ-REPORT-6729", 1)[1]
    for field in exp.REQUIRED_ARTIFACT_FIELDS:
        assert field in section
    for marker in (
        "SCENARIO-REPORT-6729-READY",
        "SCENARIO-REPORT-6729-BLOCKED",
        "SCENARIO-REPORT-6729-ATOMIC",
        exp.INFERENCE_SUBSTRATE,
        exp.RESULT_PATH.as_posix(),
    ):
        assert marker in section
    assert "REQ-HARNESS-002" in harness
    assert "REQ-HARNESS-008" in harness

    design = exp.parse_design_contract((REPO / exp.DESIGN_PATH).read_text(encoding="utf-8"))
    assert [row["task_id"] for row in design["tasks"]] == list(exp.EXPECTED_TASK_IDS)
    assert sorted(design["phases"]) == ["Phase 1", "Phase 2", "Phase 3", "Phase 4"]
    assert len(design["prd_gaps"]) == 3
    assert design["tasks"][-1]["task_id"] == exp.CAPSTONE_TASK_ID


def test_scenario_report_6729_active_manifest_blocks_on_missing_next() -> None:
    """SCENARIO-REPORT-6729-BLOCKED: missing and partial inputs stay visible."""

    artifact = exp.build_artifact(REPO, duration_s=1.0)
    assert artifact["v587_contract_ready"] is False
    assert artifact["verdict_class"] == "blocked"
    assert artifact["honest_verdict"].startswith("blocked_")
    assert len(artifact["source_receipts"]) == len(exp.PRIMARY_SOURCE_IDS)
    assert len(artifact["task_contract_rows"]) == len(exp.EXPECTED_TASK_IDS)
    checks = {(row["check"], row["unit"]) for row in artifact["gate_check_summary"]}
    assert ("precondition.local_file", exp.NEXT_ROADMAP_PATH.as_posix()) in checks
    assert ("manifest.task_count", exp.ACTIVE_ROADMAP_PATH.as_posix()) in checks
    assert any(row["observed_value"] == 7 for row in artifact["gate_check_summary"])
    assert all(row["passed"] for row in artifact["source_receipts"])
    assert exp.validate_artifact(artifact) == []


def test_scenario_report_6729_synthetic_complete_manifest_passes() -> None:
    """SCENARIO-REPORT-6729-READY: complete rows reduce to readiness."""

    manifest = exp.synthetic_ready_manifest(REPO)
    audit = _mutated_audit(manifest)
    assert audit["passed"] is True
    assert len(audit["task_contract_rows"]) == 13
    assert len(audit["gate_contract_rows"]) == 11
    assert len(audit["prior_failure_rows"]) == 12
    assert len(audit["model_policy_rows"]) == 5
    assert all(row["passed"] for row in audit["task_contract_rows"])
    assert all(row["passed"] for row in audit["gate_contract_rows"])
    assert all(row["passed"] for row in audit["prior_failure_rows"])
    assert all(row["passed"] for row in audit["model_policy_rows"])

    artifact = exp.build_artifact(
        REPO,
        duration_s=2.0,
        active_payload=manifest,
        next_payload=deepcopy(manifest),
        require_local_files=False,
    )
    assert artifact["v587_contract_ready"] is True
    assert artifact["verdict_class"] == "null"
    assert artifact["honest_verdict"].startswith("complete_null:")
    assert artifact["gate_check_summary"] == []
    assert exp.validate_artifact(artifact) == []


def test_scenario_report_6729_fail_closed_mutations_are_named() -> None:
    """SCENARIO-REPORT-6729-BLOCKED: contract attacks do not normalize away."""

    manifest = exp.synthetic_ready_manifest(REPO)
    cases: list[tuple[dict[str, object], set[str]]] = []

    duplicate = deepcopy(manifest)
    duplicate["tasks"][1]["deliverable"] = duplicate["tasks"][0]["deliverable"]
    cases.append((duplicate, {"manifest.deliverables_unique"}))

    renamed_gate = deepcopy(manifest)
    renamed_gate["tasks"][1]["gated_on"][0]["artifact_field"] = "renamed_ready"
    cases.append((renamed_gate, {"gate.producer_field"}))

    incomplete_prior = deepcopy(manifest)
    incomplete_prior["tasks"][2]["prior_failures"][0]["retire_if_same_verdict"] = False
    cases.append((incomplete_prior, {"prior.failure_contract"}))

    no_rows = deepcopy(manifest)
    no_rows["tasks"][2]["per_unit_rows"] = False
    cases.append((no_rows, {"task.per_unit_rows"}))

    arc_claim = deepcopy(manifest)
    arc_claim["tasks"][1]["prompt"] += "\nThis variant uses game source and BFS per-game adapters."
    cases.append((arc_claim, {"task.arc_boundary"}))

    missing_model = deepcopy(manifest)
    missing_model["tasks"][1]["prompt"] = missing_model["tasks"][1]["prompt"].replace(
        exp.ARC_GENERATOR_MODEL, "wrong/model-GGUF"
    )
    cases.append((missing_model, {"model.policy"}))

    for candidate, expected_checks in cases:
        audit = _mutated_audit(candidate)
        observed = {row["check"] for row in audit["failures"]}
        assert expected_checks <= observed
        assert audit["passed"] is False


def test_scenario_report_6729_atomic_cli_and_validation(tmp_path: Path) -> None:
    """SCENARIO-REPORT-6729-ATOMIC: writes are atomic and hashes replay."""

    artifact = exp.build_artifact(REPO, duration_s=1.0)
    target = tmp_path / "nested" / "artifact.json"
    exp.write_json_atomic(target, artifact)
    assert json.loads(target.read_text(encoding="utf-8")) == artifact
    assert not list(target.parent.glob("*.tmp"))

    broken = deepcopy(artifact)
    broken["v587_contract_ready"] = True
    assert "reproducibility_checksum_mismatch" in exp.validate_artifact(broken)

    bad_principles = deepcopy(artifact)
    bad_principles["field_principles"] = {}
    bad_principles["reproducibility_checksum"] = exp.reproducibility_checksum(bad_principles)
    assert "field_principles_missing" in exp.validate_artifact(bad_principles)

    bad_ready = deepcopy(artifact)
    bad_ready["gate_check_summary"] = []
    bad_ready["reproducibility_checksum"] = exp.reproducibility_checksum(bad_ready)
    assert "blocked_gate_summary_missing" in exp.validate_artifact(bad_ready)

    real_replace = exp.os.replace
    try:
        exp.os.replace = lambda *_args: (_ for _ in ()).throw(OSError("replace failed"))
        with pytest.raises(OSError, match="replace failed"):
            exp.write_json_atomic(tmp_path / "failed.json", artifact)
    finally:
        exp.os.replace = real_replace
    assert not list(tmp_path.glob("*.tmp"))

    assert exp.main(["--output", str(tmp_path / "cli.json")]) == 0
    cli_payload = json.loads((tmp_path / "cli.json").read_text(encoding="utf-8"))
    assert cli_payload["verdict_class"] == "blocked"

    assert exp.main(["--validate", "--output", str(tmp_path / "cli.json")]) == 0
    (tmp_path / "cli.json").write_text(json.dumps({"bad": "payload"}), encoding="utf-8")
    assert exp.main(["--validate", "--output", str(tmp_path / "cli.json")]) == 1

    wrapper_path = REPO / "scripts/experiments/experiment_6729_v587_activation_evidence_contract.py"
    saved_path = list(sys.path)
    for path in (REPO, REPO / "python"):
        while str(path) in sys.path:
            sys.path.remove(str(path))
    spec = importlib.util.spec_from_file_location("exp6729_wrapper", wrapper_path)
    assert spec and spec.loader
    wrapper = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(wrapper)
    finally:
        sys.path = saved_path
    assert wrapper.main(["--output", str(tmp_path / "wrapper.json")]) == 0


def test_scenario_report_6729_defensive_helper_edges(tmp_path: Path) -> None:
    """SCENARIO-REPORT-6729-BLOCKED: parser edge cases remain deterministic."""

    with pytest.raises(ValueError, match="task deliverable missing"):
        exp._next_deliverable(["### Exp 1: Missing", "### Exp 2: Next"], 0)
    assert exp._source_block("2608.08786 only source", "2608.08786") == "2608.08786 only source"
    assert exp.extract_model_specs("no specs here") == []
    assert exp._as_list("one") == ["one"]

    assert exp.build_gate_contract_rows({"tasks": [{"id": "exp1-a", "gated_on": ["bad"]}]}) == []
    assert exp.build_prior_failure_rows({"tasks": ["bad"]}) == []

    (tmp_path / "ops").mkdir()
    (tmp_path / exp.EXCLUSION_PATH).write_text("retired:\n- bad\n", encoding="utf-8")
    assert exp.retired_task_ids(tmp_path) == set()

    bad_design = {
        "tasks": [
            {"task_id": "exp1-a", "deliverable": "results/a.json"},
            {"task_id": "exp1-a", "deliverable": "results/a.json"},
        ],
        "phases": ["Phase 1"],
        "prd_gaps": [],
    }
    assert {
        "design.task_count",
        "design.task_ids_unique",
        "design.deliverables_unique",
        "design.phase_count",
        "design.prd_gap_count",
    } == {row["check"] for row in exp._design_failures(bad_design)}

    real_validate = exp.validate_artifact
    try:
        exp.validate_artifact = lambda _payload: ["forced-invalid"]
        assert exp.main(["--output", str(tmp_path / "invalid-generate.json")]) == 1
    finally:
        exp.validate_artifact = real_validate
