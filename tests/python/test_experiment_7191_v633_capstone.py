"""Tests for REQ-REPORT-7191 and its named V633 capstone scenarios."""

from __future__ import annotations

from copy import deepcopy
import importlib.util
import json
from pathlib import Path
from typing import Any

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = ROOT / "scripts/experiments/experiment_7191_v633_capstone.py"
MODULE_SPEC = importlib.util.spec_from_file_location("experiment_7191_v633_capstone", MODULE_PATH)
assert MODULE_SPEC is not None and MODULE_SPEC.loader is not None
exp = importlib.util.module_from_spec(MODULE_SPEC)
MODULE_SPEC.loader.exec_module(exp)


def _clean_adversarial(path: Path) -> dict[str, Any]:
    """Return the small clean shape emitted by the shipped checker."""

    return {
        "artifact": str(path),
        "loaded": True,
        "flag_count": 0,
        "max_severity": -1,
        "flags": [],
        "gate_version": "fixture-gate-version",
    }


def _clean_row(path: Path) -> tuple[str, list[str]]:
    """Return a row-check result without hiding the checker interface."""

    return "ok", []


def _checker_loader(_root: Path) -> tuple[Any, Any]:
    """Keep unit integration tests focused on the capstone parser."""

    return _clean_adversarial, _clean_row


def _write_contract(root: Path) -> list[dict[str, Any]]:
    """Write the frozen task records as one matching active authority."""

    tasks = []
    for task_id, title, deliverable in exp.EXPECTED_CONTRACT:
        tasks.append(
            {
                "id": task_id,
                "title": title,
                "deliverable": deliverable,
                "milestone": exp.MILESTONE,
                "per_unit_rows": True,
            }
        )
    root.mkdir(parents=True, exist_ok=True)
    (root / "research-roadmap.yaml").write_text(
        yaml.safe_dump({"milestone": exp.MILESTONE, "tasks": tasks}, sort_keys=False),
        encoding="utf-8",
    )
    return tasks


def test_req_report_7191_spec_precedes_implementation() -> None:
    """REQ-REPORT-7191: the spec names all artifact fields and scenarios."""

    text = (ROOT / exp.SPEC_PATH).read_text(encoding="utf-8")
    section = text[text.index("### REQ-REPORT-7191") :]
    assert all(field in section for field in exp.REQUIRED_ARTIFACT_FIELDS)
    for name in ("CONTRACT", "FALLBACK", "CLAIMS", "BOUNDARIES", "DECISIONS", "BLOCKED", "ARTIFACT"):
        assert f"SCENARIO-REPORT-7191-{name}" in section


def test_scenario_report_7191_contract_selects_active_then_archive(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7191-CONTRACT: milestone identity selects exact rows."""

    tasks = _write_contract(tmp_path)
    rows, sources, selected, errors = exp.resolve_contract(tmp_path)
    assert errors == []
    assert selected == "research-roadmap.yaml"
    assert [row["id"] for row in rows] == list(exp.EXPECTED_TASK_IDS)
    assert sources[0]["selected"] is True

    (tmp_path / "research-roadmap.yaml").write_text(
        yaml.safe_dump({"milestone": "wrong", "tasks": tasks}), encoding="utf-8"
    )
    archive = tmp_path / exp.ARCHIVED_ROADMAP_PATH
    archive.parent.mkdir(parents=True)
    archive.write_text(
        yaml.safe_dump({"milestone": exp.MILESTONE, "tasks": tasks}, sort_keys=False),
        encoding="utf-8",
    )
    rows, sources, selected, errors = exp.resolve_contract(tmp_path)
    assert errors == []
    assert selected == exp.ARCHIVED_ROADMAP_PATH
    assert sources[-1]["selected"] is True


def test_scenario_report_7191_contract_rejects_bad_roster(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7191-CONTRACT: a shortened authority fails closed."""

    tasks = _write_contract(tmp_path)
    tasks.pop()
    (tmp_path / "research-roadmap.yaml").write_text(
        yaml.safe_dump({"milestone": exp.MILESTONE, "tasks": tasks}), encoding="utf-8"
    )
    rows, _sources, selected, errors = exp.resolve_contract(tmp_path)
    assert selected == "research-roadmap.yaml"
    assert len(rows) == 12
    assert "contract_id_order" in errors


def test_scenario_report_7191_fallback_reads_only_canonical_path(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7191-FALLBACK: declared then canonical paths are explicit."""

    task = {
        "id": "exp7186-arc-withheld-transfer",
        "title": "ARC adapter-withheld generalization with matched live arms",
        "deliverable": "results/experiment_7186_v633_arc_withheld_transfer.json",
        "gated_on": [],
        "prior_failures": [],
    }
    canonical = tmp_path / "results/experiment_7186_arc_withheld_transfer.json"
    canonical.parent.mkdir(parents=True)
    canonical.write_text(
        json.dumps({"status": "blocked", "honest_verdict": "blocked_gate_check_failed"}),
        encoding="utf-8",
    )

    record = exp.load_task_evidence(tmp_path, task)
    assert record["declared_deliverable_path"] == task["deliverable"]
    assert record["canonical_gate_block_path"] == "results/experiment_7186_arc_withheld_transfer.json"
    assert record["selected_evidence_path"] == record["canonical_gate_block_path"]
    assert record["evidence_source"] == "conductor_gate_block"

    canonical.unlink()
    missing = exp.load_task_evidence(tmp_path, task)
    assert missing["selected_evidence_path"] is None
    assert missing["evidence_source"] == "missing"


def test_scenario_report_7191_claims_recompute_all_current_headlines() -> None:
    """SCENARIO-REPORT-7191-CLAIMS: current producer scores rebuild from rows."""

    claim_rows = []
    for task_id in exp.EXPECTED_TASK_IDS[:-1]:
        number = exp.task_number(task_id)
        path = next(ROOT.glob(f"results/experiment_{number}_v633*.json"))
        payload = json.loads(path.read_text(encoding="utf-8"))
        claim_rows.extend(exp.recompute_claims(task_id, payload))

    assert claim_rows
    assert all(row["matches"] for row in claim_rows)
    assert {row["task_id"] for row in claim_rows} == set(exp.EXPECTED_TASK_IDS[:-1])
    changed_targets = next(
        row for row in claim_rows if row["claim"] == "naive_changed_target_law_count"
    )
    assert changed_targets["declared_value"] == changed_targets["recomputed_value"] == 162


def test_scenario_report_7191_claim_mutation_is_detected() -> None:
    """SCENARIO-REPORT-7191-CLAIMS: a forged producer headline does not match rows."""

    payload = json.loads(
        (ROOT / "results/experiment_7189_v633_rust_slice_parity.json").read_text(
            encoding="utf-8"
        )
    )
    payload["rust_slice_parity_score"] = 0
    rows = exp.recompute_claims("exp7189-rust-slice-parity", payload)
    assert any(row["claim"] == "rust_slice_parity_score" and not row["matches"] for row in rows)


@pytest.fixture(scope="module")
def artifact(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    """Build one current artifact with deterministic checker responses."""

    directory = tmp_path_factory.mktemp("exp7191")
    return exp.build_artifact(
        ROOT,
        exp.RUN_DATE,
        directory / "experiment_7191.json",
        directory / "checkpoint.json",
        checker_loader=_checker_loader,
    )


def test_scenario_report_7191_blocked_matrix_is_complete(artifact: dict[str, Any]) -> None:
    """SCENARIO-REPORT-7191-BLOCKED: blocked science coexists with complete work."""

    assert artifact["status"] == "complete"
    assert artifact["verdict_class"] == "blocked"
    assert artifact["inference_substrate_class"] == "aggregation"
    assert artifact["capstone_complete_score"] == 1
    assert artifact["gate_check_summary"] == {
        "passed": False,
        "failed_check": "upstream_terminal_evidence",
        "upstream": "exp7186-arc-withheld-transfer",
        "field": "REQUIRED_SOURCE_PATHS.python/carnot/agentic/arc_eval_runner.py",
        "expected_value": "nonempty_file",
        "observed_value": 0,
    }
    assert len(artifact["rows"]) == len(artifact["evidence_matrix"]) == 13
    assert artifact["rows"][-1]["evidence_source"] == "self"


def test_scenario_report_7191_boundaries_preserve_scope(artifact: dict[str, Any]) -> None:
    """SCENARIO-REPORT-7191-BOUNDARIES: model, ARC, and hardware claims stay separate."""

    rows = {row["task_id"]: row for row in artifact["evidence_matrix"]}
    assert rows["exp7181-qwen38-symbolic-traces"]["llm_evidence_scope"] == "live_qwen_generation"
    assert rows["exp7182-grounding-energy-audit"]["llm_evidence_scope"] == "cpu_replay_of_live_qwen"
    assert rows["exp7186-arc-withheld-transfer"]["new_arc_solve_claimed"] is False
    assert rows["exp7190-board-placement-receipt"]["hardware_evidence_scope"] == "host_placement_only"
    assert rows["exp7190-board-placement-receipt"]["hardware_execution_claimed"] is False
    assert rows["exp7184-revocable-template-csl"]["measurement_completed"] is True
    assert rows["exp7184-revocable-template-csl"]["benefit_established"] is False


def test_scenario_report_7191_decisions_cover_exact_scopes(artifact: dict[str, Any]) -> None:
    """SCENARIO-REPORT-7191-DECISIONS: all tasks receive one bounded action."""

    decisions = {row["task_id"]: row for row in artifact["branch_decisions"]}
    assert set(decisions) == set(exp.EXPECTED_TASK_IDS)
    assert {row["action"] for row in decisions.values()} <= exp.BRANCH_ACTIONS
    assert decisions["exp7179-contract-receipt"]["action"] == "retire"
    assert decisions["exp7179-contract-receipt"]["prior_failure_id"] == "exp7166-v632-exact-contract-preflight"
    assert decisions["exp7186-arc-withheld-transfer"]["action"] == "needs_changed_prerequisite"
    assert decisions["exp7190-board-placement-receipt"]["action"] == "retire"
    assert decisions["exp7191-capstone"]["prior_failure_id"] == "exp7147-v627-capstone"


def test_scenario_report_7191_artifact_validator_rejects_mutations(
    artifact: dict[str, Any],
) -> None:
    """SCENARIO-REPORT-7191-ARTIFACT: derived fields and checksum fail closed."""

    assert exp.validate_artifact(artifact, root=ROOT) == []
    mutations = (
        ("capstone_complete_score", 0),
        ("rows", []),
        ("recomputed_claim_rows", []),
        ("branch_decisions", []),
        ("verdict_class", "positive"),
    )
    for field, value in mutations:
        changed = deepcopy(artifact)
        changed[field] = value
        changed["reproducibility_checksum"] = exp.reproducibility_checksum(changed)
        assert exp.validate_artifact(changed, root=ROOT), field

    changed = deepcopy(artifact)
    changed["reproducibility_checksum"] = "sha256:wrong"
    assert "reproducibility_checksum_mismatch" in exp.validate_artifact(changed)


def test_req_report_7191_preflight_failure_writes_terminal_block(tmp_path: Path) -> None:
    """REQ-REPORT-7191: an essential local absence stops before aggregation."""

    output = tmp_path / "results/result.json"
    checkpoint = tmp_path / "results/checkpoints/checkpoint.json"
    artifact = exp.build_artifact(
        tmp_path,
        exp.RUN_DATE,
        output,
        checkpoint,
        checker_loader=_checker_loader,
    )
    assert output.is_file() and checkpoint.is_file()
    assert artifact["status"] == "blocked"
    assert artifact["verdict_class"] == "blocked"
    assert artifact["inference_substrate_class"] == "blocked_no_run"
    assert artifact["capstone_complete_score"] == 0
    assert artifact["gate_check_summary"]["expected_value"] == "REQ-REPORT-7191 present"
    assert exp.validate_artifact(artifact) == []


def test_req_report_7191_checksum_ignores_only_measured_duration(
    artifact: dict[str, Any],
) -> None:
    """REQ-REPORT-7191: timing can vary while scientific rows stay bound."""

    changed = deepcopy(artifact)
    changed["duration_s"] = 999.0
    assert exp.reproducibility_checksum(changed) == artifact["reproducibility_checksum"]
    changed["evidence_matrix"][0]["honest_verdict"] = "forged"
    assert exp.reproducibility_checksum(changed) != artifact["reproducibility_checksum"]


def test_req_report_7191_cli_file_parser_gate(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7191-ARTIFACT: the command runs file to parser to gate."""

    output = tmp_path / "capstone.json"
    checkpoint = tmp_path / "checkpoint.json"
    args = [
        "--root",
        str(ROOT),
        "--date",
        exp.RUN_DATE,
        "--artifact-path",
        str(output),
        "--checkpoint-path",
        str(checkpoint),
    ]
    assert exp.main(args, checker_loader=_checker_loader) == 0
    assert exp.main(["--root", str(ROOT), "--validate", "--artifact-path", str(output)]) == 0
    output.write_text("{}", encoding="utf-8")
    assert exp.main(["--validate", "--artifact-path", str(output)]) == 1


def test_req_report_7191_rejects_bad_date_and_json(tmp_path: Path) -> None:
    """REQ-REPORT-7191: malformed dates and non-object JSON fail closed."""

    with pytest.raises(ValueError, match="20260910"):
        exp.initialize_artifact("20260909")
    path = tmp_path / "bad.json"
    path.write_text("[]", encoding="utf-8")
    payload, error = exp.read_json_object(path)
    assert payload is None
    assert error == "json_root_not_object"
