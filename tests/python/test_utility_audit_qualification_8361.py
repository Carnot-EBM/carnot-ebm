"""REQ-REPORT-8361 / REQ-VERIFY-8361: qualify real evidence without new science."""

from copy import deepcopy
import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from carnot.reporting import utility_audit_qualification_8361 as e
from carnot.reporting.current_work_receipt import atomic_json, sha256_file
from carnot.reporting import static_benefit_audit_8350 as static
from carnot.reporting import learning_retention_audit_8351 as learning


@pytest.fixture(scope="module")
def natural(tmp_path_factory: pytest.TempPathFactory) -> tuple[dict[str, Any], Path]:
    """SCENARIO-VERIFY-8361-REDUCTION: only original sealed observations enter science."""
    raw = tmp_path_factory.mktemp("8361-natural") / "raw"
    work = e.measure(e.ROOT, raw)
    assert not work["failures"], work["failures"]
    return work, raw


def test_natural_reductions(natural: tuple[dict[str, Any], Path]) -> None:
    """SCENARIO-VERIFY-8361-REDUCTION: nulls preserve support and frozen gates."""
    work, raw = natural
    value = e.build(work, raw, [dict(name="control", passed=True)])
    assert value["verdict_class"] == "null"
    assert value["static_audit_ready_score"] == value["learning_audit_ready_score"] == 1
    assert value["H1"]["bootstrap_summary"]["all_intended"]["mean_gain"] == -0.00390625
    assert value["H2"]["block_bootstrap_summary"]["mean_gain"] == 0
    assert value["source_support"]["H1"]["complete_count"] == 97
    assert value["source_support"]["H2"]["complete_count"] == 67
    assert [r["window"] for r in value["retention_windows"]] == [0, 32, 64, 96]
    assert all(
        r["intended_count"] == 32 and r["qualified_count"] == 23 for r in value["retention_windows"]
    )
    assert value["h1_development_signal_score"] == value["h2_development_signal_score"] == 0
    assert (
        value["independent_generalization_score"]
        == value["generalized_learning_benefit_score"]
        == 0
    )
    assert value["MODEL_SPECS"] == [] and value["no_model_load"]
    assert value["H1"]["optimizer_qualification"]["geometry_qualified"] is False
    assert value["qualified_scope_decision"]["H1"] == "null_optimization_limited"
    assert value["qualified_scope_decision"]["H2"] == "null_delayed_decision_benefit"
    assert value["H2"]["utility_null_informative"]
    assert value["budget_reachability"]["reachable_count"] == 2
    assert e.build(work, raw, [dict(passed=False)])["verdict_class"] == "disqualified"


@pytest.mark.parametrize("number,line", [(8350, 599), (8351, 517)])
def test_rehashed_real_rejection(
    natural: tuple[dict[str, Any], Path], tmp_path: Path, number: int, line: int
) -> None:
    """SCENARIO-VERIFY-8361-REJECTIONS: both formerly missed guards execute."""
    work, _ = natural
    result = work["controls"][str(number)]
    assert result["passed"] and result["target_line"] == line
    assert result["valid_replay_receipt"]["exit_code"] == 0
    assert result["tamper_replay_receipt"]["exit_code"] == 1
    assert line in result["executed_lines"]
    assert result["self_consistently_rehashed"]


def test_cold_replay_tamper(natural: tuple[dict[str, Any], Path], tmp_path: Path) -> None:
    """SCENARIO-REPORT-8361-CLI: rehashing summaries or primitives cannot create readiness."""
    work, raw = natural
    candidate = tmp_path / "candidate.json"
    receipts = [dict(passed=True)]
    atomic_json(candidate, e.build(work, raw, receipts))
    assert e.replay(candidate)
    changed = e.build(work, raw, receipts)
    changed["static_audit_ready_score"] = 99
    atomic_json(candidate, changed)
    assert not e.replay(candidate)
    bad = deepcopy(work)
    bad["audits"]["8350"]["measurement_reference"]["sha256"] = "changed"
    private_raw = tmp_path / "bad"
    private_raw.mkdir()
    atomic_json(private_raw / "measurement.json", bad)
    atomic_json(candidate, e.build(bad, private_raw, receipts))
    assert not e.replay(candidate)
    assert not e.replay(tmp_path / "missing.json")
    bad = deepcopy(work)
    bad["authority"]["canonical_tasks_sha256"] = "changed"
    atomic_json(private_raw / "measurement.json", bad)
    atomic_json(candidate, e.build(bad, private_raw, receipts))
    assert not e.replay(candidate)
    log = tmp_path / "log"
    log.write_text("actual")
    atomic_json(
        candidate,
        e.build(work, raw, [dict(passed=True, stdout_path=str(log), stdout_sha256="bad")]),
    )
    assert not e.replay(candidate)


def test_manifest_and_coverage(tmp_path: Path, natural: tuple[dict[str, Any], Path]) -> None:
    """REQ-REPORT-8361: freeze only owned coverage and explicitly scoped consumers."""
    plan = e.manifest(tmp_path, tmp_path / "candidate.json")
    assert plan["no_full_repository_suite"]
    assert "E2E018_021" in plan["commands"][1]["name"]
    assert all(c["deadline_s"] <= 1200 for c in plan["commands"])
    work, raw = natural
    report = tmp_path / "coverage.json"
    atomic_json(report, dict(files={p: dict(summary=dict(missing_lines=1)) for p in e.OWNED}))
    bad = dict(work, owned_coverage_reference=e.reference(report))
    assert e.build(bad, raw, [dict(passed=True)])["required_checks_passed"] is False
    atomic_json(report, dict(files={p: dict(summary=dict(missing_lines=0)) for p in e.OWNED}))
    assert e.build(bad, raw, [dict(passed=True)])["required_checks_passed"]


def test_private_e2e018_frozen_authority(tmp_path: Path) -> None:
    """REQ-VERIFY-8361: E2E-018 reads immutable V720 authority rather than today's roadmap."""
    original = sha256_file(e.ROOT / "research-roadmap.yaml")
    authority = e.historical_authority(e.ROOT, tmp_path / "authority")
    assert authority["activated"] and len(authority["tasks"]) == 14
    assert any(t["id"] == static.TASK for t in authority["tasks"])
    assert sha256_file(e.ROOT / "research-roadmap.yaml") == original
    with pytest.raises(ValueError):
        e.historical_authority(tmp_path / "missing", tmp_path / "absent")


def test_real_cli_and_external_block(tmp_path: Path) -> None:
    """SCENARIO-REPORT-8361-CLI: real child publication preserves external absence."""
    output = tmp_path / (e.NAME + ".json")
    result = e.execution.check(
        dict(
            name="private_cli",
            argv=e.cli()
            + [
                "--private-run",
                "--date",
                "20261010",
                "--root",
                str(tmp_path / "absent"),
                "--output",
                str(output),
            ],
            deadline_s=180,
        ),
        tmp_path / "logs",
    )
    assert result["passed"]
    value = json.loads(output.read_bytes())
    assert value["verdict_class"] == "blocked"
    assert value["static_audit_ready_score"] == value["learning_audit_ready_score"] == 0
    assert any(g["observed"] is None for g in value["gate_check_summary"])
    assert e.replay(output)
    assert e.main(["--cold-replay", str(output)]) == 0
    assert e.main(["--cold-replay", str(tmp_path / "missing.json")]) == 1
    with pytest.raises(SystemExit):
        e.main(["--date", "wrong"])
    with pytest.raises(SystemExit):
        e.main(["--private-run", "--output", str(e.ROOT / "results" / (e.NAME + ".json"))])


def test_real_private_success_and_owned_failure(tmp_path: Path) -> None:
    """SCENARIO-REPORT-8361-CLI: successful mechanics and failed owned checks differ."""
    output = tmp_path / "success" / (e.NAME + ".json")
    assert e.main(["--private-run", "--output", str(output)]) == 0
    value = json.loads(output.read_bytes())
    assert value["verdict_class"] == "null" and value["required_checks_passed"]
    assert e.replay(output)
    failed = tmp_path / "failed" / (e.NAME + ".json")
    original = e.execution.manifest

    def failing(private: Path, candidate: Path) -> dict[str, Any]:
        plan = original(private, candidate)
        plan["commands"] = [
            dict(
                name="real_owned_failure",
                argv=[str(e.ROOT / ".venv/bin/python"), "-c", "raise SystemExit(3)"],
                deadline_s=10,
            )
        ]
        return plan

    with patch.object(e, "manifest", failing):
        assert e.main(["--root", str(tmp_path / "absent"), "--output", str(failed)]) == 0
    assert json.loads(failed.read_bytes())["verdict_class"] == "disqualified"


def test_historical_cli_dispatch(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-8361-REJECTIONS: explicit real replay rejects absent evidence."""
    assert e.main(["--historical-replay", "8350", "--candidate", str(tmp_path / "missing")]) == 1
    assert e.main(["--historical-replay", "8351", "--candidate", str(tmp_path / "missing")]) == 1


def test_owned_control_failure(tmp_path: Path) -> None:
    """REQ-REPORT-8361: an owned child error cannot become an external block."""
    with patch.object(e, "rejection_control", side_effect=ValueError("deliberate_owned_failure")):
        raw = tmp_path / "raw"
        work = e.measure(e.ROOT, raw)
    assert work["owned_failure"]
    assert work["failures"][0]["artifact_field"] == "owned_rejection_control"
    assert e.build(work, raw, [dict(passed=True)])["verdict_class"] == "disqualified"


@pytest.mark.parametrize("mutation", ["primitive", "audit", "coverage", "receipt", "control"])
def test_rehashed_primitive_and_child_controls(
    natural: tuple[dict[str, Any], Path], tmp_path: Path, mutation: str
) -> None:
    """SCENARIO-REPORT-8361-CLI: independent child evidence survives complete rehash attacks."""
    original, _ = natural
    raw = tmp_path / "raw"
    raw.mkdir()
    work = deepcopy(original)
    primitive_path = raw / "primitive_evidence.json"
    if mutation == "audit":
        candidate = raw / "invalid-audit.json"
        atomic_json(candidate, {})
        work["audits"]["8350"]["candidate_reference"] = e.reference(candidate)
    elif mutation == "coverage":
        work["controls"]["8350"]["target_line"] = 999999
    elif mutation == "receipt":
        work["controls"]["8350"]["valid_replay_receipt"]["exit_code"] = 1
    elif mutation == "control":
        work["controls"]["8350"]["passed"] = False
    primitive = dict(audits=work["audits"], controls=work["controls"])
    if mutation == "primitive":
        primitive["extra"] = "deliberate drift"
    atomic_json(primitive_path, primitive)
    work["raw_refs"] = [e.reference(primitive_path)]
    atomic_json(raw / "measurement.json", work)
    candidate = raw / "candidate.json"
    atomic_json(candidate, e.build(work, raw, [dict(passed=True)]))
    assert not e.replay(candidate)
