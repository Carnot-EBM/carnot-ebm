import json
from pathlib import Path

from scripts.experiment_2321_retro import (
    SCHEMA,
    build_retro,
    parse_conductor_log,
    write_retro,
)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _fixture_repo(tmp_path: Path, *, include_pretest_artifact: bool = True) -> Path:
    log_lines = [
        "| 2026-05-17 22:43 UTC | Milestone 2026.05.227 activated | OK | 14 tasks queued |",
        "| 2026-05-17 22:48 UTC | Phase 0: Archive .226 and activate .227 | OK | cache hit |",
        "| 2026-05-17 23:01 UTC | Phase 0: Fix 2 Remaining Pre-Test Failures (7th At | FAIL | artifact_not_updated_past_bootstrap |",
        "| 2026-05-17 23:23 UTC | Phase 0: Fix 2 Remaining Pre-Test Failures (7th At | FAIL | timeout |",
        "| 2026-05-17 23:46 UTC | Phase 0: Fix 2 Remaining Pre-Test Failures (7th At | FAIL | timeout |",
        "| 2026-05-17 23:52 UTC | Phase 1: FST+ODAR+CASAL Real-Scale Live Generation | GATE_BLOCK | upstream retired |",
        "| 2026-05-17 23:52 UTC | Phase 1: FR-11 FST Multi-Domain Retention v4 | GATE_BLOCK | upstream retired |",
        "| 2026-05-17 23:52 UTC | Phase 1: KAN-CL n=256 Per-Knot Retention v6 | GATE_BLOCK | upstream retired |",
        "| 2026-05-17 23:52 UTC | Phase 2: NSVIF Neuro-Symbolic Z3 Extractor - First | GATE_BLOCK | upstream retired |",
        "| 2026-05-17 23:52 UTC | Phase 2: VERGE SMT Minimal Correction Subset Repai | GATE_BLOCK | upstream retired |",
        "| 2026-05-17 23:52 UTC | Phase 2: Eidoku CSP Tier 2.8 Gate - First Actual R | GATE_BLOCK | upstream retired |",
        "| 2026-05-17 23:52 UTC | Phase 2: Projected-Langevin vs CASAL Baseline v3 - | GATE_BLOCK | upstream retired |",
        "| 2026-05-17 23:52 UTC | Phase 3: KV260 RTL Verilator Lint + Icarus Simulat | GATE_BLOCK | upstream retired |",
        "| 2026-05-17 23:52 UTC | Phase 3: ML-Assisted Ising Machine Initialization | GATE_BLOCK | upstream retired |",
        "| 2026-05-17 23:52 UTC | Phase 3: Adversarial Null-Space Probe on k=16 Ense | GATE_BLOCK | upstream retired |",
        "| 2026-05-17 23:52 UTC | Phase 4: Capstone E2E Live Generation (.227) - FST | GATE_BLOCK | upstream retired |",
    ]
    (tmp_path / "ops").mkdir()
    (tmp_path / "ops/conductor-log.md").write_text("\n".join(log_lines), encoding="utf-8")

    _write_json(
        tmp_path / "results/experiment_2308_archive.json",
        {"honest_verdict": "complete: archive_ready", "archive_ready": True},
    )
    if include_pretest_artifact:
        _write_json(
            tmp_path / "results/experiment_2309_pretest_fix.json",
            {
                "honest_verdict": "complete: named_tests_fixed_suite_has_other_preexisting_failures",
                "pretest_fixed": False,
                "full_pretest_errors": 1,
                "full_pretest_failures": 2,
                "tests_fixed": [
                    "tests/python/test_experiment_1347_thrml_compatibility_parity_audit.py::test_req_sample_041_probe_reports_direct_import_success_without_version",
                    "tests/python/test_experiment_1182_paper_v5_medium_low_issues_11_18.py::TestIssue11ThroughIssue15::test_issue_14_soskan_aurocs_have_corpus_and_n",
                ],
                "remaining_preexisting_failures": [{"test": "one"}],
            },
        )
    _write_json(
        tmp_path / "results/experiment_2320_capstone_v227.json",
        {
            "schema": "blocked_gate_check_v1",
            "honest_verdict": "blocked_gate_check_failed",
            "status": "blocked",
            "gates_evaluated": [
                {"upstream": "exp2310-fst-live-gen-v7", "passed": False},
                {"upstream": "exp2312-kancl-n256-v6", "passed": False},
            ],
        },
    )
    return tmp_path


def test_parse_conductor_log_reads_rows() -> None:
    """REQ-REPORT-2321: parse conductor rows before computing milestone metrics."""

    rows = parse_conductor_log(
        "| 2026-05-17 22:43 UTC | Milestone 2026.05.227 activated | OK | 14 tasks queued |\n"
    )

    assert rows[0].title == "Milestone 2026.05.227 activated"
    assert rows[0].status == "OK"


def test_build_retro_computes_required_v70_fields(tmp_path: Path) -> None:
    """REQ-REPORT-2321: v70 artifact records .227 counts and unresolved gaps."""

    retro = build_retro(_fixture_repo(tmp_path))

    assert retro["schema"] == SCHEMA
    assert retro["honest_verdict"].startswith("complete:")
    assert retro["total_wall_time_min"] == 69.0
    assert retro["n_experiments_completed"] == 2
    assert retro["n_gate_blocks"] == 11
    assert retro["n_compute_bound"] == 0
    assert retro["criteria_met"]["display"] == "2/14"
    assert retro["top_gaps_resolved_count"]["display"] == "0/3"
    assert all(not gap["resolved"] for gap in retro["top_gaps_resolved"])
    assert retro["pretest_cascade_status"]["fully_resolved"] is False
    assert retro["pretest_cascade_status"]["manual_operator_intervention_required"]
    assert any(
        "test_experiment_1347_thrml_compatibility_parity_audit.py" in command
        for command in retro["pretest_cascade_status"]["operator_pytest_commands"]
    )
    assert any(
        "test_experiment_1182_paper_v5_medium_low_issues_11_18.py" in command
        for command in retro["pretest_cascade_status"]["operator_pytest_commands"]
    )
    assert retro["next_milestone_speedup_target_pct"] == 45.0


def test_missing_pretest_artifact_still_escalates(tmp_path: Path) -> None:
    """SCENARIO-REPORT-2321: missing Exp 2309 evidence counts as unresolved."""

    retro = build_retro(_fixture_repo(tmp_path, include_pretest_artifact=False))

    assert retro["pretest_cascade_status"]["deliverable_present"] is False
    assert retro["pretest_cascade_status"]["fully_resolved"] is False
    assert retro["pretest_cascade_status"]["manual_operator_intervention_required"]


def test_write_retro_outputs_requested_deliverable(tmp_path: Path) -> None:
    """SCENARIO-REPORT-2321: generator writes results/experiment_2321_retro.json."""

    repo = _fixture_repo(tmp_path)
    out = write_retro(repo)
    payload = json.loads(out.read_text(encoding="utf-8"))

    assert out == repo / "results/experiment_2321_retro.json"
    assert payload["schema"] == SCHEMA
    assert payload["field_principles"]["pretest_cascade_status"].startswith("Explicit field")
