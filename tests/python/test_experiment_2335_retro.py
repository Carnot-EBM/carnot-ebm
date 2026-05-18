import json
from pathlib import Path

from scripts.experiment_2335_retro import (
    SCHEMA,
    build_retro,
    parse_conductor_log,
    write_retro,
)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _fixture_repo(tmp_path: Path) -> Path:
    log_lines = [
        "| 2026-05-18 02:59 UTC | Milestone 2026.05.228 activated | OK | 14 tasks queued |",
        "| 2026-05-18 03:03 UTC | Phase 0: Archive .227 and activate .228 | OK | 81 passed, 1 warning in 1.88s |",
        "| 2026-05-18 03:26 UTC | Phase 0: Fix 3 Remaining Pre-Test Failures (9th At | FAIL | timeout |",
        "| 2026-05-18 03:50 UTC | Phase 0: Fix 3 Remaining Pre-Test Failures (9th At | FAIL | timeout |",
        "| 2026-05-18 04:13 UTC | Phase 0: Fix 3 Remaining Pre-Test Failures (9th At | FAIL | timeout |",
        "| 2026-05-18 04:15 UTC | Phase 1: FST+ODAR+CASAL Real-Scale Live Generation | GATE_BLOCK | upstream retired |",
        "| 2026-05-18 04:15 UTC | Phase 1: FR-11 FST Multi-Domain Retention v5 | GATE_BLOCK | upstream retired |",
        "| 2026-05-18 04:15 UTC | Phase 1: KAN-CL n=256 Per-Knot Retention v7 | GATE_BLOCK | upstream retired |",
        "| 2026-05-18 04:15 UTC | Phase 2: NSVIF Neuro-Symbolic Z3 Extractor - First | GATE_BLOCK | upstream retired |",
        "| 2026-05-18 04:15 UTC | Phase 2: VERGE SMT Minimal Correction Subset Repai | GATE_BLOCK | upstream retired |",
        "| 2026-05-18 04:15 UTC | Phase 2: Eidoku CSP Tier 2.8 Gate - First Actual R | GATE_BLOCK | upstream retired |",
        "| 2026-05-18 04:15 UTC | Phase 2: Projected-Langevin vs CASAL Baseline v4 - | GATE_BLOCK | upstream retired |",
        "| 2026-05-18 04:15 UTC | Phase 3: KV260 RTL Verilator Lint + Icarus Simulat | GATE_BLOCK | upstream retired |",
        "| 2026-05-18 04:15 UTC | Phase 3: ML-Assisted Ising Machine Initialization | GATE_BLOCK | upstream retired |",
        "| 2026-05-18 04:15 UTC | Phase 3: Adversarial Null-Space Probe on k=16 Ense | GATE_BLOCK | upstream retired |",
        "| 2026-05-18 04:16 UTC | Phase 4: Capstone E2E Live Generation (.228) - FST | GATE_BLOCK | upstream missing |",
        "| 2026-05-18 04:18 UTC | Phase 1: FST+ODAR+CASAL Real-Scale Live Generation | GATE_BLOCK | upstream retired |",
        "| 2026-05-18 04:18 UTC | Phase 1: FR-11 FST Multi-Domain Retention v5 | GATE_BLOCK | upstream retired |",
        "| 2026-05-18 04:18 UTC | Phase 1: KAN-CL n=256 Per-Knot Retention v7 | GATE_BLOCK | upstream retired |",
        "| 2026-05-18 04:18 UTC | Phase 2: NSVIF Neuro-Symbolic Z3 Extractor - First | GATE_BLOCK | upstream retired |",
        "| 2026-05-18 04:18 UTC | Phase 2: VERGE SMT Minimal Correction Subset Repai | GATE_BLOCK | upstream retired |",
        "| 2026-05-18 04:18 UTC | Phase 2: Eidoku CSP Tier 2.8 Gate - First Actual R | GATE_BLOCK | upstream retired |",
        "| 2026-05-18 04:18 UTC | Phase 2: Projected-Langevin vs CASAL Baseline v4 - | GATE_BLOCK | upstream retired |",
        "| 2026-05-18 04:18 UTC | Phase 3: KV260 RTL Verilator Lint + Icarus Simulat | GATE_BLOCK | upstream retired |",
        "| 2026-05-18 04:18 UTC | Phase 3: ML-Assisted Ising Machine Initialization | GATE_BLOCK | upstream retired |",
        "| 2026-05-18 04:18 UTC | Phase 3: Adversarial Null-Space Probe on k=16 Ense | GATE_BLOCK | upstream retired |",
        "| 2026-05-18 04:18 UTC | Phase 4: Capstone E2E Live Generation (.228) - FST | GATE_BLOCK | upstream missing |",
        "| 2026-05-18 04:20 UTC | Phase 1: FST+ODAR+CASAL Real-Scale Live Generation | GATE_BLOCK | upstream retired |",
        "| 2026-05-18 04:20 UTC | Phase 1: FR-11 FST Multi-Domain Retention v5 | GATE_BLOCK | upstream retired |",
        "| 2026-05-18 04:20 UTC | Phase 1: KAN-CL n=256 Per-Knot Retention v7 | GATE_BLOCK | upstream retired |",
        "| 2026-05-18 04:20 UTC | Phase 2: NSVIF Neuro-Symbolic Z3 Extractor - First | GATE_BLOCK | upstream retired |",
        "| 2026-05-18 04:20 UTC | Phase 2: VERGE SMT Minimal Correction Subset Repai | GATE_BLOCK | upstream retired |",
        "| 2026-05-18 04:20 UTC | Phase 2: Eidoku CSP Tier 2.8 Gate - First Actual R | GATE_BLOCK | upstream retired |",
        "| 2026-05-18 04:20 UTC | Phase 2: Projected-Langevin vs CASAL Baseline v4 - | GATE_BLOCK | upstream retired |",
        "| 2026-05-18 04:20 UTC | Phase 3: KV260 RTL Verilator Lint + Icarus Simulat | GATE_BLOCK | upstream retired |",
        "| 2026-05-18 04:20 UTC | Phase 3: ML-Assisted Ising Machine Initialization | GATE_BLOCK | upstream retired |",
        "| 2026-05-18 04:20 UTC | Phase 3: Adversarial Null-Space Probe on k=16 Ense | GATE_BLOCK | upstream retired |",
        "| 2026-05-18 04:20 UTC | Phase 4: Capstone E2E Live Generation (.228) - FST | GATE_BLOCK | upstream missing |",
    ]
    (tmp_path / "ops").mkdir()
    (tmp_path / "ops/conductor-log.md").write_text("\n".join(log_lines), encoding="utf-8")

    _write_json(
        tmp_path / "results/experiment_2309_pretest_fix.json",
        {
            "pretest_fixed": False,
            "remaining_preexisting_failures": [
                {
                    "test": "tests/python/test_experiment_1692_potts_v2.py::test_experiment_1692_potts_v2_artifact"
                }
            ],
        },
    )
    _write_json(
        tmp_path / "results/experiment_2322_archive.json",
        {"honest_verdict": "complete: blocked_roadmap_missing", "archive_ready": False},
    )
    _write_json(
        tmp_path / "results/experiment_2334_capstone_v228.json",
        {
            "schema": "blocked_gate_check_v1",
            "status": "blocked",
            "honest_verdict": "blocked_gate_check_failed",
            "gates_evaluated": [
                {"upstream": "exp2324-fst-live-gen-v8", "passed": False},
                {"upstream": "exp2326-kancl-n256-v7", "passed": False},
            ],
        },
    )
    return tmp_path


def test_parse_conductor_log_reads_rows() -> None:
    """REQ-REPORT-2335: parse conductor rows before computing milestone metrics."""

    rows = parse_conductor_log(
        "| 2026-05-18 02:59 UTC | Milestone 2026.05.228 activated | OK | 14 tasks queued |\n"
    )

    assert rows[0].title == "Milestone 2026.05.228 activated"
    assert rows[0].status == "OK"


def test_build_retro_computes_required_v71_fields(tmp_path: Path) -> None:
    """REQ-REPORT-2335: v71 artifact records .228 counts and unresolved gaps."""

    retro = build_retro(_fixture_repo(tmp_path))

    assert retro["schema"] == SCHEMA
    assert retro["honest_verdict"].startswith("complete:")
    assert retro["total_wall_time_min"] == 81.0
    assert retro["n_experiments_completed"] == 1
    assert retro["n_experiments_completed_including_this_retro"] == 2
    assert retro["n_gate_blocks"] == 11
    assert retro["n_gate_block_attempts"] == 33
    assert retro["n_failed_attempts"] == 3
    assert retro["n_compute_bound"] == 0
    assert retro["criteria_met"]["display"] == "2/14"
    assert retro["criteria_met"]["primary_artifact_gate_display"] == "1/14"
    assert retro["top_gaps_resolved_count"]["display"] == "0/3"
    assert all(not gap["resolved"] for gap in retro["top_gaps_resolved"])
    assert retro["pretest_cascade_status"]["status"] == "missing_deliverable_after_three_timeouts"
    assert retro["pretest_cascade_status"]["fully_resolved"] is False
    assert retro["pretest_cascade_status"]["manual_operator_intervention_required"]
    assert retro["next_milestone_speedup_target_pct"] == 55.0


def test_failed_exp2323_escalation_includes_exp2309_and_exp2323_commands(tmp_path: Path) -> None:
    """SCENARIO-REPORT-2335: missing Exp 2323 evidence triggers operator commands."""

    retro = build_retro(_fixture_repo(tmp_path))
    pretest = retro["pretest_cascade_status"]
    all_commands = "\n".join(
        pretest["exp2309_pytest_commands"] + pretest["exp2323_pytest_commands"]
    )

    assert "test_experiment_1347_thrml_compatibility_parity_audit.py" in all_commands
    assert "test_experiment_1182_paper_v5_medium_low_issues_11_18.py" in all_commands
    assert (
        "test_experiment_1692_potts_v2.py::test_experiment_1692_potts_v2_artifact" in all_commands
    )
    assert "test_experiment_390_gpu_preflight.py::TestRunGpuPreflight" in all_commands
    assert "test_experiment_294_gpu_baseline_apple.py::TestBaselineAccuracyBounds" in all_commands
    assert "-p no:xdist" in all_commands
    assert "without xdist" in pretest["conductor_gate_recommendation"]


def test_capstone_fallback_and_deferred_gguf_check_are_recorded(tmp_path: Path) -> None:
    """SCENARIO-REPORT-2335: missing requested capstone path records fallback evidence."""

    retro = build_retro(_fixture_repo(tmp_path))

    assert "results/experiment_2334_capstone_v228.json" in retro["source_artifacts"]
    assert any(
        item["path"] == "results/experiment_2334_capstone.json"
        and item["fallback_used"] == "results/experiment_2334_capstone_v228.json"
        for item in retro["requested_artifact_status"]
    )
    assert retro["gguf_availability_status"]["evaluated"] is False
    assert "gemma-4-26B" in retro["gguf_availability_status"]["deferred_precondition_command"]


def test_write_retro_outputs_requested_deliverable(tmp_path: Path) -> None:
    """SCENARIO-REPORT-2335: generator writes results/experiment_2335_retro.json."""

    repo = _fixture_repo(tmp_path)
    out = write_retro(repo)
    payload = json.loads(out.read_text(encoding="utf-8"))

    assert out == repo / "results/experiment_2335_retro.json"
    assert payload["schema"] == SCHEMA
    assert payload["field_principles"]["pretest_cascade_status"].startswith("Explicit field")
