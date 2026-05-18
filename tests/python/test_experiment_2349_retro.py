import json
from pathlib import Path

from scripts.experiment_2349_retro import (
    SCHEMA,
    build_retro,
    parse_conductor_log,
    write_retro,
)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _fixture_repo(tmp_path: Path, *, include_pretest_artifact: bool = False) -> Path:
    log_lines = [
        "| 2026-05-18 05:00 UTC | Milestone 2026.05.229 activated | OK | 14 tasks queued |",
        "| 2026-05-18 05:03 UTC | Phase 0: Archive .228 and activate .229 | OK | cache hit |",
        "| 2026-05-18 05:26 UTC | Phase 0: Fix 3 Remaining Pre-Test Failures (10th A | FAIL | timeout |",
        "| 2026-05-18 05:49 UTC | Phase 0: Fix 3 Remaining Pre-Test Failures (10th A | FAIL | timeout |",
        "| 2026-05-18 06:12 UTC | Phase 0: Fix 3 Remaining Pre-Test Failures (10th A | FAIL | timeout |",
        "| 2026-05-18 06:22 UTC | Phase 1: Semantic Energy Hallucination Detector - | OK | tests passed |",
        "| 2026-05-18 06:24 UTC | Phase 2: FST+ODAR+CASAL Real-Scale Live Generation | GATE_BLOCK | upstream retired |",
        "| 2026-05-18 06:24 UTC | Phase 2: FR-11 FST Multi-Domain Retention v6 | GATE_BLOCK | upstream retired |",
        "| 2026-05-18 06:24 UTC | Phase 2: KAN-CL n=256 Per-Knot Retention v8 | GATE_BLOCK | upstream retired |",
        "| 2026-05-18 06:24 UTC | Phase 2: NSVIF Neuro-Symbolic Z3 Extractor - First | GATE_BLOCK | upstream retired |",
        "| 2026-05-18 06:24 UTC | Phase 2: VERGE SMT Minimal Correction Subset Repai | GATE_BLOCK | upstream retired |",
        "| 2026-05-18 06:24 UTC | Phase 2: Eidoku CSP Tier 2.8 Gate - First Actual R | GATE_BLOCK | upstream retired |",
        "| 2026-05-18 06:24 UTC | Phase 2: Projected-Langevin vs CASAL Baseline v5 - | GATE_BLOCK | upstream retired |",
        "| 2026-05-18 06:24 UTC | Phase 3: KV260 RTL Verilator Lint + Icarus Simulat | GATE_BLOCK | upstream retired |",
        "| 2026-05-18 06:24 UTC | Phase 3: ML-Assisted Ising Machine Initialization | GATE_BLOCK | upstream retired |",
        "| 2026-05-18 06:24 UTC | Phase 4: Capstone E2E Live Generation (.229) - FST | GATE_BLOCK | upstream missing |",
        "| 2026-05-18 06:26 UTC | Phase 2: FST+ODAR+CASAL Real-Scale Live Generation | GATE_BLOCK | upstream retired |",
        "| 2026-05-18 06:26 UTC | Phase 2: FR-11 FST Multi-Domain Retention v6 | GATE_BLOCK | upstream retired |",
        "| 2026-05-18 06:26 UTC | Phase 2: KAN-CL n=256 Per-Knot Retention v8 | GATE_BLOCK | upstream retired |",
        "| 2026-05-18 06:26 UTC | Phase 2: NSVIF Neuro-Symbolic Z3 Extractor - First | GATE_BLOCK | upstream retired |",
        "| 2026-05-18 06:26 UTC | Phase 2: VERGE SMT Minimal Correction Subset Repai | GATE_BLOCK | upstream retired |",
        "| 2026-05-18 06:26 UTC | Phase 2: Eidoku CSP Tier 2.8 Gate - First Actual R | GATE_BLOCK | upstream retired |",
        "| 2026-05-18 06:26 UTC | Phase 2: Projected-Langevin vs CASAL Baseline v5 - | GATE_BLOCK | upstream retired |",
        "| 2026-05-18 06:26 UTC | Phase 3: KV260 RTL Verilator Lint + Icarus Simulat | GATE_BLOCK | upstream retired |",
        "| 2026-05-18 06:26 UTC | Phase 3: ML-Assisted Ising Machine Initialization | GATE_BLOCK | upstream retired |",
        "| 2026-05-18 06:26 UTC | Phase 4: Capstone E2E Live Generation (.229) - FST | GATE_BLOCK | upstream missing |",
        "| 2026-05-18 06:28 UTC | Phase 2: FST+ODAR+CASAL Real-Scale Live Generation | GATE_BLOCK | upstream retired |",
        "| 2026-05-18 06:28 UTC | Phase 2: FR-11 FST Multi-Domain Retention v6 | GATE_BLOCK | upstream retired |",
        "| 2026-05-18 06:28 UTC | Phase 2: KAN-CL n=256 Per-Knot Retention v8 | GATE_BLOCK | upstream retired |",
        "| 2026-05-18 06:28 UTC | Phase 2: NSVIF Neuro-Symbolic Z3 Extractor - First | GATE_BLOCK | upstream retired |",
        "| 2026-05-18 06:28 UTC | Phase 2: VERGE SMT Minimal Correction Subset Repai | GATE_BLOCK | upstream retired |",
        "| 2026-05-18 06:28 UTC | Phase 2: Eidoku CSP Tier 2.8 Gate - First Actual R | GATE_BLOCK | upstream retired |",
        "| 2026-05-18 06:28 UTC | Phase 2: Projected-Langevin vs CASAL Baseline v5 - | GATE_BLOCK | upstream retired |",
        "| 2026-05-18 06:28 UTC | Phase 3: KV260 RTL Verilator Lint + Icarus Simulat | GATE_BLOCK | upstream retired |",
        "| 2026-05-18 06:28 UTC | Phase 3: ML-Assisted Ising Machine Initialization | GATE_BLOCK | upstream retired |",
        "| 2026-05-18 06:28 UTC | Phase 4: Capstone E2E Live Generation (.229) - FST | GATE_BLOCK | upstream missing |",
    ]
    (tmp_path / "ops").mkdir()
    (tmp_path / "ops/conductor-log.md").write_text("\n".join(log_lines), encoding="utf-8")
    (tmp_path / "research-roadmap.yaml").write_text("milestone: 2026.05.229\n", encoding="utf-8")

    _write_json(
        tmp_path / "results/experiment_2336_archive.json",
        {"honest_verdict": "complete: blocked_roadmap_missing", "archive_ready": False},
    )
    if include_pretest_artifact:
        _write_json(
            tmp_path / "results/experiment_2337_pretest_fix.json",
            {
                "honest_verdict": "complete: partial_fix_0_of_3",
                "pretest_fixed": False,
                "operator_manual_commands": ["cd /repo", "pytest tests/python -x"],
            },
        )
    _write_json(
        tmp_path / "results/experiment_2338_semantic_energy.json",
        {
            "honest_verdict": "complete: Semantic Energy synthetic-logit prototype ran",
            "semantic_energy_validated": True,
            "semantic_energy_auroc": 1.0,
            "n_eval_examples": 100,
            "n_tests_passed": 3,
        },
    )
    _write_json(
        tmp_path / "results/experiment_2348_capstone_v229.json",
        {
            "schema": "blocked_gate_check_v1",
            "status": "blocked",
            "honest_verdict": "blocked_gate_check_failed",
            "gates_evaluated": [
                {"upstream": "exp2339-fst-live-gen-v9", "passed": False},
                {"upstream": "exp2341-kancl-n256-v8", "passed": False},
            ],
        },
    )
    return tmp_path


def test_parse_conductor_log_reads_rows() -> None:
    """REQ-REPORT-2349: parse conductor rows before computing milestone metrics."""

    rows = parse_conductor_log(
        "| 2026-05-18 05:00 UTC | Milestone 2026.05.229 activated | OK | 14 tasks queued |\n"
    )

    assert rows[0].title == "Milestone 2026.05.229 activated"
    assert rows[0].status == "OK"


def test_build_retro_computes_required_v72_fields(tmp_path: Path) -> None:
    """REQ-REPORT-2349: v72 artifact records .229 counts and gap outcomes."""

    retro = build_retro(_fixture_repo(tmp_path))

    assert retro["schema"] == SCHEMA
    assert retro["honest_verdict"].startswith("complete:")
    assert retro["total_wall_time_min"] == 88.0
    assert retro["n_experiments_completed"] == 2
    assert retro["n_experiments_completed_including_this_retro"] == 3
    assert retro["n_research_experiments_completed"] == 1
    assert retro["n_gate_blocks"] == 10
    assert retro["n_gate_block_attempts"] == 30
    assert retro["n_failed_attempts"] == 3
    assert retro["n_compute_bound"] == 0
    assert retro["criteria_met"]["display"] == "3/14"
    assert retro["criteria_met"]["primary_artifact_gate_display"] == "2/14"
    assert retro["top_gaps_resolved_count"]["display"] == "1/3"
    assert [gap["resolved"] for gap in retro["top_gaps_resolved"]] == [False, True, False]
    assert retro["pretest_cascade_status"]["status"] == "missing_deliverable_after_three_timeouts"
    assert retro["pretest_cascade_status"]["fully_resolved"] is False
    assert retro["ungated_tasks_completed"] == 3
    assert retro["structural_change_effectiveness"]["ungated_exp2338_prevented_empty_milestone"]
    assert retro["next_milestone_speedup_target_pct"] == 65.0


def test_missing_exp2337_uses_prompt_fallback_operator_commands(tmp_path: Path) -> None:
    """SCENARIO-REPORT-2349: missing Exp 2337 evidence still records operator commands."""

    retro = build_retro(_fixture_repo(tmp_path))
    pretest = retro["pretest_cascade_status"]

    assert pretest["deliverable_present"] is False
    assert "research-roadmap.yaml" in pretest["operator_manual_commands_source"]
    assert "tail -40" in "\n".join(pretest["operator_manual_commands"])
    assert "test_experiment_1692_potts_v2.py" in "\n".join(
        pretest["operator_targeted_pretest_commands"]
    )
    assert "before milestone .230 activation" in pretest["milestone_230_recommendation"]


def test_exp2337_artifact_commands_are_preserved_when_present(tmp_path: Path) -> None:
    """SCENARIO-REPORT-2349: Exp 2337 operator commands come from the artifact when available."""

    retro = build_retro(_fixture_repo(tmp_path, include_pretest_artifact=True))
    pretest = retro["pretest_cascade_status"]

    assert pretest["deliverable_present"] is True
    assert pretest["operator_manual_commands"] == ["cd /repo", "pytest tests/python -x"]
    assert pretest["operator_manual_commands_source"] == "results/experiment_2337_pretest_fix.json"


def test_capstone_fallback_and_write_path_are_recorded(tmp_path: Path) -> None:
    """SCENARIO-REPORT-2349: generator writes the requested v72 deliverable."""

    repo = _fixture_repo(tmp_path)
    out = write_retro(repo)
    payload = json.loads(out.read_text(encoding="utf-8"))

    assert out == repo / "results/experiment_2349_retro.json"
    assert payload["schema"] == SCHEMA
    assert any(
        item["path"] == "results/experiment_2348_capstone.json"
        and item["fallback_used"] == "results/experiment_2348_capstone_v229.json"
        for item in payload["requested_artifact_status"]
    )
    assert payload["field_principles"]["ungated_tasks_completed"].startswith("Records how many")
