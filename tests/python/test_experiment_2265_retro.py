import json
from pathlib import Path

from scripts.experiment_2265_retro import (
    SCHEMA,
    build_retro,
    parse_conductor_log,
    write_retro,
)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _fixture_repo(tmp_path: Path) -> Path:
    log = "\n".join(
        [
            "| 2026-05-17 16:38 UTC | Milestone 2026.05.223 activated | OK | 13 tasks queued |",
            "| 2026-05-17 16:41 UTC | Phase 0: Archive .222 and activate .223 | OK | cache hit |",
            "| 2026-05-17 16:45 UTC | Phase 0: Fix Duplicate test_compositional_energy M | OK | 83 passed |",
            "| 2026-05-17 16:47 UTC | Phase 1: FST+ODAR+CASAL Real-Scale Live Generation | GATE_BLOCK | pretest false |",
            "| 2026-05-17 16:49 UTC | Phase 1: FST+ODAR+CASAL Real-Scale Live Generation | GATE_BLOCK | pretest false |",
            "| 2026-05-17 16:51 UTC | Phase 1: FST+ODAR+CASAL Real-Scale Live Generation | GATE_BLOCK | pretest false |",
            "| 2026-05-17 16:53 UTC | Phase 1: FR-11 FST Multi-Domain Retention Validati | GATE_BLOCK | upstream retired |",
            "| 2026-05-17 17:06 UTC | Phase 1: ODAR Real-Inference Routing Overhead Benc | OK | 81 passed |",
            "| 2026-05-17 17:08 UTC | Phase 1: FR-11 FST Multi-Domain Retention Validati | GATE_BLOCK | upstream retired |",
            "| 2026-05-17 17:08 UTC | Phase 2: KAN-CL n=256 Per-Knot Regularization Clea | GATE_BLOCK | pretest false |",
            "| 2026-05-17 17:10 UTC | Phase 1: FR-11 FST Multi-Domain Retention Validati | GATE_BLOCK | upstream retired |",
            "| 2026-05-17 17:10 UTC | Phase 2: KAN-CL n=256 Per-Knot Regularization Clea | GATE_BLOCK | pretest false |",
            "| 2026-05-17 17:12 UTC | Phase 2: KAN-CL n=256 Per-Knot Regularization Clea | GATE_BLOCK | pretest false |",
            "| 2026-05-17 17:14 UTC | Phase 2: KAN-CL n=256 + CASAL Joint Constraint Enf | GATE_BLOCK | upstream retired |",
            "| 2026-05-17 17:14 UTC | Phase 3: KV260 RTL Verilator Lint + Icarus Simulat | GATE_BLOCK | pretest false |",
            "| 2026-05-17 17:16 UTC | Phase 2: KAN-CL n=256 + CASAL Joint Constraint Enf | GATE_BLOCK | upstream retired |",
            "| 2026-05-17 17:16 UTC | Phase 3: KV260 RTL Verilator Lint + Icarus Simulat | GATE_BLOCK | pretest false |",
            "| 2026-05-17 17:18 UTC | Phase 2: KAN-CL n=256 + CASAL Joint Constraint Enf | GATE_BLOCK | upstream retired |",
            "| 2026-05-17 17:18 UTC | Phase 3: KV260 RTL Verilator Lint + Icarus Simulat | GATE_BLOCK | pretest false |",
            "| 2026-05-17 17:20 UTC | Phase 3: OSS-CAD-Suite Yosys Synthesis from Lint-P | GATE_BLOCK | upstream retired |",
            "| 2026-05-17 17:20 UTC | Phase 4: Adversarial Null-Space Probe on k=16 Ense | DOOMED_RERUN_BLOCK | prior failure |",
            "| 2026-05-17 17:22 UTC | Phase 3: OSS-CAD-Suite Yosys Synthesis from Lint-P | GATE_BLOCK | upstream retired |",
            "| 2026-05-17 17:22 UTC | Phase 4: Adversarial Null-Space Probe on k=16 Ense | DOOMED_RERUN_BLOCK | prior failure |",
            "| 2026-05-17 17:24 UTC | Phase 3: OSS-CAD-Suite Yosys Synthesis from Lint-P | GATE_BLOCK | upstream retired |",
            "| 2026-05-17 17:24 UTC | Phase 4: Adversarial Null-Space Probe on k=16 Ense | DOOMED_RERUN_BLOCK | prior failure |",
            "| 2026-05-17 17:37 UTC | Phase 5: ArXiv Post-.222 Research Sweep + Referenc | OK | cache hit |",
            "| 2026-05-17 17:39 UTC | Phase 6: Capstone E2E Real-Scale Live Generation ( | GATE_BLOCK | upstream retired |",
        ]
    )
    (tmp_path / "ops").mkdir()
    (tmp_path / "ops/conductor-log.md").write_text(log, encoding="utf-8")

    _write_json(
        tmp_path / "results/experiment_2254_pretest_fix.json",
        {"pretest_fixed": False, "remaining_error": "DualGPUExecutionResult"},
    )
    _write_json(
        tmp_path / "results/experiment_2255_fst_real_scale_live_gen.json",
        {
            "status": "blocked",
            "gate_check_summary": "exp2254-pretest-fix.pretest_fixed false",
        },
    )
    _write_json(
        tmp_path / "results/experiment_2258_kancl_n256_clean_reattempt.json",
        {
            "status": "blocked",
            "gate_check_summary": "exp2254-pretest-fix.pretest_fixed false",
        },
    )
    _write_json(
        tmp_path / "results/experiment_2260_kv260_rtl_clean_reattempt.json",
        {"status": "blocked"},
    )
    _write_json(
        tmp_path / "results/experiment_2257_odar_real_benchmark.json",
        {
            "compute_reduction_pct": 37.5,
            "routing_overhead_ms": 0.015469,
            "n_corpus": 100,
        },
    )
    _write_json(
        tmp_path / "results/experiment_2263_arxiv_sweep.json",
        {"n_new_papers_found": 8},
    )
    return tmp_path


def test_parse_conductor_log_reads_markdown_rows() -> None:
    """REQ-REPORT-2265: markdown conductor rows are parsed before metrics."""

    rows = parse_conductor_log(
        "| 2026-05-17 16:38 UTC | Milestone 2026.05.223 activated | OK | 13 tasks queued |\n"
    )

    assert rows[0].title == "Milestone 2026.05.223 activated"
    assert rows[0].status == "OK"


def test_build_retro_computes_required_v66_fields(tmp_path: Path) -> None:
    """REQ-REPORT-2265: artifact fields reflect .223 completion and gap closure."""

    repo = _fixture_repo(tmp_path)
    retro = build_retro(repo)

    assert retro["schema"] == SCHEMA
    assert retro["honest_verdict"].startswith("complete:")
    assert retro["total_wall_time_min"] == 61.0
    assert retro["n_experiments_completed"] == 4
    assert retro["n_gate_blocks"] == 19
    assert retro["n_compute_bound"] == 2
    assert retro["criteria_met"]["display"] == "5/13"
    assert retro["top_gaps_resolved_count"]["display"] == "0/3"
    assert all(not gap["resolved"] for gap in retro["top_gaps_resolved"])
    assert retro["next_milestone_speedup_target_pct"] == 35.0


def test_write_retro_outputs_deliverable_json(tmp_path: Path) -> None:
    """SCENARIO-REPORT-2265: generator writes the requested deliverable."""

    repo = _fixture_repo(tmp_path)
    out = write_retro(repo)
    payload = json.loads(out.read_text(encoding="utf-8"))

    assert out == repo / "results/experiment_2265_retro.json"
    assert payload["schema"] == SCHEMA
    assert payload["field_principles"]["top_gaps_resolved"].startswith("Records")
