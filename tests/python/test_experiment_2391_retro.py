import json
from pathlib import Path

from scripts.experiment_2391_retro import (
    SCHEMA,
    build_retro,
    parse_conductor_log,
    write_retro,
)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _roadmap_text() -> str:
    tasks = [
        ("exp2378-archive-and-activate", "Phase 0: Archive .231 and activate .232"),
        ("exp2379-halt-tier0j-latent-probe", "Phase 1: HALT Latent Probe Tier 0j"),
        ("exp2380-hive-4verifier-ensemble", "Phase 1: HIVE-Style 4-Verifier Ensemble"),
        ("exp2381-fregelogic-z3-neural-hybrid", "Phase 1: FregeLogic Z3+Neural Hybrid"),
        ("exp2382-fst-live-path-ab", "Phase 2: FST Live PATH A/B"),
        ("exp2383-fr11-nsvif-online-learning", "Phase 2: FR-11 NSVIF Online Learning"),
        ("exp2384-kv260-yosys-synthesis", "Phase 3: KV260 Yosys Synthesis"),
        ("exp2385-kinetic-langevin-vs-casal", "Phase 3: Kinetic Langevin Splitting"),
        ("exp2386-kac-rbf-vs-kancl", "Phase 3: KAC RBF vs KAN-CL Hard Domains"),
        ("exp2387-nsvif-smt-lib-policy", "Phase 4: NSVIF SMT-LIB Policy Formalization"),
        ("exp2388-phase1-ship-gate-check", "Phase 4: Phase 1 Ship Gate Audit"),
        ("exp2389-paperv6-results-table", "Phase 4: Paper-v6 Real-Data Results Table"),
        ("exp2390-capstone-v232", "Phase 5: Capstone v232"),
        ("exp2391-retro-v232", "Phase 6: Milestone 2026.05.232 Operational Retrospective"),
    ]
    lines = ["milestone: 2026.05.232", "tasks:"]
    for task_id, title in tasks:
        artifact_stem = task_id.split("-", 1)[0].replace("exp", "experiment_")
        lines.extend(
            [
                f"- id: {task_id}",
                f"  title: '{title}'",
                f"  deliverable: results/{artifact_stem}.json",
            ]
        )
    return "\n".join(lines) + "\n"


def _fixture_repo(tmp_path: Path, *, include_gate_artifacts: bool = False) -> Path:
    (tmp_path / "ops").mkdir()
    (tmp_path / "ops/conductor-log.md").write_text(
        "\n".join(
            [
                "| 2026-05-18 13:12 UTC | Milestone 2026.05.232 activated | OK | 14 tasks queued |",
                "| 2026-05-18 13:18 UTC | Phase 0: Archive .231 and activate .232 | FAIL | cli error |",
                "| 2026-05-18 13:25 UTC | Phase 1: HALT Latent Probe Tier 0j | FAIL | cli error |",
                "| 2026-05-18 13:31 UTC | Phase 1: HIVE-Style 4-Verifier Ensemble | FAIL | cli error |",
                "| 2026-05-18 13:37 UTC | Phase 1: FregeLogic Z3+Neural Hybrid | FAIL | cli error |",
                "| 2026-05-18 13:44 UTC | Phase 2: FST Live PATH A/B | FAIL | cli error |",
                "| 2026-05-18 13:50 UTC | Phase 2: FR-11 NSVIF Online Learning | FAIL | cli error |",
                "| 2026-05-18 13:56 UTC | Phase 3: KV260 Yosys Synthesis | FAIL | cli error |",
                "| 2026-05-18 14:03 UTC | Phase 3: Kinetic Langevin Splitting | FAIL | cli error |",
                "| 2026-05-18 14:09 UTC | Phase 3: KAC RBF vs KAN-CL Hard Domains | FAIL | cli error |",
                "| 2026-05-18 14:15 UTC | Phase 4: NSVIF SMT-LIB Policy Formalization | FAIL | cli error |",
                "| 2026-05-18 14:22 UTC | Phase 4: Phase 1 Ship Gate Audit | FAIL | cli error |",
                "| 2026-05-18 14:34 UTC | Phase 4: Paper-v6 Real-Data Results Table | OK | tests passed |",
                "| 2026-05-18 14:36 UTC | Phase 5: Capstone v232 | GATE_BLOCK | upstream retired |",
            ]
        ),
        encoding="utf-8",
    )
    (tmp_path / "research-roadmap.yaml").write_text(_roadmap_text(), encoding="utf-8")
    _write_json(
        tmp_path / "results/experiment_2389_paperv6_table.json",
        {
            "honest_verdict": "complete: n_paper_ready_results=1",
            "best_auroc_achieved": 0.6852,
            "hallscan_gap": 0.1948,
            "n_paper_ready_results": 1,
        },
    )
    if include_gate_artifacts:
        _write_json(
            tmp_path / "results/experiment_2380_hive_ensemble.json",
            {"ensemble_auroc_4verifier": 0.9},
        )
        _write_json(
            tmp_path / "results/experiment_2382_fst_live_path_ab.json",
            {"live_inference_completed": True},
        )
        _write_json(
            tmp_path / "results/experiment_2383_fr11_nsvif_online.json",
            {"fr11_nsvif_online_passed": True},
        )
        _write_json(
            tmp_path / "results/experiment_2384_kv260_yosys.json",
            {"synthesis_succeeded": True},
        )
        _write_json(
            tmp_path / "results/experiment_2388_phase1_ship_gate.json",
            {"phase1_ship_criteria_met": True},
        )
    return tmp_path


def test_parse_conductor_log_reads_markdown_rows() -> None:
    """REQ-REPORT-2391: parse conductor rows before computing .232 metrics."""

    rows = parse_conductor_log(
        "| 2026-05-18 13:12 UTC | Milestone 2026.05.232 activated | OK | queued |\n"
    )

    assert rows[0].title == "Milestone 2026.05.232 activated"
    assert rows[0].status == "OK"


def test_build_retro_counts_failures_gate_blocks_and_missing_artifacts(tmp_path: Path) -> None:
    """REQ-REPORT-2391: v75 retro records terminal counts and absent evidence honestly."""

    retro = build_retro(_fixture_repo(tmp_path))

    assert retro["schema"] == SCHEMA
    assert retro["honest_verdict"].startswith("complete:")
    assert retro["n_planned_tasks"] == 14
    assert retro["n_experiments_completed"] == 1
    assert retro["n_gate_blocks"] == 1
    assert retro["n_failed"] == 11
    assert retro["n_skipped"] == 1
    assert retro["total_wall_time_min"] == 84.0
    assert retro["fr11_satisfied"] is False
    assert retro["fst_live_path_ab_completed"] is False
    assert retro["kv260_yosys_synthesis_succeeded"] is False
    assert retro["phase1_ship_criteria_met"] is False
    assert retro["best_232_verifier_auroc"] == 0.6852
    assert retro["auroc_gap_to_hallscan_at_232_close"] == 0.1948
    assert retro["hive_gap_closed_vs_hallscan"] == 0.0
    assert retro["retro_complete"] is True
    assert "results/experiment_2382_fst_live_path_ab.json" in retro["missing_required_artifacts"]


def test_build_retro_uses_available_gate_artifacts(tmp_path: Path) -> None:
    """SCENARIO-REPORT-2391: available source artifacts drive booleans and AUROC gap."""

    retro = build_retro(_fixture_repo(tmp_path, include_gate_artifacts=True))

    assert retro["fr11_satisfied"] is True
    assert retro["fst_live_path_ab_completed"] is True
    assert retro["kv260_yosys_synthesis_succeeded"] is True
    assert retro["phase1_ship_criteria_met"] is True
    assert retro["best_232_verifier_auroc"] == 0.9
    assert retro["auroc_gap_to_hallscan_at_232_close"] == -0.02
    assert retro["hive_gap_closed_vs_hallscan"] == 0.02


def test_write_retro_writes_requested_deliverable(tmp_path: Path) -> None:
    """SCENARIO-REPORT-2391: generator writes results/experiment_2391_retro.json."""

    repo = _fixture_repo(tmp_path)
    out = write_retro(repo)
    payload = json.loads(out.read_text(encoding="utf-8"))

    assert out == repo / "results/experiment_2391_retro.json"
    assert payload["schema"] == SCHEMA
    assert payload["field_principles"]["retro_complete"] == "Must be true."
