"""Tests for the Exp 1267 milestone .98 retrospective.

Spec: REQ-REPORT-022, SCENARIO-REPORT-019.
"""

from __future__ import annotations

import json
from pathlib import Path

from carnot.reporting.milestone_retro_98 import CRITERION_NAMES, build_artifact, run


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _scenario_sources() -> dict[int, dict[str, object]]:
    return {
        1255: {"retro_complete": False},
        1256: {"orthogonality_matrix_computed": True},
        1257: {"critical_issues_fixed": 0},
        1259: {"honest_verdict": "in_progress"},
        1260: {"phase5d_gates_passed": 0},
        1261: {"cartridge_shipped": False},
        1262: {"cartridge_shipped": False},
        1263: {"gaming_defense_measured": False},
        1264: {"tss_instrumented": True},
        1265: {"diffutruth_comparison_measured": True},
        1266: {"quantkan_3bit_auroc": 0.9801},
    }


def test_scenario_report_019_counts_milestone_98_source_criteria() -> None:
    """SCENARIO-REPORT-019: Exp1267 reports .98 5/13 from source fields."""

    artifact = build_artifact(_scenario_sources())

    assert tuple(artifact["criteria_results"]) == CRITERION_NAMES
    assert artifact["criteria_results"] == {
        "retro_97_complete": False,
        "orthogonality_matrix_measured": True,
        "critical_issues_fixed_5_of_5": False,
        "arxiv_v6_submitted": False,
        "grpo_v7_honest_result": False,
        "phase5d_4_gates_measured": False,
        "kakuro_cartridge_shipped": False,
        "masyu_cartridge_shipped": False,
        "gaming_defense_measured": False,
        "q11_tss_instrumented": True,
        "diffutruth_comparison_measured": True,
        "quantkan_3bit_auroc_measured": True,
        "retro_98_complete": True,
    }
    assert artifact["criteria_met"] == 5
    assert artifact["criteria_total"] == 13
    assert "orthogonality" in artifact["findings_summary"]
    assert "QuantKAN" in artifact["findings_summary"]
    assert len(artifact["key_carry_forwards"]) == 5
    assert len(artifact["top_successes"]) == 3
    assert len(artifact["top_gaps"]) == 3
    assert artifact["retro_complete"] is True
    assert artifact["honest_verdict"] == "milestone_98_5_of_13_criteria_met"


def test_req_report_022_run_writes_required_schema(tmp_path: Path) -> None:
    """REQ-REPORT-022: run loads result JSON files and writes the final artifact."""

    results_dir = tmp_path / "results"
    out_path = results_dir / "experiment_1267_milestone_retro_98.json"
    filenames = {
        1255: "experiment_1255_combined_retro_95_96_97.json",
        1256: "experiment_1256_verifier_orthogonality_audit_v3.json",
        1257: "experiment_1257_paper_v6_critical_issues_fix.json",
        1259: "experiment_1259_grpo_v7_progrs_vps.json",
        1260: "experiment_1260_phase5d_intermediate_scale_v3.json",
        1261: "experiment_1261_wopr_kakuro_v3.json",
        1262: "experiment_1262_wopr_masyu_v2.json",
        1263: "experiment_1263_gaming_verifiers_defense_v4.json",
        1264: "experiment_1264_q11_tss_instrumentation_v2.json",
        1265: "experiment_1265_diffutruth_vs_carnot_baseline.json",
        1266: "experiment_1266_quantkan_3bit_lut_kan.json",
    }
    for exp_id, payload in _scenario_sources().items():
        _write_json(results_dir / filenames[exp_id], payload)

    artifact = run(results_dir=results_dir, out_path=out_path)
    written = json.loads(out_path.read_text(encoding="utf-8"))

    assert artifact == written
    assert written["experiment"] == "1267_milestone_retro_98"
    assert written["schema"] == "milestone_retro_v3"
    assert written["milestone"] == "2026.04.98"
    assert written["status"] == "complete"
    assert written["criteria_met"] == 5
    assert written["criteria_total"] == 13
