import os
import json
import tempfile
from carnot.reporting.experiment_1853_retro import generate_retro

def test_generate_retro():
    """
    Test that REQ-REPORT-1853 and SCENARIO-REPORT-1853 are satisfied.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        paths = []
        p_1849 = os.path.join(tmpdir, "experiment_1849_cocom_pruning.json")
        with open(p_1849, "w") as f:
            json.dump({
                "experiment_id": "1849",
                "status": "complete",
                "honest_verdict": "cocom_pruning_implemented",
            }, f)
        paths.append(p_1849)

        p_1850 = os.path.join(tmpdir, "experiment_1850_thrml_parity_n128.json")
        with open(p_1850, "w") as f:
            json.dump({
              "acceptance_gate_passed": False,
              "honest_verdict": "complete: thrml_carnot_parity_n128_gate_failed_kl_0.28"
            }, f)
        paths.append(p_1850)

        p_1851 = os.path.join(tmpdir, "experiment_1851_nla_probe.json")
        with open(p_1851, "w") as f:
            json.dump({
              "acceptance_gate_passed": True,
              "honest_verdict": "complete: nla_probe_prototype_tpr_lift_0.98_orthogonal_coverage_10",
            }, f)
        paths.append(p_1851)

        p_1852 = os.path.join(tmpdir, "experiment_1852_findings_audit.json")
        with open(p_1852, "w") as f:
            json.dump({
                "acceptance_gate_passed": True,
                "honest_verdict": "complete: findings_audit_surfaced_80_underclaimed_results"
            }, f)
        paths.append(p_1852)

        out_path = os.path.join(tmpdir, "experiment_1853_retro.json")
        generate_retro(paths, out_path)

        assert os.path.exists(out_path)
        with open(out_path, "r") as f:
            result = json.load(f)
        
        assert result["schema"] == "carnot.milestone_research_retro.v1"
        assert result["milestone"] == "2026.05.144"
        assert result["gates_passed_count"] == 2
        assert result["gates_failed_count"] == 1
        assert "complete:" in result["honest_verdict"] or "success:" in result["honest_verdict"]
        assert len(result["tasks_summary"]) == 4
        assert len(result["paper_v6_carryforward_items"]) > 0

def test_missing_files():
    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = os.path.join(tmpdir, "experiment_1853_retro.json")
        generate_retro([], out_path)
        with open(out_path, "r") as f:
            result = json.load(f)
        assert result["gates_passed_count"] == 0
        assert result["gates_failed_count"] == 0
        assert len(result["tasks_summary"]) == 0
