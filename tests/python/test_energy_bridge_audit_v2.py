"""Tests for the Exp 1306 EBT/ARM/EBM-CoT energy bridge audit v2.

Spec: REQ-REPORT-026, SCENARIO-REPORT-026.
"""

from __future__ import annotations

import json
from pathlib import Path

from carnot.reporting import energy_bridge_audit_v2 as exp


def _prior_blocked() -> dict[str, object]:
    return {
        "experiment": 1293,
        "status": "blocked",
        "honest_verdict": "blocked_gate_check_failed",
        "gate_check_summary": "prior_failures field is missing or incomplete",
        "blocked_at_layer": "conductor_pre_gate",
    }


def _retro_100() -> dict[str, object]:
    return {
        "experiment": "1295_milestone_retro_100",
        "status": "complete",
        "continuous_repair_summary": {
            "energy_bridge_status": "blocked",
            "hardnetpp_nonlinear_repair_viable": True,
            "feasibility_channel_auc": 0.6604651162790698,
        },
    }


def _references_text() -> str:
    return """
## 2026-05-05 Planning Sweep (Milestone 2026.04.101)
### FALCON: Hard Constraints + Feasibility Repair for LLM CO
FALCON combines grammar-constrained decoding, semantic feasibility repair,
and adaptive Best-of-N sampling.
### Parallel p-bit Ising Performance-Cost Landscape
Records synchronous/asynchronous p-bit update dynamics, delay, DAC precision,
and time-multiplexed p-bit reuse.
### Extropic TSU and Logical Kona Status Check
Extropic lists XTR-0 as a 2025 research platform and Z1 as early access in
2026. Logical describes Kona as an EBM layer for constraint enforcement.
### EBM-CoT: Energy-Based Chain-of-Thought Sequence Optimization
Uses energy-based optimization over chain-of-thought sequences.
### Semantic Scholar EBT Citation Check
Semantic Scholar currently lists 14 citations for Energy-Based Transformers
are Scalable Learners and Thinkers.
### EBT and ARM-EBM Theory Became Stronger Signals
EBT reframes prediction as energy minimization over candidate outputs.
ARM-EBM provides a bridge from next-token models to sequence-level energy.
"""


def _architecture_text() -> str:
    return """
The verification pipeline uses SC-Energy, SymCodeVerifier, and Ising tiers.
Carnot's hardware path keeps KV260 FPGA, Extropic XTR-0 TSU, p-bit machines,
and photonic samplers as portability targets.
"""


def _research_program_text() -> str:
    return """
What Remains Valid: constraint verification infrastructure, energy computation
speed, parallel Ising sampler, and the FPGA/TSU hardware path. Bridge to Kona is
long-term and depends on working extraction.
"""


def test_req_report_026_writes_in_progress_artifact_first(tmp_path: Path) -> None:
    """REQ-REPORT-026: the Exp 1306 audit has a durable in-progress state."""

    out_path = tmp_path / "results" / "experiment_1306_ebt_arm_ebm_cot_energy_bridge_audit_v2.json"

    artifact = exp.write_in_progress_artifact(
        out_path,
        project_root="/home/ianblenke/github.com/ianblenke/carnot",
    )

    written = json.loads(out_path.read_text(encoding="utf-8"))
    assert written == artifact
    assert written["status"] == "in_progress"
    assert written["energy_bridge_completed"] is False
    assert written["metadata"]["run_date"] == "20260505"
    assert written["metadata"]["project_root"] == "/home/ianblenke/github.com/ianblenke/carnot"
    assert written["honest_verdict"] == "in_progress"


def test_scenario_report_026_builds_complete_local_bridge_audit() -> None:
    """SCENARIO-REPORT-026: local notes complete the bridge without new hardware."""

    artifact = exp.build_artifact(
        exp1293_payload=_prior_blocked(),
        exp1295_payload=_retro_100(),
        references_text=_references_text(),
        architecture_text=_architecture_text(),
        research_program_text=_research_program_text(),
        project_root="/home/ianblenke/github.com/ianblenke/carnot",
    )

    exp.validate_artifact(artifact)
    assert artifact["status"] == "complete"
    assert artifact["energy_bridge_completed"] is True
    assert artifact["ebt_citation_count_checked"]["citation_count"] == 14
    assert artifact["ebt_citation_count_checked"]["network_required"] is False
    assert "Implemented locally" in artifact["arm_ebm_alignment_note"]
    assert "strategic gap" in artifact["arm_ebm_alignment_note"].lower()
    assert "sequence" in artifact["ebm_cot_sequence_energy_note"].lower()
    assert "future sampler context" in artifact["extropic_kona_status_checked"]
    assert "p-bit update dynamics" in artifact["hardware_sampler_context_recorded"]
    assert "FALCON" in artifact["strategic_context_only"][0]
    assert artifact["prior_blocker"]["blocked_at_layer"] == "conductor_pre_gate"
    assert artifact["honest_verdict"] == exp.HONEST_VERDICT


def test_req_report_026_run_loads_sources_and_writes_artifact(tmp_path: Path) -> None:
    """REQ-REPORT-026: run reads local files and writes the final schema."""

    results_dir = tmp_path / "results"
    results_dir.mkdir()
    exp1293_path = results_dir / "experiment_1293_ebt_arm_ebm_cot_energy_bridge_audit.json"
    exp1295_path = results_dir / "experiment_1295_milestone_retro_100.json"
    references_path = tmp_path / "research-references.md"
    architecture_path = tmp_path / "_bmad" / "architecture.md"
    research_program_path = tmp_path / "research-program.md"
    out_path = results_dir / "experiment_1306_ebt_arm_ebm_cot_energy_bridge_audit_v2.json"

    exp1293_path.write_text(json.dumps(_prior_blocked()), encoding="utf-8")
    exp1295_path.write_text(json.dumps(_retro_100()), encoding="utf-8")
    references_path.write_text(_references_text(), encoding="utf-8")
    architecture_path.parent.mkdir()
    architecture_path.write_text(_architecture_text(), encoding="utf-8")
    research_program_path.write_text(_research_program_text(), encoding="utf-8")

    artifact = exp.run(
        out_path=out_path,
        exp1293_path=exp1293_path,
        exp1295_path=exp1295_path,
        references_path=references_path,
        architecture_path=architecture_path,
        research_program_path=research_program_path,
        project_root="/home/ianblenke/github.com/ianblenke/carnot",
    )

    written = json.loads(out_path.read_text(encoding="utf-8"))
    assert written == artifact
    assert written["metadata"]["experiment_id"] == 1306
    assert written["source_artifacts"] == [
        "results/experiment_1293_ebt_arm_ebm_cot_energy_bridge_audit.json",
        "results/experiment_1295_milestone_retro_100.json",
    ]
    assert written["energy_bridge_completed"] is True
