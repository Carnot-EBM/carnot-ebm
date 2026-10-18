#!/usr/bin/env python3
"""Milestone 2026.04.91 success-criteria retrospective.

The workflow reads the .91 source experiment artifacts, evaluates each planned
roadmap criterion from its authoritative JSON field, and writes the Exp 1177
deliverable. It keeps the publication hold decision tied to Exp 1167's current
artifact so manual operator overrides are not accidentally erased.

Spec: REQ-REPORT-011, SCENARIO-REPORT-008.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results"
DELIVERABLE_PATH = RESULTS_DIR / "experiment_1177_milestone_retro_91.json"
MILESTONE = "2026.04.91"
CRITERIA_TOTAL = 13

EXPERIMENT_FILES: dict[int, str] = {
    1165: "experiment_1165_phase4_active_inference_pilot_v1.json",
    1166: "experiment_1166_arc_agi3_leaderboard_themesis_outreach.json",
    1167: "experiment_1167_paper_v4_phase4_section.json",
    1168: "experiment_1168_sc_energy_7th_verifier.json",
    1169: "experiment_1169_fover_sota_expansion_v6.json",
    1170: "experiment_1170_beaver_live_logprobs_v2.json",
    1171: "experiment_1171_diffusion_of_thought_inference_v1.json",
    1172: "experiment_1172_nrgpt_per_token_energy_inference.json",
    1173: "experiment_1173_grpo_v5_tinyv_fn_correction.json",
    1174: "experiment_1174_bika_hardware_analysis.json",
    1175: "experiment_1175_wopr_connect_four_cartridge.json",
    1176: "experiment_1176_k6_and_compose_validation.json",
}


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_artifacts(results_dir: Path = RESULTS_DIR) -> dict[int, dict[str, Any]]:
    """Load source artifacts and represent missing deliverables explicitly."""

    artifacts: dict[int, dict[str, Any]] = {}
    for exp_id, filename in EXPERIMENT_FILES.items():
        path = results_dir / filename
        if path.exists():
            artifacts[exp_id] = _load_json(path)
        else:
            artifacts[exp_id] = {"_missing": True, "_path": str(path)}
    return artifacts


def _is_blocked(artifact: Mapping[str, Any]) -> bool:
    status = str(artifact.get("status", "")).lower()
    inference_mode = str(artifact.get("inference_mode", "")).lower()
    return bool(
        status == "blocked"
        or inference_mode.startswith("blocked")
        or artifact.get("blocked_reason")
    )


def _criterion(
    artifacts: Mapping[int, Mapping[str, Any]],
    *,
    name: str,
    exp_id: int | None,
    field: str,
    expected: bool,
    gate_blockable: bool = False,
    detail: str = "",
) -> dict[str, Any]:
    artifact = artifacts.get(exp_id, {}) if exp_id is not None else {}
    actual = True if exp_id is None else artifact.get(field)
    met = actual is expected
    status = "MET" if met else "NOT_MET"
    if not met and exp_id is not None and gate_blockable and _is_blocked(artifact):
        status = "GATE_BLOCKED"
    if exp_id is not None and artifact.get("_missing"):
        status = "NOT_MET"

    return {
        "criterion": name,
        "experiment": None if exp_id is None else f"exp{exp_id}",
        "field": field,
        "expected": expected,
        "actual": actual,
        "status": status,
        "met": status == "MET",
        "detail": detail,
    }


def evaluate_criteria(artifacts: Mapping[int, Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    """Evaluate the 13 planned .91 success criteria from source fields."""

    return {
        "phase4_prototype_operational": _criterion(
            artifacts,
            name="phase4_prototype_operational",
            exp_id=1165,
            field="prototype_operational",
            expected=True,
            detail="Exp1165 must run the Phase 4 active-inference prototype end-to-end.",
        ),
        "themesis_leaderboard_comparison_documented": _criterion(
            artifacts,
            name="themesis_leaderboard_comparison_documented",
            exp_id=1166,
            field="themesis_email_drafted",
            expected=True,
            detail="Exp1166 must document leaderboard context and draft Themesis outreach.",
        ),
        "paper_v4_phase4_section_integrated": _criterion(
            artifacts,
            name="paper_v4_phase4_section_integrated",
            exp_id=1167,
            field="paper_ready_for_arxiv_hold_lift",
            expected=True,
            detail="Exp1167 readiness follows the current artifact after any operator override.",
        ),
        "sc_energy_7th_verifier_auroc_above_threshold": _criterion(
            artifacts,
            name="sc_energy_7th_verifier_auroc_above_threshold",
            exp_id=1168,
            field="sc_energy_auroc_above_threshold",
            expected=True,
            detail="Exp1168 must clear the SC-Energy AUROC and correlation gate.",
        ),
        "fover_sota_pairs_v6_above_500": _criterion(
            artifacts,
            name="fover_sota_pairs_v6_above_500",
            exp_id=1169,
            field="fover_sota_pairs_v6_above_500",
            expected=True,
            detail="Exp1169 must add at least 500 SOTA-labeled FoVer pairs.",
        ),
        "beaver_live_logprobs_sound_bound": _criterion(
            artifacts,
            name="beaver_live_logprobs_sound_bound",
            exp_id=1170,
            field="mock_logprobs_used",
            expected=False,
            detail="Exp1170 must use live llama.cpp logprobs rather than mock probabilities.",
        ),
        "dot_inference_pareto_measured": _criterion(
            artifacts,
            name="dot_inference_pareto_measured",
            exp_id=1171,
            field="dot_inference_pareto_measured",
            expected=True,
            detail="Exp1171 must measure the DoT accuracy-vs-compute Pareto frontier.",
        ),
        "nrgpt_per_token_energy_above_batch": _criterion(
            artifacts,
            name="nrgpt_per_token_energy_above_batch",
            exp_id=1172,
            field="nrgpt_per_token_energy_above_batch",
            expected=True,
            detail="Exp1172 must show per-token energy meeting or beating batch energy.",
        ),
        "grpo_v5_honest_result": _criterion(
            artifacts,
            name="grpo_v5_honest_result",
            exp_id=1173,
            field="grpo_v5_honest_result",
            expected=True,
            gate_blockable=True,
            detail="Exp1173 must complete GRPO v5; blocked runtime status is gate-blocked.",
        ),
        "bika_hardware_analysis_complete": _criterion(
            artifacts,
            name="bika_hardware_analysis_complete",
            exp_id=1174,
            field="bika_hardware_analysis_complete",
            expected=True,
            detail="Exp1174 must complete RM/BOP/NABS and NPU feasibility analysis.",
        ),
        "connect_four_cartridge_shipped": _criterion(
            artifacts,
            name="connect_four_cartridge_shipped",
            exp_id=1175,
            field="cartridge_shipped",
            expected=True,
            detail="Exp1175 must ship the Connect Four WOPR cartridge.",
        ),
        "k6_and_compose_auroc_measured": _criterion(
            artifacts,
            name="k6_and_compose_auroc_measured",
            exp_id=1176,
            field="k6_and_compose_auroc_measured",
            expected=True,
            detail="Exp1176 must report k=6 versus k=5 AUROC.",
        ),
        "retro_complete": _criterion(
            artifacts,
            name="retro_complete",
            exp_id=None,
            field="retro_complete",
            expected=True,
            detail="Exp1177 retrospective artifact was assembled.",
        ),
    }


def criteria_status(criteria: Mapping[str, Mapping[str, Any]]) -> dict[str, str]:
    return {name: str(item["status"]) for name, item in criteria.items()}


def criteria_met_count(criteria: Mapping[str, Mapping[str, Any]]) -> int:
    return sum(1 for item in criteria.values() if item["status"] == "MET")


def criteria_results(criteria: Mapping[str, Mapping[str, Any]]) -> dict[str, bool]:
    return {name: item["status"] == "MET" for name, item in criteria.items()}


def record_honest_verdicts(
    artifacts: Mapping[int, Mapping[str, Any]],
    self_verdict: str | None = None,
) -> dict[str, str]:
    verdicts: dict[str, str] = {}
    for exp_id in sorted(EXPERIMENT_FILES):
        artifact = artifacts.get(exp_id, {"_missing": True})
        verdicts[f"exp{exp_id}"] = (
            "MISSING"
            if artifact.get("_missing")
            else str(artifact.get("honest_verdict", "NO_VERDICT"))
        )
    if self_verdict is not None:
        verdicts["exp1177"] = self_verdict
    return verdicts


def phase4_hold_lift_ready(artifacts: Mapping[int, Mapping[str, Any]]) -> bool:
    return bool(artifacts.get(1167, {}).get("paper_ready_for_arxiv_hold_lift"))


def phase4_hold_lift_note(ready: bool) -> str:
    if ready:
        return (
            "Phase 4 hold-lift prerequisites are met; operator should review "
            "docs/arxiv-paper/main.pdf and carnot-arxiv-v5.tar.gz for arXiv submission."
        )
    return (
        "Phase 4 hold-lift prerequisites are not met; Exp1167 is still blocked by "
        "the figure-integrity and hardware-claim audit called out in ops/known-issues.md."
    )


def top_3_successes(artifacts: Mapping[int, Mapping[str, Any]]) -> list[str]:
    e1165 = artifacts.get(1165, {})
    e1168 = artifacts.get(1168, {})
    e1169 = artifacts.get(1169, {})
    e1170 = artifacts.get(1170, {})
    e1172 = artifacts.get(1172, {})
    e1174 = artifacts.get(1174, {})
    e1175 = artifacts.get(1175, {})
    return [
        (
            "Phase 4 pilot became operational: Exp1165 evaluated "
            f"{e1165.get('n_puzzles_evaluated')} puzzles, solved_rate="
            f"{e1165.get('phase4_solved_rate')}, action_count_ratio="
            f"{e1165.get('action_count_ratio')}."
        ),
        (
            "Verifier data and certificates advanced: SC-Energy AUROC="
            f"{e1168.get('sc_energy_auroc')}, FoVer added "
            f"{e1169.get('n_new_pairs')} pairs, and BEAVER used mock_logprobs_used="
            f"{e1170.get('mock_logprobs_used')}."
        ),
        (
            "Phase 3/hardware/gallery shipped useful assets: NRGPT per-token AUROC="
            f"{e1172.get('per_token_auroc')}, BiKA NPU verdict="
            f"{e1174.get('npu_feasibility_verdict')}, Connect Four tests="
            f"{e1175.get('n_tests_passing')}."
        ),
    ]


def top_3_gaps(artifacts: Mapping[int, Mapping[str, Any]]) -> list[str]:
    e1167 = artifacts.get(1167, {})
    e1173 = artifacts.get(1173, {})
    e1176 = artifacts.get(1176, {})
    return [
        (
            "Publication hold remains active: Exp1167 structural paper work completed, but "
            f"paper_ready_for_arxiv_hold_lift={e1167.get('paper_ready_for_arxiv_hold_lift')} "
            f"after verdict {e1167.get('honest_verdict')}."
        ),
        (
            "GRPO v5 did not produce an honest training result: status="
            f"{e1173.get('status')}, dualgpu_confirmed={e1173.get('dualgpu_confirmed')}, "
            f"blocked_reason={e1173.get('blocked_reason')}."
        ),
        (
            "k=6 AND-compose was measured but did not improve: k6_auroc="
            f"{e1176.get('k6_auroc')} versus k5_eval={e1176.get('k5_auroc_on_eval')}, "
            f"k6_above_k5={e1176.get('k6_above_k5')}."
        ),
    ]


def open_items_for_92() -> list[str]:
    return [
        "Run the publication figure-integrity audit across docs/figures/fig1-fig7.",
        "Run the hardware-claim audit over docs/arxiv-paper/main.tex and fix or remove fig3.",
        "Rerun or redesign GRPO v5 only after llama.cpp GPU offload is available and verified.",
        "Investigate why SC-Energy lowered k=6 AND-compose AUROC despite low correlations.",
        "Extend DoT beyond measured non-monotone Pareto results before making performance claims.",
    ]


def build_artifact(artifacts: Mapping[int, Mapping[str, Any]]) -> dict[str, Any]:
    criteria = evaluate_criteria(artifacts)
    n_met = criteria_met_count(criteria)
    honest_verdict = f"{n_met}_of_{CRITERIA_TOTAL}_criteria_met"
    ready = phase4_hold_lift_ready(artifacts)

    return {
        "experiment": "1177_milestone_retro_91",
        "schema": "milestone_retro_v2",
        "run_date": dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "milestone": MILESTONE,
        "criteria_results": criteria_results(criteria),
        "criteria_status": criteria_status(criteria),
        "criteria_detail": criteria,
        "criteria_met": n_met,
        "criteria_total": CRITERIA_TOTAL,
        "experiment_honest_verdicts": record_honest_verdicts(artifacts, honest_verdict),
        "phase4_hold_lift_ready": ready,
        "phase4_hold_lift_note": phase4_hold_lift_note(ready),
        "top_3_successes": top_3_successes(artifacts),
        "top_3_gaps": top_3_gaps(artifacts),
        "open_items_for_92": open_items_for_92(),
        "retro_complete": True,
        "honest_verdict": honest_verdict,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--out", type=Path, default=DELIVERABLE_PATH)
    args = parser.parse_args(argv)

    artifacts = load_artifacts(args.results_dir)
    artifact = build_artifact(artifacts)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    print(
        f"[exp1177] {artifact['criteria_met']}/{artifact['criteria_total']} criteria met; "
        f"phase4_hold_lift_ready={artifact['phase4_hold_lift_ready']}; out={args.out}"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
