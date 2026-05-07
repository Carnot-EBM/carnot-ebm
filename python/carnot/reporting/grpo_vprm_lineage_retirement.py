"""Build the Exp 1456 GRPO/VPRM lineage retirement artifact.

Spec: REQ-REPORT-042, SCENARIO-REPORT-042.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
RUN_DATE = "20260507"
PROJECT_ROOT_FOR_METADATA = "/home/ianblenke/github.com/ianblenke/carnot"
LINEAGE_NAME = "GRPO/VPRM"
SCHEMA = "grpo_vprm_lineage_retirement_v1"
EXPERIMENT = "1456_grpo_vprm_lineage_consolidation_retirement"
EXCLUSION_MARKER = "grpo_vprm_v15_scope_closed"

DEFAULT_OUT_PATH = (
    REPO_ROOT / "results" / "experiment_1456_grpo_vprm_lineage_consolidation_retirement.json"
)
DEFAULT_NOTE_PATH = REPO_ROOT / "ops" / "lineage-retirements" / "grpo_vprm_lineage_retired.md"
DEFAULT_MANIFEST_PATH = REPO_ROOT / "ops" / "exclusion_manifest.yaml"

REQUIRED_ARTIFACT_FIELDS = {
    "status",
    "lineage_name",
    "experiments_reviewed",
    "consolidation_note_path",
    "grpo_lineage_retired",
    "exclusion_manifest_updated",
    "lessons_retained",
    "future_reopen_conditions",
    "honest_verdict",
}

LESSONS_RETAINED = [
    (
        "Energy-shaped process rewards produced real early signal in small slices "
        "(Exp 1118 +4pp, Exp 1129 +8.51pp, Exp 1159 +10pp), but those results do "
        "not justify more variant churn after the later blocked and no-improvement path."
    ),
    (
        "Step-level process supervision is the useful lesson to keep: GRPO-VPS "
        "showed +24pp in Exp 1209 and +15pp over the v4 floor in Exp 1220."
    ),
    (
        "TinyV false-negative correction must be calibrated before use; Exp 1208 "
        "over-abstained on 62.5% of rewards and regressed by -35pp."
    ),
    (
        "Candidate-pool saturation is a hard blocker. Future selector work must "
        "change the candidate pool or target false-acceptance reduction instead "
        "of repeating best-of-N selection."
    ),
    (
        "Formal verifier rewards need non-UNKNOWN candidate diversity. Exp 1383 "
        "and Exp 1393 both produced 0pp held-out improvement because rewards stayed "
        "all zero or UNKNOWN."
    ),
    (
        "Gate discipline worked: SOTA certificate, DVI lossless acceptance, parse, "
        "and non-forgetting gates should block downstream GRPO/VPRM instead of "
        "launching placeholder variants."
    ),
]

FUTURE_REOPEN_CONDITIONS = [
    (
        "An operator explicitly reopens the GRPO/VPRM scope and names the prior "
        "failure mode being addressed."
    ),
    (
        "The proposal identifies a root cause not already tested by TinyV, VPS, "
        "FSPO, PROGRS, PRIME/VPRM, JURY-RL, or NGRPO variants."
    ),
    (
        "The proposal changes a prerequisite that failed before, such as non-UNKNOWN "
        "reward diversity, calibrated false-negative correction, an unsaturated "
        "candidate pool, live SOTA certificate parse/truthfulness gates, or DVI "
        "lossless acceptance."
    ),
    (
        "The proposal states a falsifiable acceptance gate: at least +10pp over the "
        "best retained v4/VPS baseline on 50 or more evaluation cases, with headline "
        "eligibility and no missing upstream artifacts."
    ),
]

LINEAGE_REVIEW_ROWS: list[dict[str, str]] = [
    {
        "experiment_id": "exp1084",
        "title": "Step-Level PRM Data Generation",
        "verdict": "7349-step corpus generated",
        "measured_positive": "large process-reward corpus created",
        "blocker": "corpus generation alone was not a GRPO result",
        "lesson": "Process-level labels are useful infrastructure.",
    },
    {
        "experiment_id": "exp1111",
        "title": "ThinkPRM v2 Retrain on 7349-Step PRM Corpus",
        "verdict": "retrain attempted",
        "measured_positive": "ThinkPRM reward source available for v1/v2",
        "blocker": "300-sample retrain quality was not production-grade",
        "lesson": "Reward model quality must be verified on held-out slices.",
    },
    {
        "experiment_id": "exp1118",
        "title": "GRPO with ThinkPRM v2 Energy Reward",
        "verdict": "positive_improvement",
        "measured_positive": "+4pp on 25 live-GPU eval questions",
        "blocker": "small eval slice",
        "lesson": "Energy-shaped rewards can move in-distribution slices.",
    },
    {
        "experiment_id": "exp1129",
        "title": "GRPO Energy PRM Full Training v2",
        "verdict": "positive_improvement",
        "measured_positive": "+8.51pp on 47 completed eval questions",
        "blocker": "evaluation wall budget hit",
        "lesson": "DRA diversity and proxy reuse were useful but budget-sensitive.",
    },
    {
        "experiment_id": "exp1146",
        "title": "GRPO Reflection Reward v3",
        "verdict": "positive_below_exp1129",
        "measured_positive": "+2.86pp",
        "blocker": "below Exp 1129 and reflection reward mean stayed 0.0",
        "lesson": "Reflection reward did not add a stronger signal by itself.",
    },
    {
        "experiment_id": "exp1159",
        "title": "GRPO Reflection Reward v4 Structural Warm-up",
        "verdict": "structural_warmup_above_0851",
        "measured_positive": "+10pp, +1.49pp over Exp 1129",
        "blocker": "still small and lineage-specific",
        "lesson": "Structural warm-up was the best GRPO-only historical result.",
    },
    {
        "experiment_id": "exp1173",
        "title": "GRPO v5 + TinyV False-Negative Correction",
        "verdict": "training_wall_hit",
        "measured_positive": "",
        "blocker": "llama.cpp runtime lacked GPU offload",
        "lesson": "Runtime prerequisites must gate training variants.",
    },
    {
        "experiment_id": "exp1184",
        "title": "GRPO v5 + TinyV v2",
        "verdict": "gpu_offload_prerequisite_not_met",
        "measured_positive": "",
        "blocker": "CPU-only llama.cpp build",
        "lesson": "Do not score blocked runtime setup as science.",
    },
    {
        "experiment_id": "exp1187",
        "title": "Latent-GRPO Energy Reward",
        "verdict": "latent_grpo_no_delta",
        "measured_positive": "",
        "blocker": "0.0pp delta on the proxy",
        "lesson": "Invalid-sample masking needs actual invalid samples or a nonzero target.",
    },
    {
        "experiment_id": "exp1195",
        "title": "GRPO v5 TinyV v2 rerun",
        "verdict": "missing",
        "measured_positive": "",
        "blocker": "missing artifact in .93 retro",
        "lesson": "Missing gated artifacts must not be treated as successful retries.",
    },
    {
        "experiment_id": "exp1196",
        "title": "GRPO-VPS Step-Level Process Supervision",
        "verdict": "blocked_gate_check_failed",
        "measured_positive": "",
        "blocker": "prior_failures metadata incomplete",
        "lesson": "Prior-failure hygiene must be complete before reruns.",
    },
    {
        "experiment_id": "exp1208",
        "title": "GRPO v5 TinyV Confidence Abstention",
        "verdict": "improvement_below_v4",
        "measured_positive": "",
        "blocker": "TinyV abstained on 62.5% of rewards and regressed -35pp",
        "lesson": "False-negative correction can suppress valid rewards if over-calibrated.",
    },
    {
        "experiment_id": "exp1209",
        "title": "GRPO-VPS Step-Level Supervision",
        "verdict": "step_supervision_improves_over_outcome",
        "measured_positive": "+24pp over outcome-only baseline",
        "blocker": "process-supervision result, not a reason for GRPO v15 churn",
        "lesson": "Preserve VPS as process-supervision evidence.",
    },
    {
        "experiment_id": "exp1219",
        "title": "GRPO v5 Regression Diagnosis",
        "verdict": "root_cause_identified",
        "measured_positive": "diagnosed TinyV abstention root cause",
        "blocker": "diagnosis confirmed v5 regression",
        "lesson": "Abstention thresholds need calibration and reward-mass checks.",
    },
    {
        "experiment_id": "exp1220",
        "title": "GRPO-VPS Full Training",
        "verdict": "vps_training_beats_v4",
        "measured_positive": "+15pp over v4 floor",
        "blocker": "VPS is the retained lesson, not open-ended GRPO expansion",
        "lesson": "Step rewards are more promising than more variant labels.",
    },
    {
        "experiment_id": "exp1221",
        "title": "GRPO v6 FSPO + VPS",
        "verdict": "insufficient_logprob_coverage",
        "measured_positive": "",
        "blocker": "-6.11pp on only 9 questions",
        "lesson": "FSPO needs logprob coverage before claims.",
    },
    {
        "experiment_id": "exp1235",
        "title": "GRPO v6 FSPO + VPS Extended",
        "verdict": "in_progress",
        "measured_positive": "",
        "blocker": "no terminal improvement artifact",
        "lesson": "In-progress skeletons are not evidence.",
    },
    {
        "experiment_id": "exp1236",
        "title": "GRPO Execution-Grounded Credit",
        "verdict": "missing",
        "measured_positive": "",
        "blocker": "planned artifact absent",
        "lesson": "Execution-grounded credit never produced terminal evidence.",
    },
    {
        "experiment_id": "exp1247",
        "title": "GRPO v7 Simplified",
        "verdict": "in_progress",
        "measured_positive": "",
        "blocker": "no measured outcome",
        "lesson": "Simplification did not close the lineage.",
    },
    {
        "experiment_id": "exp1259",
        "title": "GRPO v7 PROGRS + VPS",
        "verdict": "in_progress",
        "measured_positive": "",
        "blocker": "no measured outcome",
        "lesson": "PROGRS did not produce terminal evidence.",
    },
    {
        "experiment_id": "exp1272",
        "title": "PRIME Verifier Selection Audit",
        "verdict": "prime weights selected",
        "measured_positive": "process/outcome alignment weights selected",
        "blocker": "audit only",
        "lesson": "Verifier weighting is useful as an input to process supervision.",
    },
    {
        "experiment_id": "exp1273",
        "title": "GRPO v8 PRIME+VPRM Smoke",
        "verdict": "smoke_only_not_headline",
        "measured_positive": "smoke delta +83.798pp",
        "blocker": "headline_result_allowed=false",
        "lesson": "Smoke-only deltas must not become headline claims.",
    },
    {
        "experiment_id": "exp1289",
        "title": "GRPO/VPRM v9 SOTA Headline Attempt",
        "verdict": "gated/missing",
        "measured_positive": "",
        "blocker": "SOTA certificate path did not open",
        "lesson": "SOTA/DVI gates must precede headline GRPO/VPRM.",
    },
    {
        "experiment_id": "exp1304",
        "title": "GRPO/VPRM v10 SOTA Gated",
        "verdict": "gated/missing",
        "measured_positive": "",
        "blocker": "absent SOTA certificate headline result",
        "lesson": "No downstream GRPO launch without live certificate evidence.",
    },
    {
        "experiment_id": "exp1317",
        "title": "GRPO/VPRM v11 Headline Gate",
        "verdict": "grpo_vprm_v11_positive_headline_gate",
        "measured_positive": "+0.45 score delta on 40 replay cases",
        "blocker": "deterministic replay audit; no large training job or new generation",
        "lesson": "Keep as micro-audit evidence, not a mandate for variants.",
    },
    {
        "experiment_id": "exp1330",
        "title": "GRPO/VPRM v12 Micro-Audit",
        "verdict": "missing/gated",
        "measured_positive": "",
        "blocker": "DVI lossless claim gate absent",
        "lesson": "DVI acceptance gates correctly block downstream claims.",
    },
    {
        "experiment_id": "exp1346",
        "title": "GRPO/VPRM v13 Micro-Audit",
        "verdict": "missing/gated",
        "measured_positive": "",
        "blocker": "DVI lossless claim gate absent",
        "lesson": "Repeating the same gate does not add evidence.",
    },
    {
        "experiment_id": "exp1360",
        "title": "GRPO/VPRM v14 Micro-Audit",
        "verdict": "blocked_gate_check_failed",
        "measured_positive": "",
        "blocker": "missing exp1359 lossless_acceptance_claim_allowed",
        "lesson": "Final v14 attempt stayed downstream of a closed DVI gate.",
    },
    {
        "experiment_id": "exp1383",
        "title": "GRPO v7 JURY-RL Formal Verifier Rewards",
        "verdict": "grpo_v7_jury_rl_no_improvement",
        "measured_positive": "",
        "blocker": "0pp held-out improvement on all-UNKNOWN rewards",
        "lesson": "Formal-verifier reward policies need non-UNKNOWN diversity.",
    },
    {
        "experiment_id": "exp1388",
        "title": "FR-11 Self-Learning v4 DVI + GRPO Integration",
        "verdict": "dvi_only_headline_allowed",
        "measured_positive": "DVI path integrated 59 fresh verified cases",
        "blocker": "grpo_cases_integrated=0",
        "lesson": "Self-learning should stay DVI-only until GRPO produces positive evidence.",
    },
    {
        "experiment_id": "exp1393",
        "title": "GRPO v8 NGRPO Zero-Reward Fix",
        "verdict": "grpo_v8_ngrpo_no_improvement_all_unknown_retired",
        "measured_positive": "",
        "blocker": "0pp improvement, UNKNOWN rollout rate 1.0, retire_if_same_verdict=true",
        "lesson": "NGRPO calibration did not fix the zero-reward root cause.",
    },
]


def _write_json(path: Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    artifact = dict(payload)
    path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return artifact


def _relative_path(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        parts = path.parts
        for anchor in ("ops", "results"):
            if anchor in parts:
                return str(Path(*parts[parts.index(anchor) :]))
        return path.name


def write_in_progress_artifact(out_path: Path | str = DEFAULT_OUT_PATH) -> dict[str, Any]:
    """REQ-REPORT-042: record the bootstrap state before terminal scoring."""

    artifact: dict[str, Any] = {field: None for field in REQUIRED_ARTIFACT_FIELDS}
    artifact.update(
        {
            "experiment": EXPERIMENT,
            "schema": SCHEMA,
            "run_date": RUN_DATE,
            "project_root": PROJECT_ROOT_FOR_METADATA,
            "status": "in_progress",
            "lineage_name": LINEAGE_NAME,
            "experiments_reviewed": [],
            "consolidation_note_path": _relative_path(DEFAULT_NOTE_PATH),
            "grpo_lineage_retired": False,
            "exclusion_manifest_updated": False,
            "lessons_retained": [],
            "future_reopen_conditions": [],
            "honest_verdict": "in_progress",
        }
    )
    return _write_json(Path(out_path), artifact)


def _manifest_contains_block(manifest_text: str) -> bool:
    return (
        EXCLUSION_MARKER in manifest_text
        and "GRPO v15" in manifest_text
        and "VPRM v15" in manifest_text
    )


def ensure_manifest_block(manifest_text: str) -> tuple[str, bool]:
    """REQ-REPORT-042: append a planner-visible GRPO v15/VPRM v15 block once."""

    if _manifest_contains_block(manifest_text):
        return manifest_text, False
    text = manifest_text.rstrip()
    if "retired_extras:" not in text:
        text = f"{text}\n\nretired_extras:" if text else "retired_extras:"
    block = f"""

  # Added by Exp 1456 - GRPO/VPRM lineage consolidation and retirement.
  - id: {EXCLUSION_MARKER}
    reason: |
      GRPO/VPRM v1-v14 is retired as active research scope after repeated
      blocked, missing, smoke-only, no-improvement, and non-headline outcomes.
      Future GRPO v15 or VPRM v15 proposals are blocked unless an operator
      explicitly reopens the scope with a new root cause and falsifiable gate.
    experiment_ids:
      - exp1118
      - exp1129
      - exp1146
      - exp1159
      - exp1173
      - exp1184
      - exp1187
      - exp1195
      - exp1196
      - exp1208
      - exp1209
      - exp1219
      - exp1220
      - exp1221
      - exp1235
      - exp1236
      - exp1247
      - exp1259
      - exp1273
      - exp1289
      - exp1304
      - exp1317
      - exp1330
      - exp1346
      - exp1360
      - exp1383
      - exp1388
      - exp1393
    blocked_patterns:
      - "GRPO v15"
      - "VPRM v15"
      - "GRPO/VPRM v15"
    retired_milestone: "2026.04.112"
    retired_by_artifact: "results/experiment_1456_grpo_vprm_lineage_consolidation_retirement.json"
    operator_reopen_required: true
    retire_if_same_verdict: true
"""
    return text + block, True


def _md_cell(value: object) -> str:
    text = str(value) if value else "none"
    return text.replace("\n", " ").replace("|", "\\|")


def _experiment_ids(review_rows: Iterable[Mapping[str, object]]) -> list[str]:
    return [str(row["experiment_id"]) for row in review_rows]


def build_artifact(
    *,
    review_rows: Iterable[Mapping[str, object]],
    consolidation_note_path: str,
    manifest_text: str,
    manifest_block_added: bool,
) -> dict[str, Any]:
    """REQ-REPORT-042: assemble the terminal retirement artifact."""

    rows = [dict(row) for row in review_rows]
    manifest_has_block = _manifest_contains_block(manifest_text)
    complete = bool(rows) and manifest_has_block
    artifact: dict[str, Any] = {
        "experiment": EXPERIMENT,
        "schema": SCHEMA,
        "run_date": RUN_DATE,
        "project_root": PROJECT_ROOT_FOR_METADATA,
        "status": "complete" if complete else "blocked",
        "lineage_name": LINEAGE_NAME,
        "experiments_reviewed": _experiment_ids(rows),
        "experiment_review_rows": rows,
        "consolidation_note_path": consolidation_note_path,
        "grpo_lineage_retired": complete,
        "exclusion_manifest_updated": manifest_has_block,
        "exclusion_manifest_block_added": manifest_block_added,
        "lessons_retained": list(LESSONS_RETAINED),
        "future_reopen_conditions": list(FUTURE_REOPEN_CONDITIONS),
        "honest_verdict": (
            "grpo_vprm_lineage_retired_no_v15_without_operator_reopen"
            if complete
            else "grpo_vprm_lineage_retirement_blocked_missing_manifest_or_review"
        ),
    }
    return artifact


def render_consolidation_note(
    review_rows: Iterable[Mapping[str, object]],
    artifact: Mapping[str, Any],
) -> str:
    """SCENARIO-REPORT-042: render the operator-facing consolidation note."""

    lines = [
        "# GRPO/VPRM Lineage Retirement",
        "",
        f"Run date: `{RUN_DATE}`",
        f"Lineage: `{LINEAGE_NAME}`",
        f"Artifact: `results/experiment_1456_grpo_vprm_lineage_consolidation_retirement.json`",
        "",
        "## Experiments Reviewed",
        "",
        "| experiment | verdict | measured positives | repeated blockers | retained lesson |",
        "|---|---|---|---|---|",
    ]
    for row in review_rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    _md_cell(row.get("experiment_id")),
                    _md_cell(row.get("verdict")),
                    _md_cell(row.get("measured_positive")),
                    _md_cell(row.get("blocker")),
                    _md_cell(row.get("lesson")),
                ]
            )
            + " |"
        )
    lines.extend(["", "## Measured Positives", ""])
    for lesson in artifact["lessons_retained"][:2]:
        lines.append(f"- {lesson}")
    lines.extend(["", "## Repeated Blockers", ""])
    for lesson in artifact["lessons_retained"][2:]:
        lines.append(f"- {lesson}")
    lines.extend(["", "## Future Reopen Conditions", ""])
    for condition in artifact["future_reopen_conditions"]:
        lines.append(f"- {condition}")
    lines.extend(
        [
            "",
            "## Final Decision",
            "",
            (
                "GRPO/VPRM is retired as active research scope. GRPO v15 and VPRM v15 "
                "variant proposals are blocked unless an operator explicitly reopens "
                "the scope under the conditions above."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def run(
    root: Path | str = REPO_ROOT,
    out_path: Path | str = DEFAULT_OUT_PATH,
    note_path: Path | str = DEFAULT_NOTE_PATH,
    manifest_path: Path | str = DEFAULT_MANIFEST_PATH,
    review_rows: Iterable[Mapping[str, object]] = LINEAGE_REVIEW_ROWS,
) -> dict[str, Any]:
    """REQ-REPORT-042: write bootstrap, note, manifest block, and terminal JSON."""

    _ = Path(root)
    out = Path(out_path)
    note = Path(note_path)
    manifest = Path(manifest_path)
    write_in_progress_artifact(out)
    manifest_text = manifest.read_text(encoding="utf-8") if manifest.exists() else ""
    updated_manifest, block_added = ensure_manifest_block(manifest_text)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(updated_manifest + ("" if updated_manifest.endswith("\n") else "\n"), encoding="utf-8")
    artifact = build_artifact(
        review_rows=review_rows,
        consolidation_note_path=_relative_path(note),
        manifest_text=updated_manifest,
        manifest_block_added=block_added,
    )
    note.parent.mkdir(parents=True, exist_ok=True)
    note.write_text(render_consolidation_note(review_rows, artifact), encoding="utf-8")
    return _write_json(out, artifact)


if __name__ == "__main__":  # pragma: no cover - thin CLI convenience
    run()
