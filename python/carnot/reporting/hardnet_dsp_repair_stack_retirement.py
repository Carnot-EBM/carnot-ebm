"""Build the Exp 1458 HardNet++/DSP repair-stack retirement artifact.

Spec: REQ-REPORT-044, SCENARIO-REPORT-044.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
RUN_DATE = "20260507"
PROJECT_ROOT_FOR_METADATA = "/home/ianblenke/github.com/ianblenke/carnot"
SCHEMA = "hardnet_dsp_repair_stack_retirement_v1"
EXPERIMENT = "1458_hardnet_dsp_repair_stack_consolidation"
EXCLUSION_MARKER = "hardnet_dsp_repair_stack_scope_closed"

DEFAULT_OUT_PATH = (
    REPO_ROOT / "results" / "experiment_1458_hardnet_dsp_repair_stack_consolidation.json"
)
DEFAULT_NOTE_PATH = (
    REPO_ROOT / "ops" / "lineage-retirements" / "hardnet_dsp_repair_stack_retired.md"
)
DEFAULT_MANIFEST_PATH = REPO_ROOT / "ops" / "exclusion_manifest.yaml"

REQUIRED_ARTIFACT_FIELDS = {
    "status",
    "hardnet_dsp_experiments_reviewed",
    "consolidation_note_path",
    "hardnet_dsp_lineage_retired",
    "exclusion_manifest_updated",
    "lessons_retained",
    "cited_recent_constraint_papers",
    "future_reopen_conditions",
    "honest_verdict",
}

LESSONS_RETAINED = [
    (
        "Hard projection and repair layers are valuable when the domain has an "
        "explicit feasible set; they should remain available for continuous "
        "numeric repair and Phase-3 substrate work."
    ),
    (
        "FSNet and SnareNet-style feasibility steps reduce hard violations, but "
        "their local-linear repair behavior should stop when nonlinear residuals "
        "remain instead of spawning more variants."
    ),
    (
        "HardNet++ is the retained route for residual nonlinear feasibility cases "
        "because it reached hard feasibility where repeated local-linear repairs "
        "left violations."
    ),
    (
        "DSP feasibility channels are useful telemetry, but the measured AUC and "
        "false-continue behavior were marginal as a learned general stop rule."
    ),
    (
        "Conservative replay is the retained operator gate: stop once hard "
        "feasibility is reached, continue only when hard violations remain, and "
        "route nonlinear residuals to the certifying repair operator."
    ),
]

FUTURE_REOPEN_CONDITIONS = [
    (
        "An operator explicitly reopens the line and names the new root cause "
        "that was not addressed by Exps 1292, 1305, and 1318."
    ),
    (
        "A proposal shows non-replay held-out evidence that a learned DSP or "
        "HardNet++/DSP policy improves over the conservative replay gate by a "
        "predeclared margin."
    ),
    (
        "The proposal ties the repair layer to a production verifier failure or "
        "Phase-3 continuous-latent substrate gate, not just another variant of "
        "the same repair-stack benchmark."
    ),
    (
        "The proposal includes a falsifiable acceptance gate, a fresh corpus or "
        "OOD split, and an explicit retire-if-same-verdict rule."
    ),
]

CITED_RECENT_CONSTRAINT_PAPERS: list[dict[str, str]] = [
    {
        "name": "HardNet++",
        "citation": "arXiv:2604.19669",
        "source": "https://arxiv.org/abs/2604.19669",
        "lesson": (
            "Differentiable nonlinear projection supports Carnot's hard "
            "feasibility-first repair lesson."
        ),
    },
    {
        "name": "KKT-Hardnet",
        "citation": "arXiv:2507.08124",
        "source": "https://arxiv.org/abs/2507.08124",
        "lesson": (
            "KKT projection remains a possible future mechanism if a reopened "
            "scope needs machine-precision equality/inequality feasibility."
        ),
    },
    {
        "name": "SnareNet",
        "citation": "arXiv:2602.09317",
        "source": "https://arxiv.org/abs/2602.09317",
        "lesson": (
            "Adaptive repair layers validate feasibility repair, while Carnot's "
            "lineage shows they should not proliferate without new evidence."
        ),
    },
    {
        "name": "Differentiable Symbolic Planning with Feasibility Channels",
        "citation": "arXiv:2604.02350",
        "source": "research-references.md",
        "lesson": (
            "Feasibility channels are worth retaining as signals, but Carnot's "
            "DSP replay tests did not establish a broad learned stop rule."
        ),
    },
]

HARDNET_DSP_REVIEW_ROWS: list[dict[str, str]] = [
    {
        "experiment_id": "exp1147",
        "title": "HardNet++-Style Projection Repair Layer for Arithmetic Constraints",
        "artifact_path": "results/experiment_1147_hardnet_projection_repair.json",
        "verdict": "projection_accurate_and_fast",
        "evidence": (
            "20 violations tested, projection_repair_accuracy=1.0, and "
            "projection_repair_latency_us=117.33625."
        ),
        "lesson": "Hard projection can certify feasibility in explicit numeric domains.",
    },
    {
        "experiment_id": "exp1275",
        "title": "FSNet Feasibility Step for Continuous EBM",
        "artifact_path": "results/experiment_1275_fsnet_feasibility_step_continuous_ebm.json",
        "verdict": "feasibility_step_viable",
        "evidence": (
            "feasibility_delta_overall=4.58324736985576 and violation_count_mean "
            "dropped from 5.0 to 0.0."
        ),
        "lesson": "Feasibility seeking should be an operator inside repair, not a headline claim.",
    },
    {
        "experiment_id": "exp1276",
        "title": "SnareNet Repair Layer - Gated on FSNet Feasibility Delta",
        "artifact_path": "results/experiment_1276_snarenet_repair_layer_gated.json",
        "verdict": "adaptive_repair_improves_fsnet",
        "evidence": (
            "final_constraint_satisfaction=0.9895926247512347 and "
            "repair_delta_over_fsnet=0.2199604492292856."
        ),
        "lesson": "Adaptive repair improves local behavior but does not justify open-ended variants.",
    },
    {
        "experiment_id": "exp1291",
        "title": "HardNet++ Nonlinear Repair Benchmark",
        "artifact_path": "results/experiment_1291_hardnetpp_nonlinear_repair_benchmark.json",
        "verdict": "hardnetpp_nonlinear_repair_viable",
        "evidence": (
            "hardnetpp_delta_over_snarenet=1.2207222442957435 with "
            "nonlinear_repair_viable=true."
        ),
        "lesson": "Route nonlinear residual cases to HardNet++ rather than repeated local repair.",
    },
    {
        "experiment_id": "exp1292",
        "title": "DSP Feasibility-Channel Diagnostic",
        "artifact_path": "results/experiment_1292_dsp_feasibility_channel_diagnostic.json",
        "verdict": "feasibility_channel_predictive_marginal",
        "evidence": (
            "n_cases=156, feasibility_channel_auc=0.6604651162790698, "
            "repair_help_prediction_accuracy=0.6538461538461539, and "
            "false_continue_rate=0.7714285714285715."
        ),
        "lesson": "DSP phi is useful telemetry but marginal as a learned stop signal.",
    },
    {
        "experiment_id": "exp1305",
        "title": "HardNet++ + DSP Feasibility Stop Policy",
        "artifact_path": "results/experiment_1305_hardnetpp_dsp_feasibility_stop_policy.json",
        "verdict": "conservative_replay_policy_useful_dsp_marginal",
        "evidence": (
            "policy_stop_accuracy=1.0, stop_policy_precision=1.0, and "
            "baseline_dsp_continue_precision=0.6142857142857143."
        ),
        "lesson": "Conservative replay is the retained operator gate.",
    },
    {
        "experiment_id": "exp1318",
        "title": "HardNet++/DSP Learned Stop Policy Generalization",
        "artifact_path": "results/experiment_1318_hardnetpp_dsp_learned_stop_policy.json",
        "verdict": "learned_policy_matched_conservative_replay",
        "evidence": (
            "held_out_count=36, dsp_feasibility_auc=0.640625, "
            "stop_policy_precision=1.0, stop_policy_recall=1.0, and "
            "hardnetpp_delta_over_replay_policy=0.0."
        ),
        "lesson": "The learned policy did not prove general value beyond replay distribution.",
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
    """REQ-REPORT-044: record bootstrap state before evidence review completes."""

    artifact: dict[str, Any] = {field: None for field in REQUIRED_ARTIFACT_FIELDS}
    artifact.update(
        {
            "experiment": EXPERIMENT,
            "schema": SCHEMA,
            "run_date": RUN_DATE,
            "project_root": PROJECT_ROOT_FOR_METADATA,
            "status": "in_progress",
            "hardnet_dsp_experiments_reviewed": [],
            "consolidation_note_path": _relative_path(DEFAULT_NOTE_PATH),
            "hardnet_dsp_lineage_retired": False,
            "exclusion_manifest_updated": False,
            "lessons_retained": [],
            "cited_recent_constraint_papers": [],
            "future_reopen_conditions": [],
            "honest_verdict": "in_progress",
        }
    )
    return _write_json(Path(out_path), artifact)


def _manifest_contains_block(manifest_text: str) -> bool:
    return (
        EXCLUSION_MARKER in manifest_text
        and "HardNet++/DSP" in manifest_text
        and "operator explicitly reopens the line" in manifest_text
    )


def ensure_manifest_block(manifest_text: str) -> tuple[str, bool]:
    """REQ-REPORT-044: append one planner-visible block for variant proposals."""

    if _manifest_contains_block(manifest_text):
        return manifest_text, False
    text = manifest_text.rstrip()
    if "retired_extras:" not in text:
        text = f"{text}\n\nretired_extras:" if text else "retired_extras:"
    block = f"""

  # Added by Exp 1458 - HardNet++/DSP repair-stack consolidation and retirement.
  - id: {EXCLUSION_MARKER}
    reason: |
      HardNet++/DSP repair variants are retired as active headline research scope
      after the stack repeatedly reduced to the same retained lesson:
      hard-constraint repair is useful, but the learned DSP stop policy matched
      conservative replay and did not prove a broad general rule. Future
      HardNet++/DSP, DSP feasibility-channel, SnareNet/FSNet repair, or
      KKT-Hardnet variant tasks are blocked unless an operator explicitly reopens the line
      with a new root cause, non-replay evidence, and a
      falsifiable gate.
    experiment_ids:
      - exp1147
      - exp1275
      - exp1276
      - exp1291
      - exp1292
      - exp1305
      - exp1318
    blocked_patterns:
      - "HardNet++"
      - "HardNet++/DSP"
      - "KKT-Hardnet"
      - "DSP feasibility channel"
      - "DSP stop policy"
      - "SnareNet repair"
      - "FSNet repair"
      - "conservative replay repair policy"
      - "repair stack variant"
    retained_lesson: "hard constraints remain valuable; variant proliferation does not"
    retired_milestone: "2026.04.112"
    retired_by_artifact: "results/experiment_1458_hardnet_dsp_repair_stack_consolidation.json"
    operator_reopen_required: true
    retire_if_same_verdict: true
"""
    return text + block, True


def _md_cell(value: object) -> str:
    text = str(value) if value else "none"
    return text.replace("\n", " ").replace("|", "\\|")


def _experiment_ids(review_rows: Iterable[Mapping[str, object]]) -> list[str]:
    return [str(row["experiment_id"]) for row in review_rows]


def _lessons_from_rows(review_rows: Iterable[Mapping[str, object]]) -> list[str]:
    row_lessons = [str(row["lesson"]) for row in review_rows if row.get("lesson")]
    return list(dict.fromkeys([*LESSONS_RETAINED, *row_lessons]))


def build_artifact(
    *,
    review_rows: Iterable[Mapping[str, object]],
    consolidation_note_path: str,
    manifest_text: str,
    manifest_block_added: bool,
    cited_recent_constraint_papers: Iterable[Mapping[str, object]] = (
        CITED_RECENT_CONSTRAINT_PAPERS
    ),
) -> dict[str, Any]:
    """REQ-REPORT-044: assemble the terminal HardNet++/DSP retirement artifact."""

    rows = [dict(row) for row in review_rows]
    papers = [dict(paper) for paper in cited_recent_constraint_papers]
    manifest_has_block = _manifest_contains_block(manifest_text)
    complete = bool(rows) and bool(papers) and manifest_has_block
    artifact: dict[str, Any] = {
        "experiment": EXPERIMENT,
        "schema": SCHEMA,
        "run_date": RUN_DATE,
        "project_root": PROJECT_ROOT_FOR_METADATA,
        "status": "complete" if complete else "blocked",
        "hardnet_dsp_experiments_reviewed": _experiment_ids(rows),
        "hardnet_dsp_review_rows": rows,
        "consolidation_note_path": consolidation_note_path,
        "hardnet_dsp_lineage_retired": complete,
        "exclusion_manifest_updated": manifest_has_block,
        "exclusion_manifest_block_added": manifest_block_added,
        "lessons_retained": _lessons_from_rows(rows),
        "cited_recent_constraint_papers": papers,
        "future_reopen_conditions": list(FUTURE_REOPEN_CONDITIONS),
        "honest_verdict": (
            "hardnet_dsp_lineage_retired_conservative_replay_retained_no_new_variants"
            if complete
            else "hardnet_dsp_retirement_blocked_missing_manifest_review_or_citations"
        ),
    }
    return artifact


def render_consolidation_note(
    review_rows: Iterable[Mapping[str, object]],
    artifact: Mapping[str, Any],
) -> str:
    """SCENARIO-REPORT-044: render the operator-facing retirement note."""

    lines = [
        "# HardNet++/DSP Repair Stack Retirement",
        "",
        f"Run date: `{RUN_DATE}`",
        f"Artifact: `results/experiment_1458_hardnet_dsp_repair_stack_consolidation.json`",
        "",
        "## Experiments Reviewed",
        "",
        "| experiment | title | verdict | evidence | lesson |",
        "|---|---|---|---|---|",
    ]
    for row in review_rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    _md_cell(row.get("experiment_id")),
                    _md_cell(row.get("title")),
                    _md_cell(row.get("verdict")),
                    _md_cell(row.get("evidence")),
                    _md_cell(row.get("lesson")),
                ]
            )
            + " |"
        )
    lines.extend(["", "## Cited Recent Constraint Papers", ""])
    for paper in artifact["cited_recent_constraint_papers"]:
        lines.append(
            f"- {paper['name']} ({paper['citation']}): {paper['lesson']}"
        )
    lines.extend(["", "## Hard Constraint Lesson", ""])
    for lesson in artifact["lessons_retained"]:
        lines.append(f"- {lesson}")
    lines.extend(
        [
            "",
            "## Why It Is Not Active Headline Scope",
            "",
            (
                "The hard-constraint result is retained, but the active lineage is "
                "closed. Exp 1305 showed the useful policy was conservative replay: "
                "stop after hard feasibility, continue only while violations remain, "
                "and route nonlinear residuals to HardNet++. Exp 1318 then showed a "
                "learned policy matched that conservative replay policy with "
                "hardnetpp_delta_over_replay_policy=0.0 on the held-out split. That "
                "does not justify another HardNet++/DSP variant during .112."
            ),
            "",
            "## Future Reopen Conditions",
            "",
        ]
    )
    for condition in artifact["future_reopen_conditions"]:
        lines.append(f"- {condition}")
    lines.extend(
        [
            "",
            "## Final Decision",
            "",
            (
                "The HardNet++/DSP repair stack is retired as active headline scope. "
                "The hard-constraint lesson stays in the project, while new "
                "HardNet++/DSP, FSNet/SnareNet, KKT-Hardnet, or DSP stop-policy "
                "variants are blocked unless the reopen conditions above are met."
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
    review_rows: Iterable[Mapping[str, object]] = HARDNET_DSP_REVIEW_ROWS,
    cited_recent_constraint_papers: Iterable[Mapping[str, object]] = (
        CITED_RECENT_CONSTRAINT_PAPERS
    ),
) -> dict[str, Any]:
    """REQ-REPORT-044: write bootstrap, note, manifest block, and terminal JSON."""

    _ = Path(root)
    out = Path(out_path)
    note = Path(note_path)
    manifest = Path(manifest_path)
    write_in_progress_artifact(out)
    manifest_text = manifest.read_text(encoding="utf-8") if manifest.exists() else ""
    updated_manifest, block_added = ensure_manifest_block(manifest_text)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(
        updated_manifest + ("" if updated_manifest.endswith("\n") else "\n"), encoding="utf-8"
    )
    artifact = build_artifact(
        review_rows=review_rows,
        consolidation_note_path=_relative_path(note),
        manifest_text=updated_manifest,
        manifest_block_added=block_added,
        cited_recent_constraint_papers=cited_recent_constraint_papers,
    )
    note.parent.mkdir(parents=True, exist_ok=True)
    note.write_text(render_consolidation_note(review_rows, artifact), encoding="utf-8")
    return _write_json(out, artifact)


if __name__ == "__main__":  # pragma: no cover - thin CLI convenience
    run()
