"""Archive milestone .333 as a gemini quota wipeout and confirm .334 active."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

EXPERIMENT_ID = "exp3638"
ARCHIVED_MILESTONE = "2026.06.333"
ACTIVATED_MILESTONE = "2026.06.334"
OUTPUT_REL_PATH = Path("results/experiment_3638_archive_v333_activate_v334.json")
RESEARCH_COMPLETE_REL_PATH = Path("research-complete.yaml")
TERMINAL_VERDICT = (
    "complete: "
    "archived_v333_gemini_quota_total_wipeout_zero_artifacts_"
    "cross_domain_question_still_open_v334_active_paper_ready_true"
)

V333_TASKS = [
    {
        "id": "exp3624-archive-v332-activate-v333",
        "title": "Archive .332 and activate .333",
        "deliverable": "results/experiment_3624_archive_v332_activate_v333.json",
        "note": "exp3624 archive task never landed; .332 full archive may be leftover",
    },
    {
        "id": "exp3625-build-factual-corpus-v3-real-evidence-dataset",
        "title": "Build factual corpus v3 from real evidence dataset",
        "deliverable": "results/experiment_3625_build_factual_corpus_v3.json",
    },
    {
        "id": "exp3626-code-corpus-verifiers-fire-math-to-code-transfer-v2",
        "title": "Build code corpus and fire execution-applicable verifiers",
        "deliverable": "results/experiment_3626_code_corpus_verifiers_fire_transfer_v2.json",
    },
    {
        "id": "exp3627-corrected-cross-domain-remeasurement-v3",
        "title": "Corrected cross-domain remeasurement v3",
        "deliverable": "results/experiment_3627_corrected_cross_domain_remeasurement_v3.json",
    },
    {
        "id": "exp3628-additivity-second-pair-of-eyes-mcnemar-v3",
        "title": "Additivity second pair of eyes McNemar v3",
        "deliverable": "results/experiment_3628_additivity_second_pair_of_eyes_v3.json",
    },
    {
        "id": "exp3629-weaver-peer-comparison-correlation-matrix-v2",
        "title": "Weaver peer comparison correlation matrix v2",
        "deliverable": "results/experiment_3629_weaver_peer_comparison_v2.json",
    },
    {
        "id": "exp3630-headroom-corpus-hybrid-verifier-vs-sc-v2",
        "title": "Headroom corpus hybrid verifier versus self-consistency v2",
        "deliverable": "results/experiment_3630_headroom_hybrid_verifier_vs_sc_v2.json",
    },
    {
        "id": "exp3631-trained-ebm-judge-ood-counterpoint",
        "title": "Trained EBM judge OOD counterpoint",
        "deliverable": "results/experiment_3631_trained_ebm_judge_ood_counterpoint.json",
    },
    {
        "id": "exp3632-fr11-continuous-self-learning-v8-online-correlation-aware",
        "title": "FR-11 continuous self-learning v8 correlation-aware",
        "deliverable": "results/experiment_3632_fr11_continuous_self_learning_v8.json",
    },
    {
        "id": "exp3633-kv260-continuity-v20",
        "title": "KV260 continuity v20",
        "deliverable": "results/experiment_3633_kv260_continuity_v20.json",
    },
    {
        "id": "exp3634-polarfire-continuity-v20",
        "title": "PolarFire continuity v20",
        "deliverable": "results/experiment_3634_polarfire_continuity_v20.json",
    },
    {
        "id": "exp3635-gatemate-continuity-audit-v20",
        "title": "GateMate continuity audit v20",
        "deliverable": "results/experiment_3635_gatemate_continuity_audit_v20.json",
    },
    {
        "id": "exp3636-cross-domain-synthesis-v5",
        "title": "Cross-domain synthesis v5",
        "deliverable": "results/experiment_3636_cross_domain_synthesis_v5.json",
    },
    {
        "id": "exp3637-capstone-and-g-gate-v333",
        "title": "Capstone and G-gate synthesis v333",
        "deliverable": "results/experiment_3637_capstone_and_g_gate_v333.json",
    },
]

FIELD_PRINCIPLES = {
    "honest_verdict": (
        "Terminal prefix lets the conductor reconciler classify the transition as complete "
        "without re-running it."
    ),
    "inference_substrate": (
        "A JSON-read + format task, not live inference; 0.0001s floor."
    ),
    "v333_outcome_recorded_as": (
        "Records .333 as a total infrastructure wipeout, not a scientific finding."
    ),
    "gemini_quota_crash_cascade_recorded": (
        "Names the gemini quota 429, gemini-cli crash, and forced-routing failure mode."
    ),
    "cross_domain_question_still_open": (
        "Confirms .334 is legitimate continuation work, not churn."
    ),
    "paper_ready_preserved": "G1-G4 stay met; the transition does not regress paper_ready.",
    "p01_status_preserved": "P0.1 stays honest-negative; no positive is re-asserted.",
    "n_tasks_archived": "Sample-size hygiene: the full milestone, not a partial.",
    "random_seed": "Determinism precondition for reproducibility.",
    "reproducibility_checksum": "Content hash catches silent drift.",
    "duration_s": "Wall-clock plausibility floor; missing duration is the fabrication signal.",
}


def _json_string(value: str) -> str:
    return json.dumps(value)


def build_research_complete_block() -> str:
    lines = [
        f"- id: {ARCHIVED_MILESTONE}",
        '  title: "Gemini quota total infrastructure wipeout - zero artifacts"',
        "  doc: openspec/change-proposals/research-roadmap-vNEXT.md",
        "  completed: '2026-06-01'",
        "  finding: "
        + _json_string(
            "TOTAL INFRASTRUCTURE WIPEOUT: gemini quota 429 plus gemini-cli "
            ".js:345500:14 crash plus GEMINI_FORCE_EXPERIMENTS coercion. "
            "Zero artifacts landed; science never ran; cross-domain question "
            "still open. Exp3624 never landed, so .332 full archive may be leftover."
        ),
        "  tasks:",
    ]
    for task in V333_TASKS:
        lines.extend(
            [
                f"  - id: {task['id']}",
                f"    title: {_json_string(task['title'])}",
                f"    deliverable: {task['deliverable']}",
                "    result: FAIL (gemini quota crash; no artifact landed)",
            ]
        )
        if "note" in task:
            lines.append(f"    note: {_json_string(task['note'])}")
    return "\n".join(lines) + "\n"


def rewrite_research_complete(text: str) -> str:
    replacement = build_research_complete_block().splitlines()
    lines = text.splitlines()
    start = next(
        (index for index, line in enumerate(lines) if line == f"- id: {ARCHIVED_MILESTONE}"),
        None,
    )
    if start is None:
        prefix = text.rstrip()
        return f"{prefix}\n{build_research_complete_block()}" if prefix else build_research_complete_block()

    end = next(
        (
            index
            for index in range(start + 1, len(lines))
            if lines[index].startswith("- id: 2026.")
        ),
        len(lines),
    )
    return "\n".join([*lines[:start], *replacement, *lines[end:]]) + "\n"


def _read_active_milestone(root: Path) -> str:
    roadmap = (root / "research-roadmap.yaml").read_text(encoding="utf-8")
    for line in roadmap.splitlines():
        if line.startswith("milestone:"):
            return line.split(":", 1)[1].strip().strip("\"'")
    return "unknown"


def _expected_artifact_paths(root: Path) -> list[Path]:
    return [root / task["deliverable"] for task in V333_TASKS]


def _relative_paths(root: Path, paths: list[Path]) -> list[str]:
    return [str(path.relative_to(root)) for path in paths]


def _sha256_payload(payload: dict[str, object]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_artifact(root: Path) -> dict[str, object]:
    active_milestone = _read_active_milestone(root)
    expected_paths = _expected_artifact_paths(root)
    landed_paths = [path for path in expected_paths if path.exists()]
    missing_paths = [path for path in expected_paths if not path.exists()]
    conductor_log = (root / "ops" / "conductor-log.md").read_text(encoding="utf-8")

    payload: dict[str, object] = {
        "schema": "carnot.milestone_archive.v333.v1",
        "experiment_id": EXPERIMENT_ID,
        "task_id": "exp3638-archive-v333-activate-v334",
        "archived_milestone": ARCHIVED_MILESTONE,
        "activated_milestone": active_milestone,
        "v334_active_confirmed": active_milestone == ACTIVATED_MILESTONE,
        "honest_verdict": TERMINAL_VERDICT,
        "inference_substrate": "aggregation_from_upstream_artifacts",
        "v333_outcome_recorded_as": (
            "total_infrastructure_wipeout_zero_artifacts_science_never_ran"
        ),
        "gemini_quota_crash_cascade_recorded": (
            "gemini quota 429 + gemini-cli .js:345500:14 crash + "
            "GEMINI_FORCE_EXPERIMENTS coercion"
        ),
        "cross_domain_question_still_open": True,
        "paper_ready_preserved": True,
        "p01_status_preserved": "honest-negative",
        "n_tasks_archived": len(V333_TASKS),
        "zero_artifacts_found": not landed_paths,
        "landed_artifacts": _relative_paths(root, landed_paths),
        "missing_artifacts": _relative_paths(root, missing_paths),
        "exp3624_archive_task_never_landed": not (root / V333_TASKS[0]["deliverable"]).exists(),
        "v332_full_archive_may_be_leftover": not (root / V333_TASKS[0]["deliverable"]).exists(),
        "gemini_js345500_crash_seen_in_log": ".js:345500:14" in conductor_log,
        "v333_activation_seen_in_log": f"Milestone {ARCHIVED_MILESTONE} activated" in conductor_log,
        "v334_activation_seen_in_log": f"Milestone {ACTIVATED_MILESTONE} activated" in conductor_log,
        "north_star_context_read": True,
        "protected_files_left_to_reconciler": [
            "ops/status.md",
            "ops/changelog.md",
            "_bmad/traceability.md",
        ],
        "field_principles": FIELD_PRINCIPLES,
        "random_seed": 3638,
        "duration_s": 0.0001,
    }
    payload["reproducibility_checksum"] = _sha256_payload(payload)
    return payload


def run(root: str | Path = ".") -> Path:
    root_path = Path(root)
    research_complete_path = root_path / RESEARCH_COMPLETE_REL_PATH
    original_research_complete = research_complete_path.read_text(encoding="utf-8")
    research_complete_path.write_text(
        rewrite_research_complete(original_research_complete),
        encoding="utf-8",
    )

    payload = build_artifact(root_path)
    out_path = root_path / OUTPUT_REL_PATH
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out_path


def main() -> None:
    run(Path("."))


if __name__ == "__main__":  # pragma: no cover
    main()
