"""Build the Exp 1425 `.110` carry-forward activation audit.

Spec: REQ-REPORT-033, SCENARIO-REPORT-033.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
PROJECT_ROOT_FOR_METADATA = "/home/ianblenke/github.com/ianblenke/carnot"
RUN_DATE = "20260506"
PRIOR_MILESTONE = "2026.04.109"
TARGET_MILESTONE = "2026.04.110"
EXPERIMENT = "1425_109_carryforward_activation_audit"
SCHEMA = "milestone_110_carryforward_activation_audit_v1"

DEFAULT_OUT_PATH = REPO_ROOT / "results" / "experiment_1425_109_carryforward_activation_audit.json"
DEFAULT_MANIFEST_PATH = REPO_ROOT / "ops" / "milestone_110_carryforward_manifest.md"
RETRO_FILE = "experiment_1424_milestone_109_retro.json"
ROADMAP_PROPOSAL = "openspec/change-proposals/research-roadmap-vNEXT.md"

REQUIRED_ARTIFACT_FIELDS = {
    "status",
    "prior_milestone",
    "carryforward_manifest_path",
    "carryforward_manifest_complete",
    "carryforward_task_count",
    "same_verdict_retirement_rules",
    "forbidden_exact_reruns",
    "honest_verdict",
}

SOURCE_FILES = {
    "exp1414": "experiment_1414_certificate_llm_repair_executor_v1.json",
    "exp1415": "experiment_1415_dvi_v3_1508_fresh_cases.json",
    "exp1419": "experiment_1419_fullscale_pipeline_v3_repair_executor.json",
    "exp1420": "experiment_1420_dpo_verified_pairs_1508.json",
    "exp1421": "experiment_1421_test_suite_execution_debt_v1.json",
    "exp1423": "experiment_1423_process_reward_model_v1_fover_1508.json",
}

PRIOR_EXPERIMENT_TO_SOURCE = {
    "exp1414-certificate-llm-repair-executor-v1": "exp1414",
    "exp1415-dvi-v3-1508-fresh-cases": "exp1415",
    "exp1419-fullscale-pipeline-v3-repair-executor": "exp1419",
    "exp1420-dpo-verified-pairs-1508": "exp1420",
    "exp1421-test-suite-execution-debt-v1": "exp1421",
    "exp1423-process-reward-model-v1-fover-1508": "exp1423",
}

TRACK_MAPPINGS = {
    "repair-executor-v2-root-cause": {
        "track": "Repair executor v2 / full pipeline scale-up",
        "tasks": [
            "exp1427-repair-executor-rejection-ledger",
            "exp1428-dccd-schema-constrained-repair-v2",
            "exp1429-mcmc-constrained-repair-candidate-search",
            "exp1430-prm-guided-repair-selector",
            "exp1431-fullscale-pipeline-v4-micro-gated",
        ],
        "gate_rule": (
            "exp1431 stays gated until exp1428 proves repaired_case_success_rate > 0.0 "
            "and exp1430 is ready; no 200-case rerun before micro evidence."
        ),
        "retire_rule": (
            "Retire the exact zero-accepted-repair path if repair v2 again produces "
            "complete_repair_executor_no_successful_repairs or the exp1419 verdict."
        ),
    },
    "dvi-v3-nonforgetting-gate-fix": {
        "track": "DVI v3 nonforgetting",
        "tasks": ["exp1432-dvi-v3-nonforgetting-replay-balanced"],
        "gate_rule": (
            "Deploy only when dvi_v3_deployed=true, nonforgetting_rate >= 0.99, "
            "and AUROC does not regress below DVI v2."
        ),
        "retire_rule": (
            "Retire this DVI v3 repair variant if the same "
            "dvi_v3_blocked_nonforgetting_below_gate verdict recurs."
        ),
    },
    "fr11-v6-after-dvi-v3": {
        "track": "FR-11 v6 continuous self-learning",
        "tasks": ["exp1433-fr11-self-learning-v6-dvi-v3-gated"],
        "gate_rule": "Launch only after exp1432 records dvi_v3_deployed=true.",
        "retire_rule": (
            "Do not retire FR-11 on an upstream DVI gate block; keep it gated until "
            "deployable DVI evidence exists."
        ),
    },
    "dpo-headline-validation-or-finetune-support": {
        "track": "DPO provenance",
        "tasks": ["exp1435-dpo-headline-provenance-audit"],
        "gate_rule": (
            "Headline provenance requires direct local adapter/fine-tune support; "
            "otherwise the track is relabeled reranker-only."
        ),
        "retire_rule": (
            "If direct local DPO remains unsupported, retire headline DPO wording and "
            "carry only the reranker benchmark."
        ),
    },
    "test-suite-remaining-debt": {
        "track": "Full Python suite and spec-coverage debt",
        "tasks": ["exp1426-test-suite-remaining-debt-cluster-map"],
        "gate_rule": (
            "Diagnostic scope only: map remaining clusters and recommend one next bounded "
            "fix without reopening the fixed embedding-store cluster."
        ),
        "retire_rule": (
            "Do not retire the whole test-debt track on another red full-suite result; "
            "split by named failure cluster."
        ),
    },
    "prm-label-completion": {
        "track": "PRM label completion",
        "tasks": ["exp1434-fover-prm-label-completion-v2"],
        "gate_rule": (
            "Fill at least 478 missing labels or write an exact residual blocker ledger "
            "before any 1508-trace PRM claim."
        ),
        "retire_rule": (
            "Retire the 1508-trace PRM claim if labels remain missing without a blocker "
            "ledger; keep measured PRM v1 as partial evidence only."
        ),
    },
}


def _write_json(path: Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    artifact = dict(payload)
    path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return artifact


def write_in_progress_artifact(out_path: Path | str = DEFAULT_OUT_PATH) -> dict[str, Any]:
    """REQ-REPORT-033: create the bootstrap artifact before reading source evidence."""

    artifact = {field: None for field in REQUIRED_ARTIFACT_FIELDS}
    artifact.update(
        {
            "experiment": EXPERIMENT,
            "schema": SCHEMA,
            "run_date": RUN_DATE,
            "project_root": PROJECT_ROOT_FOR_METADATA,
            "target_milestone": TARGET_MILESTONE,
            "status": "in_progress",
        }
    )
    return _write_json(Path(out_path), artifact)


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _read_text(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8")


def _load_sources(root: Path) -> dict[str, dict[str, Any]]:
    sources: dict[str, dict[str, Any]] = {}
    for source_id, filename in SOURCE_FILES.items():
        payload = _read_json(root / "results" / filename)
        if payload is not None:
            sources[source_id] = payload
    return sources


def _relative_path(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        parts = path.parts
        if "ops" in parts:
            return str(Path(*parts[parts.index("ops") :]))
        if "results" in parts:
            return str(Path(*parts[parts.index("results") :]))
        return path.name


def _honest_verdict(payload: Mapping[str, Any] | None) -> str | None:
    if not payload:
        return None
    verdict = payload.get("honest_verdict")
    return None if verdict is None else str(verdict)


def _source_id_for_prior(prior_experiment_id: str) -> str | None:
    if prior_experiment_id in PRIOR_EXPERIMENT_TO_SOURCE:
        return PRIOR_EXPERIMENT_TO_SOURCE[prior_experiment_id]
    prefix = prior_experiment_id.split("-", 1)[0]
    return prefix if prefix in SOURCE_FILES else None


def _exact_prior_verdict(prior: Mapping[str, Any], sources: Mapping[str, Mapping[str, Any]]) -> str:
    source_id = _source_id_for_prior(str(prior.get("experiment_id", "")))
    source_verdict = _honest_verdict(sources.get(source_id or ""))
    if source_verdict:
        return source_verdict
    return str(prior.get("verdict", "missing_prior_verdict"))


def _source_path_for_prior(prior_experiment_id: str) -> str | None:
    source_id = _source_id_for_prior(prior_experiment_id)
    if source_id is None:
        return None
    return f"results/{SOURCE_FILES[source_id]}"


def _source_metric_summary(source_id: str | None, payload: Mapping[str, Any] | None) -> str:
    if source_id == "exp1414":
        return (
            f"repaired_cases_successful={payload.get('repaired_cases_successful')}, "
            f"repaired_case_success_rate={payload.get('repaired_case_success_rate')}"
        )
    if source_id == "exp1415":
        return (
            f"nonforgetting_rate={payload.get('nonforgetting_rate')}, "
            f"dvi_v3_deployed={payload.get('dvi_v3_deployed')}"
        )
    if source_id == "exp1419":
        return (
            f"cases_evaluated={payload.get('cases_evaluated')}, "
            f"repaired_cases_successful={payload.get('repaired_cases_successful')}, "
            f"full_pipeline_pass_rate={payload.get('full_pipeline_pass_rate')}"
        )
    if source_id == "exp1420":
        return (
            f"dpo_full_finetune_performed={payload.get('dpo_full_finetune_performed')}, "
            f"headline_result_allowed={payload.get('headline_result_allowed')}"
        )
    if source_id == "exp1421":
        return f"remaining_debt={bool(payload.get('remaining_debt'))}"
    if source_id == "exp1423":
        return (
            f"training_traces_used={payload.get('training_traces_used')}, "
            f"missing_trace_labels={payload.get('missing_trace_labels')}"
        )
    return "source metrics unavailable"


def _prior_evidence(
    task: Mapping[str, Any],
    sources: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    evidence: list[dict[str, Any]] = []
    for prior in task.get("prior_failures", []):
        if not isinstance(prior, Mapping):
            continue
        experiment_id = str(prior.get("experiment_id", "missing_experiment_id"))
        source_id = _source_id_for_prior(experiment_id)
        payload = sources.get(source_id or "")
        evidence.append(
            {
                "experiment_id": experiment_id,
                "source_artifact": _source_path_for_prior(experiment_id),
                "prior_verdict": _exact_prior_verdict(prior, sources),
                "retro_verdict": str(prior.get("verdict", "missing_prior_verdict")),
                "metrics": _source_metric_summary(source_id, payload),
                "retro_retire_if_same_verdict": bool(prior.get("retire_if_same_verdict")),
            }
        )
    return evidence


def _row_for_task(
    task: Mapping[str, Any],
    sources: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    track_id = str(task.get("id", "missing_track_id"))
    mapping = TRACK_MAPPINGS.get(track_id)
    evidence = _prior_evidence(task, sources)
    if mapping is None:
        title = str(task.get("title", track_id))
        return {
            "track_id": track_id,
            "track": f"{title} ({track_id})",
            "prior_evidence": evidence,
            "mapped_110_tasks": [],
            "gate_rule": "unmapped carry-forward track",
            "retire_if_same_verdict_rule": "unmapped carry-forward track",
            "mapped": False,
        }
    return {
        "track_id": track_id,
        "track": mapping["track"],
        "prior_evidence": evidence,
        "mapped_110_tasks": list(mapping["tasks"]),
        "gate_rule": mapping["gate_rule"],
        "retire_if_same_verdict_rule": mapping["retire_rule"],
        "mapped": True,
    }


def _flatten_retirement_rules(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rules: list[dict[str, Any]] = []
    for row in rows:
        for evidence in row["prior_evidence"]:
            rules.append(
                {
                    "track_id": row["track_id"],
                    "experiment_id": evidence["experiment_id"],
                    "source_artifact": evidence["source_artifact"],
                    "prior_verdict": evidence["prior_verdict"],
                    "retro_retire_if_same_verdict": evidence["retro_retire_if_same_verdict"],
                    "mapped_110_tasks": row["mapped_110_tasks"],
                    "retire_if_same_verdict_rule": row["retire_if_same_verdict_rule"],
                }
            )
    return rules


def _forbidden_exact_reruns(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not any(row["track_id"] == "repair-executor-v2-root-cause" for row in rows):
        return []
    return [
        {
            "experiment_id": "exp1419-fullscale-pipeline-v3-repair-executor",
            "result_artifact": "results/experiment_1419_fullscale_pipeline_v3_repair_executor.json",
            "forbidden_scope": (
                "exact exp1419 200-case full-scale pipeline rerun without nonzero "
                "accepted repair evidence"
            ),
            "required_unlock_evidence": (
                "exp1428.repaired_case_success_rate > 0.0 and downstream validation uses "
                "the exp1431 micro-gated path before any larger scale run"
            ),
            "prior_verdict": "not_headline_full_pipeline_below_0_40",
            "retire_if_same_verdict": True,
        }
    ]


def _md_cell(value: object) -> str:
    return str(value).replace("\n", " ").replace("|", "\\|")


def _render_prior_evidence(evidence: list[dict[str, Any]]) -> str:
    parts = []
    for item in evidence:
        source = item["source_artifact"] or "retro-only evidence"
        parts.append(
            f"{item['experiment_id']} ({source}): {item['prior_verdict']} [{item['metrics']}]"
        )
    return "; ".join(parts) if parts else "No prior evidence recorded"


def render_manifest(rows: list[dict[str, Any]], forbidden: list[dict[str, Any]]) -> str:
    """REQ-REPORT-033: render the operator-readable carry-forward table."""

    lines = [
        "# Milestone .110 Carry-Forward Manifest",
        "",
        f"Prior milestone: `{PRIOR_MILESTONE}`",
        f"Target milestone: `{TARGET_MILESTONE}`",
        f"Run date: `{RUN_DATE}`",
        "",
        "| track | prior evidence | .110 task | gate rule | retire-if-same-verdict rule |",
        "|---|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    _md_cell(row["track"]),
                    _md_cell(_render_prior_evidence(row["prior_evidence"])),
                    _md_cell(", ".join(row["mapped_110_tasks"]) or "EXPLICITLY UNMAPPED"),
                    _md_cell(row["gate_rule"]),
                    _md_cell(row["retire_if_same_verdict_rule"]),
                ]
            )
            + " |"
        )
    lines.extend(["", "## Forbidden Exact Reruns", ""])
    if forbidden:
        for item in forbidden:
            lines.append(
                "- exp1419 200-case full-scale pipeline rerun without nonzero accepted "
                "repair evidence is forbidden. Required unlock evidence: "
                f"{item['required_unlock_evidence']}."
            )
    else:
        lines.append("- None recorded.")
    lines.extend(
        [
            "",
            "## No-Change Confirmation",
            "",
            "- scripts/research_conductor.py: no activation-audit changes needed",
            "- research-roadmap.yaml: no activation-audit changes needed",
            "",
        ]
    )
    return "\n".join(lines)


def build_artifact(
    *,
    retro: Mapping[str, Any],
    sources: Mapping[str, Mapping[str, Any]],
    manifest_path: str,
    roadmap_text: str,
) -> tuple[dict[str, Any], str]:
    """REQ-REPORT-033: map each `.109` carry-forward track into `.110` scope."""

    tasks = [task for task in retro.get("carry_forward_tasks", []) if isinstance(task, Mapping)]
    rows = [_row_for_task(task, sources) for task in tasks]
    unmapped_tracks = [row["track_id"] for row in rows if not row["mapped"]]
    forbidden = _forbidden_exact_reruns(rows)
    complete = bool(rows) and not unmapped_tracks and bool(forbidden)
    manifest = render_manifest(rows, forbidden)
    missing_roadmap_task_ids = [
        task_id
        for row in rows
        for task_id in row["mapped_110_tasks"]
        if task_id not in roadmap_text
    ]
    artifact = {
        "experiment": EXPERIMENT,
        "schema": SCHEMA,
        "run_date": RUN_DATE,
        "project_root": PROJECT_ROOT_FOR_METADATA,
        "target_milestone": TARGET_MILESTONE,
        "status": "complete" if complete else "blocked",
        "prior_milestone": str(retro.get("milestone", PRIOR_MILESTONE)),
        "prior_milestone_honest_verdict": retro.get("honest_verdict"),
        "carryforward_manifest_path": manifest_path,
        "carryforward_manifest_complete": complete,
        "carryforward_task_count": len(rows),
        "manifest_rows": rows,
        "same_verdict_retirement_rules": _flatten_retirement_rules(rows),
        "forbidden_exact_reruns": forbidden,
        "source_artifacts_checked": [
            {
                "experiment_id": source_id,
                "path": f"results/{filename}",
                "exists": source_id in sources,
                "honest_verdict": _honest_verdict(sources.get(source_id)),
            }
            for source_id, filename in SOURCE_FILES.items()
        ],
        "roadmap_mapping_check": {
            "roadmap_proposal_present": bool(roadmap_text),
            "missing_mapped_task_ids_in_proposal": missing_roadmap_task_ids,
        },
        "no_change_confirmations": {
            "scripts/research_conductor.py": "no activation-audit changes needed",
            "research-roadmap.yaml": "no activation-audit changes needed",
        },
        "unmapped_tracks": unmapped_tracks,
        "honest_verdict": (
            f"milestone_110_carryforward_manifest_complete_{len(rows)}_tracks_"
            "exp1419_exact_rerun_forbidden"
            if complete
            else "carryforward_manifest_incomplete_unmapped_tracks"
        ),
    }
    return artifact, manifest


def run(
    root: Path | str = REPO_ROOT,
    out_path: Path | str = DEFAULT_OUT_PATH,
    manifest_path: Path | str = DEFAULT_MANIFEST_PATH,
) -> dict[str, Any]:
    """REQ-REPORT-033: write bootstrap, manifest, and terminal JSON artifact."""

    root_path = Path(root)
    out = Path(out_path)
    manifest_out = Path(manifest_path)
    write_in_progress_artifact(out)
    retro = _read_json(root_path / "results" / RETRO_FILE) or {
        "milestone": PRIOR_MILESTONE,
        "carry_forward_tasks": [],
    }
    artifact, manifest = build_artifact(
        retro=retro,
        sources=_load_sources(root_path),
        manifest_path=_relative_path(manifest_out),
        roadmap_text="\n".join(
            [
                _read_text(root_path / ROADMAP_PROPOSAL),
                _read_text(root_path / "research-roadmap.yaml"),
            ]
        ),
    )
    manifest_out.parent.mkdir(parents=True, exist_ok=True)
    manifest_out.write_text(manifest, encoding="utf-8")
    return _write_json(out, artifact)


if __name__ == "__main__":  # pragma: no cover - thin CLI convenience
    run()
