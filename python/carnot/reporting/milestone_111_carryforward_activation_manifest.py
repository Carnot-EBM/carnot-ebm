"""Build the Exp 1439 `.111` carry-forward activation manifest.

Spec: REQ-REPORT-036, SCENARIO-REPORT-036.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
PROJECT_ROOT_FOR_METADATA = "/home/ianblenke/github.com/ianblenke/carnot"
RUN_DATE = "20260506"
PRIOR_MILESTONE = "2026.04.110"
TARGET_MILESTONE = "2026.04.111"
EXPERIMENT = "1439_110_carryforward_activation_manifest"
SCHEMA = "milestone_111_carryforward_activation_manifest_v1"

DEFAULT_OUT_PATH = (
    REPO_ROOT / "results" / "experiment_1439_110_carryforward_activation_manifest.json"
)
DEFAULT_MANIFEST_PATH = REPO_ROOT / "ops" / "milestone_111_carryforward_manifest.md"
RETRO_FILE = "experiment_1438_milestone_110_retro.json"
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
    "exp1426": "experiment_1426_test_suite_remaining_debt_cluster_map.json",
    "exp1428": "experiment_1428_dccd_schema_constrained_repair_v2.json",
    "exp1429": "experiment_1429_mcmc_constrained_repair_candidate_search.json",
    "exp1430": "experiment_1430_prm_guided_repair_selector.json",
    "exp1431": "experiment_1431_fullscale_pipeline_v4_micro_gated.json",
    "exp1433": "experiment_1433_fr11_self_learning_v6_dvi_v3_gated.json",
    "exp1435": "experiment_1435_dpo_headline_provenance_audit.json",
    "exp1437": "experiment_1437_discrete_sb_kv260_rtl_lint_sim.json",
}

TRACK_MAPPINGS = {
    "repair_v2_live_sota_headline_scaleup": {
        "track": "Live-SOTA repair provenance and headline-eligible scale-up",
        "tasks": ["exp1442", "exp1443", "exp1444", "exp1445"],
        "gate_rule": (
            "exp1442 must prove local_sota_runtime_ready=true before exp1443; exp1445 "
            "requires nonzero live repair success and energy-reranker readiness."
        ),
        "retire_rule": (
            "If the next repair scale attempt again reports prototype/no-live-SOTA or "
            "no-headline-scaleup with the same path, retire headline repair scale claims "
            "until runtime provenance changes."
        ),
    },
    "fr11_positive_growth_followup": {
        "track": "FR-11 positive promoted growth after deployed DVI v3",
        "tasks": ["exp1446", "exp1447"],
        "gate_rule": (
            "exp1446 must identify the zero-growth root cause before exp1447 changes "
            "promotion thresholds, candidate generation, or memory policy."
        ),
        "retire_rule": (
            "If DVI remains deployed and FR-11 again reports zero promoted growth, retire "
            "this FR-11 v6/v7 variant and require a new root-cause plan."
        ),
    },
    "test_debt_spec_coverage_cluster": {
        "track": "Spec-coverage traceability metadata debt",
        "tasks": ["exp1440"],
        "gate_rule": (
            "Fix the named spec_coverage_traceability_metadata cluster first and do not "
            "reopen the already-fixed embedding-store cluster."
        ),
        "retire_rule": (
            "Do not retire whole test debt on another red result; split persistent "
            "failures by named cluster."
        ),
    },
    "dpo_adapter_or_reranker_only": {
        "track": "DPO provenance limits",
        "tasks": [
            "NON-HEADLINE RETIREMENT: no .111 DPO headline task until direct local "
            "adapter or conversion tooling exists"
        ],
        "gate_rule": (
            "Headline DPO remains closed unless concrete local adapter/conversion tooling "
            "is named before execution."
        ),
        "retire_rule": (
            "If direct GGUF fine-tune support remains absent, retire DPO headline wording "
            "and preserve reranker-only status."
        ),
    },
    "hardware_rtl_source_before_lint_sim": {
        "track": "Discrete SB RTL source before lint/simulation",
        "tasks": ["exp1441", "exp1451"],
        "gate_rule": (
            "exp1451 may rerun lint/simulation only after exp1441 creates "
            "hardware/kv260/discrete_sb_256.v and a testbench."
        ),
        "retire_rule": (
            "If the exact lint/sim rerun still lacks the RTL source, retire that rerun "
            "and require source implementation first."
        ),
    },
    "prm_selector_no_improvement": {
        "track": "PRM selector no-improvement path",
        "tasks": ["exp1448"],
        "gate_rule": (
            "exp1448 must use PRM v2 labels and online stepwise scoring before any "
            "claim that PRM selection improves repair acceptance."
        ),
        "retire_rule": (
            "Retire the exact PRM v1 no-improvement selector rerun; a future PRM claim "
            "must change labels, scoring policy, or candidate source."
        ),
    },
}


def _write_json(path: Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    artifact = dict(payload)
    path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return artifact


def write_in_progress_artifact(out_path: Path | str = DEFAULT_OUT_PATH) -> dict[str, Any]:
    """REQ-REPORT-036: record a bootstrap artifact before evidence is loaded."""

    artifact: dict[str, Any] = {field: None for field in REQUIRED_ARTIFACT_FIELDS}
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
    if prior_experiment_id in SOURCE_FILES:
        return prior_experiment_id
    prefix = prior_experiment_id.split("-", 1)[0]
    return prefix if prefix in SOURCE_FILES else None


def _source_path_for_prior(prior_experiment_id: str) -> str | None:
    source_id = _source_id_for_prior(prior_experiment_id)
    if source_id is None:
        return None
    return f"results/{SOURCE_FILES[source_id]}"


def _exact_prior_verdict(prior: Mapping[str, Any], sources: Mapping[str, Mapping[str, Any]]) -> str:
    source_id = _source_id_for_prior(str(prior.get("experiment_id", "")))
    source_verdict = _honest_verdict(sources.get(source_id or ""))
    if source_verdict:
        return source_verdict
    return str(prior.get("verdict", "missing_prior_verdict"))


def _source_metric_summary(source_id: str | None, payload: Mapping[str, Any] | None) -> str:
    if source_id == "exp1426":
        return (
            f"spec_coverage_debt_count={payload.get('spec_coverage_debt_count')}, "
            f"next_cluster_recommended={payload.get('next_cluster_recommended')}"
        )
    if source_id == "exp1428":
        return (
            f"repaired_cases_successful={payload.get('repaired_cases_successful')}, "
            f"repaired_case_success_rate={payload.get('repaired_case_success_rate')}, "
            f"local_sota_model_inference_used={payload.get('local_sota_model_inference_used')}"
        )
    if source_id == "exp1430":
        return (
            f"selection_improvement_pp={payload.get('selection_improvement_pp')}, "
            f"prm_guided_selection_ready={payload.get('prm_guided_selection_ready')}"
        )
    if source_id == "exp1431":
        return (
            f"cases_evaluated={payload.get('cases_evaluated')}, "
            f"full_pipeline_pass_rate={payload.get('full_pipeline_pass_rate')}, "
            "runtime_evidence_allows_headline_scaleup="
            f"{payload.get('runtime_evidence_allows_headline_scaleup')}"
        )
    if source_id == "exp1433":
        return (
            f"v6_new_promoted_count={payload.get('v6_new_promoted_count')}, "
            f"self_learning_delta_overall={payload.get('self_learning_delta_overall')}, "
            f"headline_result_allowed={payload.get('headline_result_allowed')}"
        )
    if source_id == "exp1435":
        return (
            f"direct_gguf_finetune_supported={payload.get('direct_gguf_finetune_supported')}, "
            f"reranker_track_relabelled={payload.get('reranker_track_relabelled')}"
        )
    if source_id == "exp1437":
        return (
            f"rtl_lint_complete={payload.get('rtl_lint_complete')}, "
            f"simulation_complete={payload.get('simulation_complete')}, "
            f"hardware_claim_allowed={payload.get('hardware_claim_allowed')}"
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


def _synthetic_prm_task(
    retro: Mapping[str, Any],
    sources: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any] | None:
    verdict = _honest_verdict(sources.get("exp1430"))
    summary = str(retro.get("prm_verdict", {}).get("summary", ""))
    if verdict is None and "no_improvement" not in summary:
        return None
    return {
        "id": "prm_selector_no_improvement",
        "title": "Retire exact PRM v1 selector no-improvement reruns",
        "prior_failures": [
            {
                "experiment_id": "exp1430",
                "verdict": verdict or summary or "prm_selector_no_improvement",
                "retire_if_same_verdict": True,
            }
        ],
    }


def _tasks_from_retro(
    retro: Mapping[str, Any],
    sources: Mapping[str, Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    tasks = [task for task in retro.get("carry_forward_tasks", []) if isinstance(task, Mapping)]
    if not any(task.get("id") == "prm_selector_no_improvement" for task in tasks):
        prm_task = _synthetic_prm_task(retro, sources)
        if prm_task is not None:
            tasks.append(prm_task)
    return tasks


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
            "mapped_111_tasks": [],
            "gate_rule": "unmapped carry-forward track",
            "retire_if_same_verdict_rule": "unmapped carry-forward track",
            "mapped": False,
        }
    return {
        "track_id": track_id,
        "track": mapping["track"],
        "prior_evidence": evidence,
        "mapped_111_tasks": list(mapping["tasks"]),
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
                    "mapped_111_tasks": row["mapped_111_tasks"],
                    "retire_if_same_verdict_rule": row["retire_if_same_verdict_rule"],
                }
            )
    return rules


def _prior_verdicts_for_track(rows: list[dict[str, Any]], track_id: str) -> list[str]:
    for row in rows:
        if row["track_id"] == track_id:
            return [str(item["prior_verdict"]) for item in row["prior_evidence"]]
    return []  # pragma: no cover - callers ask only for tracks already present.


def _forbidden_exact_reruns(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    forbidden: list[dict[str, Any]] = []
    if any(row["track_id"] == "repair_v2_live_sota_headline_scaleup" for row in rows):
        forbidden.append(
            {
                "forbidden_scope_id": "prototype_repair_scaleup",
                "track_id": "repair_v2_live_sota_headline_scaleup",
                "forbidden_scope": (
                    "exact prototype-only repair scale-up without live local SOTA "
                    "inference and headline runtime provenance"
                ),
                "required_unlock_evidence": (
                    "exp1442.local_sota_runtime_ready=true and exp1443.live_sota_"
                    "inference_used=true before exp1445 scale-up"
                ),
                "prior_verdicts": _prior_verdicts_for_track(
                    rows, "repair_v2_live_sota_headline_scaleup"
                ),
                "retire_if_same_verdict": True,
            }
        )
    if any(row["track_id"] == "fr11_positive_growth_followup" for row in rows):
        forbidden.append(
            {
                "forbidden_scope_id": "fr11_zero_growth",
                "track_id": "fr11_positive_growth_followup",
                "forbidden_scope": (
                    "exact FR-11 zero-growth rerun with deployed DVI and unchanged "
                    "promotion or memory policy"
                ),
                "required_unlock_evidence": (
                    "exp1446.fr11_zero_growth_root_cause_identified=true and exp1447 "
                    "changes promotion thresholds, candidate generation, or memory policy"
                ),
                "prior_verdicts": _prior_verdicts_for_track(rows, "fr11_positive_growth_followup"),
                "retire_if_same_verdict": True,
            }
        )
    if any(row["track_id"] == "prm_selector_no_improvement" for row in rows):
        forbidden.append(
            {
                "forbidden_scope_id": "prm_v1_no_improvement",
                "track_id": "prm_selector_no_improvement",
                "forbidden_scope": (
                    "exact PRM v1 no-improvement selector rerun on the same prototype "
                    "candidate pool"
                ),
                "required_unlock_evidence": (
                    "exp1448 uses PRM v2 labels and a changed online process-reward scoring policy"
                ),
                "prior_verdicts": _prior_verdicts_for_track(rows, "prm_selector_no_improvement"),
                "retire_if_same_verdict": True,
            }
        )
    if any(row["track_id"] == "hardware_rtl_source_before_lint_sim" for row in rows):
        forbidden.append(
            {
                "forbidden_scope_id": "missing_source_rtl_lint_sim",
                "track_id": "hardware_rtl_source_before_lint_sim",
                "forbidden_scope": (
                    "exact missing-source RTL lint/sim path before "
                    "hardware/kv260/discrete_sb_256.v exists"
                ),
                "required_unlock_evidence": (
                    "exp1441.rtl_source_created=true before exp1451 reruns lint or simulation"
                ),
                "prior_verdicts": _prior_verdicts_for_track(
                    rows, "hardware_rtl_source_before_lint_sim"
                ),
                "retire_if_same_verdict": True,
            }
        )
    return forbidden


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
    """REQ-REPORT-036: render the operator-readable `.111` activation table."""

    lines = [
        "# Milestone .111 Carry-Forward Manifest",
        "",
        f"Prior milestone: `{PRIOR_MILESTONE}`",
        f"Target milestone: `{TARGET_MILESTONE}`",
        f"Run date: `{RUN_DATE}`",
        "",
        "| track | prior evidence | .111 task | gate rule | retire-if-same-verdict rule |",
        "|---|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    _md_cell(row["track"]),
                    _md_cell(_render_prior_evidence(row["prior_evidence"])),
                    _md_cell(", ".join(row["mapped_111_tasks"]) or "EXPLICITLY UNMAPPED"),
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
                f"- {item['forbidden_scope_id']}: {item['forbidden_scope']}. "
                f"Required unlock evidence: {item['required_unlock_evidence']}."
            )
    else:
        lines.append("- None recorded.")
    lines.extend(
        [
            "",
            "## No-Change Confirmation",
            "",
            "- scripts/research_conductor.py: no activation-manifest changes needed",
            "- research-roadmap.yaml: no activation-manifest changes needed",
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
    """REQ-REPORT-036: map `.110` unresolved tracks into `.111` work."""

    rows = [_row_for_task(task, sources) for task in _tasks_from_retro(retro, sources)]
    unmapped_tracks = [row["track_id"] for row in rows if not row["mapped"]]
    forbidden = _forbidden_exact_reruns(rows)
    missing_required_forbidden = sorted(
        {
            "prototype_repair_scaleup",
            "fr11_zero_growth",
            "prm_v1_no_improvement",
            "missing_source_rtl_lint_sim",
        }
        - {str(item["forbidden_scope_id"]) for item in forbidden}
    )
    complete = bool(rows) and not unmapped_tracks and not missing_required_forbidden
    manifest = render_manifest(rows, forbidden)
    missing_roadmap_task_ids = [
        task_id
        for row in rows
        for task_id in row["mapped_111_tasks"]
        if task_id.startswith("exp") and task_id not in roadmap_text
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
        "missing_required_forbidden_exact_reruns": missing_required_forbidden,
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
            "scripts/research_conductor.py": "no activation-manifest changes needed",
            "research-roadmap.yaml": "no activation-manifest changes needed",
        },
        "unmapped_tracks": unmapped_tracks,
        "honest_verdict": (
            f"milestone_111_carryforward_manifest_complete_{len(rows)}_tracks_"
            "prototype_fr11_prm_rtl_exact_reruns_forbidden"
            if complete
            else "carryforward_manifest_incomplete_unmapped_or_missing_forbidden_reruns"
        ),
    }
    return artifact, manifest


def run(
    root: Path | str = REPO_ROOT,
    out_path: Path | str = DEFAULT_OUT_PATH,
    manifest_path: Path | str = DEFAULT_MANIFEST_PATH,
) -> dict[str, Any]:
    """REQ-REPORT-036: write bootstrap, markdown manifest, and terminal JSON."""

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
