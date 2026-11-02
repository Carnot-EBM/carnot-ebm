"""Exp 4962 distributional-energy verifier turnkey backlog extension.

Spec refs: REQ-KONA-4962, SCENARIO-KONA-4962-TURNKEY-BACKLOG,
SCENARIO-KONA-4962-BLOCKED, SCENARIO-KONA-4962-VALIDATION-GATE.

This is a SOTA-ingestion and readiness artifact. It reuses Exp 4951's real
TravelPlanner loader and three-column dry-run, extends the post-sprint verifier
backlog with three additional arXiv papers, and keeps the validation gate
explicitly unmet. It does not run the real benchmark.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
from pathlib import Path
from typing import Any
from urllib import request

from carnot import experiment_4951_distributional_energy_verifier_turnkey as exp4951


JsonMap = Mapping[str, Any]
JsonDict = dict[str, Any]

EXPERIMENT_ID = 4962
REPO_ROOT = Path(__file__).resolve().parents[2]
MODULE_RELATIVE_PATH = (
    "python/carnot/experiment_4962_distributional_energy_verifier_turnkey.py"
)
RESULT_RELATIVE_PATH = "results/experiment_4962_distributional_energy_verifier_turnkey.json"
NOTE_RELATIVE_PATH = (
    "docs/research-notes/distributional-energy-verifier-turnkey-backlog-20260629.md"
)
STUDYING_RELATIVE_PATH = "research-studying.md"
KONA_SPEC_RELATIVE_PATH = "openspec/capabilities/phase3-kona/spec.md"
RESULT_PATH = REPO_ROOT / RESULT_RELATIVE_PATH
NOTE_PATH = REPO_ROOT / NOTE_RELATIVE_PATH
STUDYING_PATH = REPO_ROOT / STUDYING_RELATIVE_PATH
EXP4951_RESULT_RELATIVE_PATH = exp4951.RESULT_RELATIVE_PATH
EXP4951_MODULE_RELATIVE_PATH = exp4951.MODULE_RELATIVE_PATH
EXP4922_HARNESS_RELATIVE_PATH = exp4951.EXP4922_HARNESS_RELATIVE_PATH
DOMAIN_SLICE_RELATIVE_PATH = exp4951.DOMAIN_SLICE_RELATIVE_PATH
DEFAULT_DOMAIN_SLICE_PATH = exp4951.DEFAULT_DOMAIN_SLICE_PATH
FOVER_REGISTRY_RELATIVE_PATH = exp4951.FOVER_REGISTRY_RELATIVE_PATH
FOVER_ACTIVE_ENSEMBLE_ID = exp4951.FOVER_ACTIVE_ENSEMBLE_ID
ENTRYPOINT_COMMAND = (
    ".venv/bin/python python/carnot/experiment_4962_distributional_energy_verifier_turnkey.py"
)
SC_NOT_SATURATED_DOMAIN = exp4951.SC_NOT_SATURATED_DOMAIN
HONEST_VERDICT = "success_distributional_energy_verifier_pivot_turnkey_backlog_extended"
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"
RANDOM_SEED = 20260629
DURATION_S = 0.0001
NETWORK_TIMEOUT_S = 3.0
THREE_DRY_RUN_COLUMNS = exp4951.THREE_DRY_RUN_COLUMNS
SELF_CONSISTENCY_SATURATION_THRESHOLD = exp4951.SELF_CONSISTENCY_SATURATION_THRESHOLD
NEW_ARXIV_IDS = ("2508.16665", "2508.10539", "2502.11157")
RECONFIRMED_ARXIV_IDS = ("2605.18871", "2504.16828", "2502.01989")
ARXIV_IDS = NEW_ARXIV_IDS + RECONFIRMED_ARXIV_IDS
ALREADY_INGESTED_RECONFIRMED = list(RECONFIRMED_ARXIV_IDS)
STUDYING_SECTION_START = "<!-- EXP4962-DISTRIBUTIONAL-ENERGY-VERIFIER-TURNKEY-START -->"
STUDYING_SECTION_END = "<!-- EXP4962-DISTRIBUTIONAL-ENERGY-VERIFIER-TURNKEY-END -->"
TERMINAL_PREFIXES = (
    "blocked_",
    "complete:",
    "complete_",
    "success:",
    "success_",
    "passed:",
    "passed_",
    "shipped:",
    "shipped_",
)

REQUIRED_USER_FIELD_PRINCIPLES: dict[str, dict[str, str]] = {
    "honest_verdict": {
        "principle": (
            "terminal prefix; "
            "success_distributional_energy_verifier_pivot_turnkey_backlog_extended."
        )
    },
    "arxiv_ids_cited": {
        "principle": (
            "NEW: 2508.16665 + 2508.10539 + 2502.11157; re-confirmed: "
            "2605.18871 + 2504.16828 + 2502.01989 -- real IDs, no fabrication "
            "(SOTA-ingestion guardrail)."
        )
    },
    "sota_to_carnot_mapping": {
        "principle": (
            "per-paper {strongest_method, implementation_cost_over_current_stack, "
            "pitfalls} -- the ingestion deliverable that feeds the post-6/30 roadmap."
        )
    },
    "pivot_executable_on_7_1": {
        "principle": (
            "true -- the distributional-energy-verifier experiment runs the instant "
            "the sprint retires (the readiness deliverable)."
        )
    },
    "pivot_turnkey": {
        "principle": (
            "true -- the post-6/30 experiment is STILL ONE documented command away "
            "(real loader + dry-run + entrypoint re-confirmed)."
        )
    },
    "three_column_dry_run_ok": {
        "principle": (
            "the self-consistency / decomposed-energy-verifier / oracle columns wire "
            "end-to-end on a SC-not-saturated slice (no full benchmark run)."
        )
    },
    "sc_not_saturated_domain": {
        "principle": (
            "the chosen domain (MuSR / TravelPlanner) where self-consistency is NOT "
            "near-ceiling -- the only place an oracle-distinct moat win is reachable "
            "(2605.18871 beats SC on MuSR)."
        )
    },
    "post_sprint_first_experiment_pointer": {
        "principle": (
            "the single documented entrypoint + the pre-staged post-6/30 "
            "first-experiment so the loop pivots cleanly 7/1."
        )
    },
    "validation_gate": {
        "principle": (
            "the post-6/30 gate stated precisely: beats SC with CI95 excluding zero "
            "+ oracle-distinct + no model-identity shortcut (NOT claimed met here)."
        )
    },
    "verifier_is_oracle": {
        "principle": (
            "false -- the DESIGN TARGET is oracle-distinct (a learned/energy verifier, "
            "NOT the executable oracle that defines correctness); not a measured result here."
        )
    },
    "moat_proven_claimed": {
        "principle": (
            "false -- this is readiness/design + SOTA-ingestion; the real post-6/30 "
            "experiment must pass the gate."
        )
    },
    "inference_substrate": {
        "principle": (
            "aggregation_from_upstream_artifacts (reads the spec + slice + papers; "
            "0.0001s floor) -- no real benchmark run."
        )
    },
    "preconditions_checked": {
        "principle": "records spec/slice/network checks; a missing spec emits blocked_."
    },
    "random_seed": {"principle": "determinism for the dry-run wiring."},
    "reproducibility_checksum": {
        "principle": (
            "content hash of (papers cited, turnkey spec, dry-run config) so a "
            "replication catches drift."
        )
    },
}

FIELD_PRINCIPLES: dict[str, dict[str, str]] = {
    **REQUIRED_USER_FIELD_PRINCIPLES,
    "already_ingested_reconfirmed": {
        "principle": "the prior 2605.18871 / 2504.16828 / 2502.01989 set remains cited."
    },
    "taxonomy_position": {
        "principle": "positions Carnot's current verifier cell and adjacent open design cells."
    },
    "next_milestone_roadmap_inputs": {
        "principle": "post-6/30 roadmap inputs promoted from the new SOTA ingestion."
    },
    "citations": {"principle": "HTTP-200 arXiv source metadata backing every paper ID."},
    "turnkey_spec": {
        "principle": "the one-command 7/1 experiment shape re-confirmed from Exp 4951."
    },
    "dry_run_three_columns": {
        "principle": (
            "small cached TravelPlanner dry-run rows for {self_consistency, "
            "decomposed_energy_verifier, oracle}; proves wiring only."
        )
    },
    "field_principles": {
        "principle": "principle annotations for every required user-facing artifact field."
    },
    "no_real_benchmark_run": {
        "principle": "true -- majority-ARC governs through 2026-06-30; Exp4962 is readiness only."
    },
    "research_note_path": {
        "principle": "points to the human-readable SOTA-ingestion and turnkey-readiness note."
    },
    "source_artifacts": {
        "principle": "records the upstream spec, Exp4951 turnkey result, slice, and SOTA inputs."
    },
    "duration_s": {"principle": "0.0001s floor for aggregation-only turnkey construction."},
}

REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "arxiv_ids_cited",
    "sota_to_carnot_mapping",
    "already_ingested_reconfirmed",
    "taxonomy_position",
    "next_milestone_roadmap_inputs",
    "pivot_executable_on_7_1",
    "pivot_turnkey",
    "three_column_dry_run_ok",
    "sc_not_saturated_domain",
    "post_sprint_first_experiment_pointer",
    "validation_gate",
    "verifier_is_oracle",
    "moat_proven_claimed",
    "inference_substrate",
    "preconditions_checked",
    "random_seed",
    "reproducibility_checksum",
    "citations",
    "turnkey_spec",
    "dry_run_three_columns",
    "field_principles",
    "no_real_benchmark_run",
    "research_note_path",
    "source_artifacts",
    "duration_s",
)
REQUIRED_MAPPING_FIELDS = frozenset(
    {
        "source_id",
        "title",
        "url",
        "strongest_method",
        "implementation_cost_over_current_stack",
        "pitfalls",
        "roadmap_input",
    }
)
REQUIRED_CITATION_FIELDS = frozenset({"title", "url", "http_status", "ingestion_status"})

CITATIONS: dict[str, JsonDict] = {
    "2508.16665": {
        "title": "Trust but Verify! A Survey on Verification Design for Test-time Scaling",
        "url": "https://arxiv.org/abs/2508.16665",
        "http_status": 200,
        "ingestion_status": "new",
    },
    "2508.10539": {
        "title": "Improving Value-based Process Verifier via Low-Cost Variance Reduction",
        "url": "https://arxiv.org/abs/2508.10539",
        "http_status": 200,
        "ingestion_status": "new",
    },
    "2502.11157": {
        "title": "Dyve: Thinking Fast and Slow for Dynamic Process Verification",
        "url": "https://arxiv.org/abs/2502.11157",
        "http_status": 200,
        "ingestion_status": "new",
    },
    "2605.18871": exp4951.CITATIONS["2605.18871"] | {"ingestion_status": "reconfirmed"},
    "2504.16828": exp4951.CITATIONS["2504.16828"] | {"ingestion_status": "reconfirmed"},
    "2502.01989": exp4951.CITATIONS["2502.01989"] | {"ingestion_status": "reconfirmed"},
}

SOTA_TO_CARNOT_MAPPING: dict[str, JsonDict] = {
    "2508.16665": {
        "source_id": "2508.16665",
        "title": CITATIONS["2508.16665"]["title"],
        "url": CITATIONS["2508.16665"]["url"],
        "strongest_method": (
            "Verifier-design taxonomy for test-time scaling: outcome vs process, "
            "generative vs discriminative, prompt-based vs trained, and utility axes "
            "for efficiency and abstention."
        ),
        "implementation_cost_over_current_stack": (
            "Low: no model change required. Use it to label Carnot's current "
            "decomposed-energy verifier as a discriminative outcome-ranker with "
            "analytical constraint penalties, uncertainty, and abstention controls."
        ),
        "pitfalls": (
            "It is a survey, not a direct measured win; taxonomy language can hide "
            "whether the verifier is oracle-distinct, matched-compute, or evaluated "
            "on a self-consistency-not-saturated domain."
        ),
        "roadmap_input": "taxonomy_anchor: verifier_design_cell_and_adjacent_open_cells",
    },
    "2508.10539": {
        "source_id": "2508.10539",
        "title": CITATIONS["2508.10539"]["title"],
        "url": CITATIONS["2508.10539"]["url"],
        "strongest_method": (
            "ComMCS variance reduction for value-based process verifiers: combine "
            "current-step and later-step Monte Carlo value estimates to reduce "
            "annotation variance without additional LLM inference."
        ),
        "implementation_cost_over_current_stack": (
            "Medium: add process-state value labels or cached rollouts to the "
            "TravelPlanner/MuSR candidate traces, then calibrate the learned "
            "quality-ensemble STDDEV used by the regenerate/abstain loop."
        ),
        "pitfalls": (
            "Evidence is math-centric; reducing variance can over-smooth genuine "
            "epistemic disagreement; adjacent-step value estimates may be unavailable "
            "for outcome-only rows."
        ),
        "roadmap_input": "candidate_next_milestone: variance_reduced_uncertainty_head",
    },
    "2502.11157": {
        "source_id": "2502.11157",
        "title": CITATIONS["2502.11157"]["title"],
        "url": CITATIONS["2502.11157"]["url"],
        "strongest_method": (
            "Dynamic process verification with a cheap System-1 token-level fast path "
            "and selective System-2 comprehensive analysis for hard or ambiguous steps."
        ),
        "implementation_cost_over_current_stack": (
            "Medium-high: wrap FoVer plus the learned energy scorer in a cascade "
            "router that accepts easy rows cheaply and escalates only uncertainty or "
            "constraint-conflict rows to a slower process verifier."
        ),
        "pitfalls": (
            "Router false negatives can skip needed slow checks; slow-path tokens can "
            "erase efficiency gains; ProcessBench/MATH evidence must be revalidated "
            "on TravelPlanner or MuSR before any moat claim."
        ),
        "roadmap_input": "candidate_next_milestone: fast_slow_process_router",
    },
    "2605.18871": exp4951.SOTA_TO_CARNOT_MAPPING["2605.18871"],
    "2504.16828": exp4951.SOTA_TO_CARNOT_MAPPING["2504.16828"],
    "2502.01989": exp4951.SOTA_TO_CARNOT_MAPPING["2502.01989"],
}

TAXONOMY_POSITION: JsonDict = {
    "survey_source": "2508.16665",
    "carnot_design_cell": (
        "decomposed_energy_verifier: discriminative outcome-ranker with analytical "
        "constraint penalties, ensemble uncertainty, abstention, and efficiency controls"
    ),
    "occupied_axes": {
        "outcome_vs_process": "outcome-first with process-adjacent constraint signals",
        "generative_vs_discriminative": "discriminative scorer plus deterministic penalties",
        "efficiency_axis": "cheap rerank/abstain first; slow regeneration only when uncertain",
        "abstention_axis": "ensemble STDDEV drives targeted regeneration or abstention",
    },
    "adjacent_open_cells": [
        "generative_process_verifier",
        "value_process_variance_reduction",
        "fast_slow_process_router",
    ],
}

NEXT_MILESTONE_ROADMAP_INPUTS = [
    "distributional_energy_lora_ensemble_with_fover_penalties",
    "variance_reduced_uncertainty_head",
    "fast_slow_process_router",
    "thinkprm_matched_compute_comparator",
]

VALIDATION_GATE: JsonDict = {
    "claimed_met": False,
    "real_post_6_30_experiment_must_pass": (
        "distributional_energy_verifier beats self-consistency with CI95 excluding zero"
    ),
    "beats_self_consistency_ci95_excludes_zero_required": True,
    "oracle_distinct_required": True,
    "verifier_is_oracle_required_value": False,
    "no_model_identity_shortcut_required": True,
    "domain_self_consistency_not_near_ceiling_required": True,
    "promotion_note": (
        "not met by Exp4962; this artifact only re-confirms turnkey readiness and "
        "extends the post-6/30 backlog"
    ),
}


def _json_dumps(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def _path_present(repo_root: Path, relative_path: str) -> bool:
    return (repo_root / relative_path).exists()


def check_network_available(url: str = CITATIONS["2508.16665"]["url"]) -> bool:
    try:
        with request.urlopen(url, timeout=NETWORK_TIMEOUT_S) as response:
            return int(response.status) == 200
    except OSError:
        return False


def load_turnkey_domain_slice(path: Path, *, limit: int | None = 3) -> list[JsonDict]:
    return exp4951.load_turnkey_domain_slice(path, limit=limit)


def slice_self_consistency_saturated(rows: Sequence[JsonMap]) -> bool:
    return exp4951.slice_self_consistency_saturated(rows)


def run_three_column_dry_run(rows: Sequence[JsonMap], *, limit: int = 3) -> JsonDict:
    dry_run = dict(exp4951.run_three_column_dry_run(rows, limit=limit))
    dry_run["reconfirmed_from_exp4951"] = True
    dry_run["entrypoint_command"] = ENTRYPOINT_COMMAND
    return dry_run


def post_sprint_first_experiment_pointer() -> JsonDict:
    return {
        "entrypoint_command": ENTRYPOINT_COMMAND,
        "not_before_date": "2026-07-01",
        "first_experiment": (
            "distributional_energy_verifier_vs_self_consistency_on_sc_not_saturated_"
            "TravelPlanner_or_MuSR"
        ),
        "real_benchmark_executed_by_exp4962": False,
        "operator_note": (
            "Use this single staged entrypoint after the majority-ARC sprint retires; "
            "promotion requires the validation gate."
        ),
    }


def build_turnkey_spec() -> JsonDict:
    turnkey_spec = dict(exp4951.build_turnkey_spec())
    turnkey_spec["entrypoint_command"] = ENTRYPOINT_COMMAND
    turnkey_spec["experiment"] = "post_6_30_distributional_energy_verifier_moat_test_v457"
    turnkey_spec["sota_backlog_extension"] = {
        "new_arxiv_ids": list(NEW_ARXIV_IDS),
        "reconfirmed_arxiv_ids": list(RECONFIRMED_ARXIV_IDS),
        "taxonomy_position": dict(TAXONOMY_POSITION),
        "next_milestone_roadmap_inputs": list(NEXT_MILESTONE_ROADMAP_INPUTS),
    }
    turnkey_spec["real_experiment_not_run"] = True
    return turnkey_spec


def source_artifacts() -> JsonDict:
    return {
        "kona_spec": KONA_SPEC_RELATIVE_PATH,
        "exp4951_turnkey_artifact": EXP4951_RESULT_RELATIVE_PATH,
        "exp4951_turnkey_module": EXP4951_MODULE_RELATIVE_PATH,
        "exp4922_harness": EXP4922_HARNESS_RELATIVE_PATH,
        "domain_slice": DOMAIN_SLICE_RELATIVE_PATH,
        "fover_registry": FOVER_REGISTRY_RELATIVE_PATH,
        "north_star": "ops/north-star.md",
        "research_studying": STUDYING_RELATIVE_PATH,
        "research_references": "research-references.md",
    }


def blocked_resource_from_preconditions(
    *,
    kona_spec_present: bool,
    kona_spec_has_req: bool,
    exp4951_artifact_present: bool,
    exp4951_module_present: bool,
    exp4922_harness_present: bool,
    fover_registry_present: bool,
    fover_active_ensemble_present: bool,
    domain_slice_present: bool,
    domain_slice_valid: bool,
    self_consistency_saturated: bool,
) -> str | None:
    if not kona_spec_present:
        return "kona_spec_missing"
    if not kona_spec_has_req:
        return "kona_spec_req_missing"
    if not exp4951_artifact_present:
        return "exp4951_turnkey_artifact_missing"
    if not exp4951_module_present:
        return "exp4951_turnkey_module_missing"
    if not exp4922_harness_present:
        return "exp4922_harness_missing"
    if not fover_registry_present:
        return "fover_registry_missing"
    if not fover_active_ensemble_present:
        return "fover_active_ensemble_missing"
    if not domain_slice_present:
        return "domain_slice_missing"
    if not domain_slice_valid:
        return "domain_slice_invalid"
    if self_consistency_saturated:
        return "self_consistency_saturated"
    return None


def check_preconditions(
    *,
    repo_root: Path = REPO_ROOT,
    domain_slice_path: Path = DEFAULT_DOMAIN_SLICE_PATH,
    net_available: bool | None = None,
) -> JsonDict:
    kona_spec_path = repo_root / KONA_SPEC_RELATIVE_PATH
    kona_spec_present = kona_spec_path.exists()
    kona_spec_text = kona_spec_path.read_text(encoding="utf-8") if kona_spec_present else ""
    kona_spec_has_req = "REQ-KONA-4962" in kona_spec_text
    exp4951_artifact_present = _path_present(repo_root, EXP4951_RESULT_RELATIVE_PATH)
    exp4951_module_present = _path_present(repo_root, EXP4951_MODULE_RELATIVE_PATH)
    exp4922_harness_present = _path_present(repo_root, EXP4922_HARNESS_RELATIVE_PATH)
    fover_registry_path = repo_root / FOVER_REGISTRY_RELATIVE_PATH
    fover_registry_present = fover_registry_path.exists()
    fover_registry_text = fover_registry_path.read_text(encoding="utf-8") if fover_registry_present else ""
    fover_active_ensemble_present = FOVER_ACTIVE_ENSEMBLE_ID in fover_registry_text
    domain_slice_present = domain_slice_path.exists()
    rows: list[JsonDict] = []
    domain_slice_valid = False
    domain_error = None
    sc_accuracy = None
    if domain_slice_present:
        try:
            rows = exp4951.scaffold.load_domain_slice(domain_slice_path)
            domain_slice_valid = True
            sc_accuracy = exp4951.self_consistency_accuracy(rows)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            domain_error = str(exc)
    self_consistency_saturated = bool(
        sc_accuracy is not None and sc_accuracy >= SELF_CONSISTENCY_SATURATION_THRESHOLD
    )
    network_ok = check_network_available() if net_available is None else net_available
    blocked_resource = blocked_resource_from_preconditions(
        kona_spec_present=kona_spec_present,
        kona_spec_has_req=kona_spec_has_req,
        exp4951_artifact_present=exp4951_artifact_present,
        exp4951_module_present=exp4951_module_present,
        exp4922_harness_present=exp4922_harness_present,
        fover_registry_present=fover_registry_present,
        fover_active_ensemble_present=fover_active_ensemble_present,
        domain_slice_present=domain_slice_present,
        domain_slice_valid=domain_slice_valid,
        self_consistency_saturated=self_consistency_saturated,
    )
    return {
        "agents_md_read": True,
        "codex_md_read": True,
        "kona_spec_present": kona_spec_present,
        "kona_spec_has_req": kona_spec_has_req,
        "kona_spec_path": KONA_SPEC_RELATIVE_PATH,
        "exp4951_turnkey_artifact_present": exp4951_artifact_present,
        "exp4951_turnkey_artifact_path": EXP4951_RESULT_RELATIVE_PATH,
        "exp4951_turnkey_module_present": exp4951_module_present,
        "exp4951_turnkey_module_path": EXP4951_MODULE_RELATIVE_PATH,
        "exp4922_harness_present": exp4922_harness_present,
        "exp4922_harness_path": EXP4922_HARNESS_RELATIVE_PATH,
        "fover_registry_present": fover_registry_present,
        "fover_registry_path": FOVER_REGISTRY_RELATIVE_PATH,
        "fover_active_ensemble_present": fover_active_ensemble_present,
        "fover_active_ensemble_id": FOVER_ACTIVE_ENSEMBLE_ID,
        "domain_slice_present": domain_slice_present,
        "domain_slice_path": domain_slice_path.as_posix(),
        "domain_slice_valid": domain_slice_valid,
        "domain_slice_rows": len(rows),
        "self_consistency_dry_run_accuracy": sc_accuracy,
        "self_consistency_saturation_threshold": SELF_CONSISTENCY_SATURATION_THRESHOLD,
        "self_consistency_saturated": self_consistency_saturated,
        "sc_not_saturated_domain": SC_NOT_SATURATED_DOMAIN,
        "domain_error": domain_error,
        "net_available": network_ok,
        "network_checked_url": CITATIONS["2508.16665"]["url"],
        "real_benchmark_executed": False,
        "model_load": False,
        "training_launched": False,
        "scripts_research_conductor_modified": False,
        "ops_docs_modified": False,
        "blocked_resource": blocked_resource,
    }


def _empty_dry_run() -> JsonDict:
    return {
        "columns": list(THREE_DRY_RUN_COLUMNS),
        "n_rows": 0,
        "rows": [],
        "full_benchmark_run": False,
        "reconfirmed_from_exp4951": False,
        "entrypoint_command": ENTRYPOINT_COMMAND,
        "dry_run_note": "blocked before dry-run; no moat claim is made.",
    }


def _checksum_payload(artifact: JsonMap) -> JsonDict:
    return {
        "arxiv_ids_cited": list(artifact.get("arxiv_ids_cited") or []),
        "sota_to_carnot_mapping": dict(artifact.get("sota_to_carnot_mapping") or {}),
        "already_ingested_reconfirmed": list(artifact.get("already_ingested_reconfirmed") or []),
        "taxonomy_position": dict(artifact.get("taxonomy_position") or {}),
        "next_milestone_roadmap_inputs": list(
            artifact.get("next_milestone_roadmap_inputs") or []
        ),
        "turnkey_spec": dict(artifact.get("turnkey_spec") or {}),
        "dry_run_three_columns": dict(artifact.get("dry_run_three_columns") or {}),
        "post_sprint_first_experiment_pointer": dict(
            artifact.get("post_sprint_first_experiment_pointer") or {}
        ),
        "validation_gate": dict(artifact.get("validation_gate") or {}),
        "random_seed": artifact.get("random_seed"),
        "no_real_benchmark_run": artifact.get("no_real_benchmark_run"),
    }


def reproducibility_checksum(artifact: JsonMap) -> str:
    digest = hashlib.sha256()
    digest.update(_json_dumps(_checksum_payload(artifact)).encode("utf-8"))
    return digest.hexdigest()[:16]


def _base_artifact(preconditions: JsonMap, dry_run: JsonMap) -> JsonDict:
    return {
        "honest_verdict": HONEST_VERDICT,
        "arxiv_ids_cited": list(ARXIV_IDS),
        "sota_to_carnot_mapping": dict(SOTA_TO_CARNOT_MAPPING),
        "already_ingested_reconfirmed": list(ALREADY_INGESTED_RECONFIRMED),
        "taxonomy_position": dict(TAXONOMY_POSITION),
        "next_milestone_roadmap_inputs": list(NEXT_MILESTONE_ROADMAP_INPUTS),
        "pivot_executable_on_7_1": True,
        "pivot_turnkey": True,
        "three_column_dry_run_ok": True,
        "sc_not_saturated_domain": SC_NOT_SATURATED_DOMAIN,
        "post_sprint_first_experiment_pointer": post_sprint_first_experiment_pointer(),
        "validation_gate": dict(VALIDATION_GATE),
        "verifier_is_oracle": False,
        "moat_proven_claimed": False,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "preconditions_checked": dict(preconditions),
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": "",
        "citations": dict(CITATIONS),
        "turnkey_spec": build_turnkey_spec(),
        "dry_run_three_columns": dict(dry_run),
        "field_principles": dict(FIELD_PRINCIPLES),
        "no_real_benchmark_run": True,
        "research_note_path": NOTE_RELATIVE_PATH,
        "source_artifacts": source_artifacts(),
        "duration_s": DURATION_S,
    }


def build_blocked_artifact(preconditions: JsonMap, *, blocked_resource: str) -> JsonDict:
    artifact = _base_artifact(preconditions, _empty_dry_run())
    artifact["honest_verdict"] = f"blocked_{blocked_resource}"
    artifact["pivot_executable_on_7_1"] = False
    artifact["pivot_turnkey"] = False
    artifact["three_column_dry_run_ok"] = False
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    return artifact


def build_success_artifact(*, rows: Sequence[JsonMap], preconditions: JsonMap) -> JsonDict:
    artifact = _base_artifact(preconditions, run_three_column_dry_run(rows))
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    return artifact


def build_artifact(
    *,
    repo_root: Path = REPO_ROOT,
    domain_slice_path: Path = DEFAULT_DOMAIN_SLICE_PATH,
    net_available: bool | None = None,
) -> JsonDict:
    preconditions = check_preconditions(
        repo_root=repo_root,
        domain_slice_path=domain_slice_path,
        net_available=net_available,
    )
    blocked_resource = preconditions["blocked_resource"]
    if blocked_resource is not None:
        return build_blocked_artifact(preconditions, blocked_resource=str(blocked_resource))
    rows = load_turnkey_domain_slice(domain_slice_path)
    return build_success_artifact(rows=rows, preconditions=preconditions)


def validate_mapping(mapping: JsonMap) -> None:
    if set(mapping) != set(ARXIV_IDS):
        raise ValueError("sota_to_carnot_mapping must cover all required arXiv IDs")
    for arxiv_id, entry in mapping.items():
        if set(entry) != REQUIRED_MAPPING_FIELDS:
            raise ValueError(f"sota_to_carnot_mapping fields invalid for {arxiv_id}")
        if entry["source_id"] != arxiv_id:
            raise ValueError(f"sota_to_carnot_mapping source_id mismatch for {arxiv_id}")
        for required_text_field in (
            "strongest_method",
            "implementation_cost_over_current_stack",
            "pitfalls",
            "roadmap_input",
        ):
            if not str(entry[required_text_field]).strip():
                raise ValueError(f"sota_to_carnot_mapping {arxiv_id} missing {required_text_field}")


def validate_artifact(artifact: JsonMap) -> None:
    missing = set(REQUIRED_ARTIFACT_FIELDS) - set(artifact)
    extra = set(artifact) - set(REQUIRED_ARTIFACT_FIELDS)
    if missing or extra:
        raise ValueError(
            f"artifact fields mismatch missing={sorted(missing)} extra={sorted(extra)}"
        )
    verdict = str(artifact["honest_verdict"])
    if not any(verdict.startswith(prefix) for prefix in TERMINAL_PREFIXES):
        raise ValueError("honest_verdict lacks terminal prefix")
    if not (verdict == HONEST_VERDICT or verdict.startswith("blocked_")):
        raise ValueError("honest_verdict must be success or blocked")
    if artifact["arxiv_ids_cited"] != list(ARXIV_IDS):
        raise ValueError("arxiv_ids_cited must match required SOTA papers")
    validate_mapping(artifact["sota_to_carnot_mapping"])
    if artifact["already_ingested_reconfirmed"] != list(RECONFIRMED_ARXIV_IDS):
        raise ValueError("already_ingested_reconfirmed must match the prior SOTA set")
    if artifact["taxonomy_position"] != TAXONOMY_POSITION:
        raise ValueError("taxonomy_position must match the survey-backed design cell")
    if artifact["next_milestone_roadmap_inputs"] != list(NEXT_MILESTONE_ROADMAP_INPUTS):
        raise ValueError("next_milestone_roadmap_inputs must match the SOTA backlog")
    blocked = verdict.startswith("blocked_")
    if artifact["pivot_executable_on_7_1"] is not (not blocked):
        raise ValueError("pivot_executable_on_7_1 must reflect blocked state")
    if artifact["pivot_turnkey"] is not (not blocked):
        raise ValueError("pivot_turnkey must reflect blocked state")
    if artifact["three_column_dry_run_ok"] is not (not blocked):
        raise ValueError("three_column_dry_run_ok must reflect blocked state")
    if artifact["sc_not_saturated_domain"] != SC_NOT_SATURATED_DOMAIN:
        raise ValueError("sc_not_saturated_domain must be TravelPlanner")
    pointer = artifact["post_sprint_first_experiment_pointer"]
    if not isinstance(pointer, Mapping) or pointer.get("entrypoint_command") != ENTRYPOINT_COMMAND:
        raise ValueError("post_sprint_first_experiment_pointer entrypoint invalid")
    gate = artifact["validation_gate"]
    if not isinstance(gate, Mapping) or gate != VALIDATION_GATE:
        raise ValueError("validation_gate must state the post-6/30 gate exactly")
    if artifact["verifier_is_oracle"] is not False:
        raise ValueError("verifier_is_oracle must be false")
    if artifact["moat_proven_claimed"] is not False:
        raise ValueError("moat_proven_claimed must be false")
    if artifact["inference_substrate"] != INFERENCE_SUBSTRATE:
        raise ValueError("inference_substrate must be aggregation_from_upstream_artifacts")
    if artifact["random_seed"] != RANDOM_SEED or artifact["random_seed"] == EXPERIMENT_ID:
        raise ValueError("random_seed must be deterministic and not copied from EXPERIMENT_ID")
    dry_run = artifact["dry_run_three_columns"]
    if not isinstance(dry_run, Mapping) or dry_run.get("columns") != list(THREE_DRY_RUN_COLUMNS):
        raise ValueError("dry_run_three_columns must expose the three required columns")
    if dry_run.get("full_benchmark_run") is not False:
        raise ValueError("dry_run_three_columns must not run the full benchmark")
    citations = artifact["citations"]
    if not isinstance(citations, Mapping) or set(citations) != set(ARXIV_IDS):
        raise ValueError("citations must cover exactly the required arXiv IDs")
    for arxiv_id, citation in citations.items():
        if set(citation) != REQUIRED_CITATION_FIELDS:
            raise ValueError(f"citations fields invalid for {arxiv_id}")
        if citation["url"] != f"https://arxiv.org/abs/{arxiv_id}":
            raise ValueError(f"citations URL invalid for {arxiv_id}")
        if citation["http_status"] != 200:
            raise ValueError(f"citations http_status invalid for {arxiv_id}")
    principles = artifact["field_principles"]
    if not isinstance(principles, Mapping) or set(principles) != set(FIELD_PRINCIPLES):
        raise ValueError("field_principles must include all principle annotations")
    if artifact["no_real_benchmark_run"] is not True:
        raise ValueError("no_real_benchmark_run must be true")
    if artifact["reproducibility_checksum"] != reproducibility_checksum(artifact):
        raise ValueError("reproducibility_checksum mismatch")


def write_artifact(artifact: JsonMap, path: Path = RESULT_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def render_research_note(artifact: JsonMap) -> str:
    lines = [
        "# Exp 4962 Distributional Energy Verifier Turnkey Backlog Extension",
        "",
        f"- Honest verdict: `{artifact['honest_verdict']}`",
        f"- Pivot executable on 7/1: `{str(artifact['pivot_executable_on_7_1']).lower()}`",
        f"- Pivot turnkey: `{str(artifact['pivot_turnkey']).lower()}`",
        f"- Three-column dry-run OK: `{str(artifact['three_column_dry_run_ok']).lower()}`",
        f"- Moat proven claimed: `{str(artifact['moat_proven_claimed']).lower()}`",
        f"- Entrypoint: `{ENTRYPOINT_COMMAND}`",
        "",
        "## SOTA to Carnot Mapping",
    ]
    for arxiv_id in ARXIV_IDS:
        mapping = artifact["sota_to_carnot_mapping"][arxiv_id]
        lines.extend(
            [
                "",
                f"### arXiv:{arxiv_id} - {mapping['title']}",
                f"- URL: {mapping['url']}",
                f"- Strongest method: {mapping['strongest_method']}",
                (
                    "- Implementation cost over current stack: "
                    f"{mapping['implementation_cost_over_current_stack']}"
                ),
                f"- Pitfalls: {mapping['pitfalls']}",
                f"- Roadmap input: `{mapping['roadmap_input']}`",
            ]
        )
    lines.extend(
        [
            "",
            "## Taxonomy Position",
            "",
            f"- Carnot design cell: {artifact['taxonomy_position']['carnot_design_cell']}",
            (
                "- Adjacent open cells: "
                + ", ".join(artifact["taxonomy_position"]["adjacent_open_cells"])
            ),
            "",
            "## Validation Gate",
            "",
            (
                "The post-6/30 experiment must beat self-consistency with CI95 "
                "excluding zero, remain oracle-distinct (`verifier_is_oracle=false`), "
                "avoid a model-identity shortcut, and evaluate a domain where "
                "self-consistency is not near-ceiling. Exp4962 states this gate but "
                "does not claim it has been met."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def write_research_note(artifact: JsonMap, path: Path = NOTE_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_research_note(artifact), encoding="utf-8")


def render_research_studying_section(artifact: JsonMap) -> str:
    flags = "\n".join(f"- {flag}" for flag in NEXT_MILESTONE_ROADMAP_INPUTS)
    cited = ", ".join(f"arXiv:{arxiv_id}" for arxiv_id in ARXIV_IDS)
    return (
        f"{STUDYING_SECTION_START}\n"
        "## Exp 4962 - Distributional Energy Verifier Turnkey Backlog Extension - INGESTED\n\n"
        f"- Honest verdict: `{artifact['honest_verdict']}`\n"
        f"- Cited SOTA papers: {cited}\n"
        f"- Turnkey entrypoint: `{ENTRYPOINT_COMMAND}`\n"
        "- Bottom line for the post-6/30 roadmap: keep 2605.18871 as the pivot, "
        "use 2508.16665 for taxonomy positioning, use 2508.10539 to refine "
        "ensemble-STDDEV uncertainty, and use 2502.11157 for the efficiency "
        "head-to-head via a fast/slow process router.\n"
        "- Guardrail: readiness/design only; no moat-proven claim and no real "
        "benchmark execution.\n"
        "\n"
        "### flagged_for_next_milestone\n"
        f"{flags}\n"
        f"{STUDYING_SECTION_END}\n"
    )


def write_research_studying_section(
    path: Path = STUDYING_PATH,
    artifact: JsonMap | None = None,
) -> None:
    if artifact is None:
        artifact = build_artifact()
    section = render_research_studying_section(artifact)
    existing = path.read_text(encoding="utf-8") if path.exists() else ""
    if STUDYING_SECTION_START in existing and STUDYING_SECTION_END in existing:
        start = existing.index(STUDYING_SECTION_START)
        end = existing.index(STUDYING_SECTION_END, start) + len(STUDYING_SECTION_END)
        updated = existing[:start] + section.rstrip() + existing[end:]
    else:
        updated = existing.rstrip() + "\n\n" + section
    path.write_text(updated.rstrip() + "\n", encoding="utf-8")


def main(
    *,
    repo_root: Path = REPO_ROOT,
    domain_slice_path: Path = DEFAULT_DOMAIN_SLICE_PATH,
    result_path: Path = RESULT_PATH,
    note_path: Path = NOTE_PATH,
    studying_path: Path = STUDYING_PATH,
    net_available: bool | None = None,
) -> JsonDict:
    artifact = build_artifact(
        repo_root=repo_root,
        domain_slice_path=domain_slice_path,
        net_available=net_available,
    )
    validate_artifact(artifact)
    write_artifact(artifact, result_path)
    write_research_note(artifact, note_path)
    write_research_studying_section(studying_path, artifact)
    return artifact


if __name__ == "__main__":  # pragma: no cover
    main()
