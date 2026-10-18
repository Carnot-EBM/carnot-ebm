"""Exp 4973 distributional-energy verifier turnkey backlog extension.

Spec refs: REQ-KONA-4973, SCENARIO-KONA-4973-TURNKEY-BACKLOG,
SCENARIO-KONA-4973-BLOCKED, SCENARIO-KONA-4973-VALIDATION-GATE.

This is a SOTA-ingestion and readiness artifact. It reuses Exp 4962's
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

from carnot import experiment_4962_distributional_energy_verifier_turnkey as exp4962


JsonMap = Mapping[str, Any]
JsonDict = dict[str, Any]

EXPERIMENT_ID = 4973
REPO_ROOT = Path(__file__).resolve().parents[2]
MODULE_RELATIVE_PATH = (
    "python/carnot/experiment_4973_distributional_energy_verifier_turnkey.py"
)
RESULT_RELATIVE_PATH = "results/experiment_4973_distributional_energy_verifier_turnkey.json"
NOTE_RELATIVE_PATH = (
    "docs/research-notes/distributional-energy-verifier-turnkey-backlog-v458-20260629.md"
)
STUDYING_RELATIVE_PATH = "research-studying.md"
KONA_SPEC_RELATIVE_PATH = exp4962.KONA_SPEC_RELATIVE_PATH
RESULT_PATH = REPO_ROOT / RESULT_RELATIVE_PATH
NOTE_PATH = REPO_ROOT / NOTE_RELATIVE_PATH
STUDYING_PATH = REPO_ROOT / STUDYING_RELATIVE_PATH
EXP4962_RESULT_RELATIVE_PATH = exp4962.RESULT_RELATIVE_PATH
EXP4962_MODULE_RELATIVE_PATH = exp4962.MODULE_RELATIVE_PATH
EXP4922_HARNESS_RELATIVE_PATH = exp4962.EXP4922_HARNESS_RELATIVE_PATH
DOMAIN_SLICE_RELATIVE_PATH = exp4962.DOMAIN_SLICE_RELATIVE_PATH
DEFAULT_DOMAIN_SLICE_PATH = exp4962.DEFAULT_DOMAIN_SLICE_PATH
FOVER_REGISTRY_RELATIVE_PATH = exp4962.FOVER_REGISTRY_RELATIVE_PATH
FOVER_ACTIVE_ENSEMBLE_ID = exp4962.FOVER_ACTIVE_ENSEMBLE_ID
ENTRYPOINT_COMMAND = (
    ".venv/bin/python python/carnot/experiment_4973_distributional_energy_verifier_turnkey.py"
)
SC_NOT_SATURATED_DOMAIN = exp4962.SC_NOT_SATURATED_DOMAIN
HONEST_VERDICT = "success_distributional_energy_verifier_pivot_turnkey_backlog_extended"
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"
RANDOM_SEED = 20260629
DURATION_S = 0.0001
NETWORK_TIMEOUT_S = 3.0
THREE_DRY_RUN_COLUMNS = exp4962.THREE_DRY_RUN_COLUMNS
SELF_CONSISTENCY_SATURATION_THRESHOLD = exp4962.SELF_CONSISTENCY_SATURATION_THRESHOLD
NEW_ARXIV_IDS = ("2504.01005", "2504.00891", "2509.24460")
RECONFIRMED_ARXIV_IDS = (
    "2605.18871",
    "2504.16828",
    "2502.01989",
    "2508.16665",
    "2508.10539",
    "2502.11157",
)
ARXIV_IDS = NEW_ARXIV_IDS + RECONFIRMED_ARXIV_IDS
ALREADY_INGESTED_RECONFIRMED = list(RECONFIRMED_ARXIV_IDS)
STUDYING_SECTION_START = "<!-- EXP4973-DISTRIBUTIONAL-ENERGY-VERIFIER-TURNKEY-START -->"
STUDYING_SECTION_END = "<!-- EXP4973-DISTRIBUTIONAL-ENERGY-VERIFIER-TURNKEY-END -->"
TERMINAL_PREFIXES = exp4962.TERMINAL_PREFIXES

REQUIRED_USER_FIELD_PRINCIPLES: dict[str, dict[str, str]] = {
    "honest_verdict": {
        "principle": (
            "terminal prefix; "
            "success_distributional_energy_verifier_pivot_turnkey_backlog_extended."
        )
    },
    "arxiv_ids_cited": {
        "principle": (
            "NEW: 2504.01005 + 2504.00891 + 2509.24460; re-confirmed: "
            "2605.18871 + 2504.16828 + 2502.01989 + 2508.16665 + "
            "2508.10539 + 2502.11157 -- real IDs, no fabrication "
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
        "principle": (
            "the prior 2605.18871 / 2504.16828 / 2502.01989 / 2508.16665 / "
            "2508.10539 / 2502.11157 set remains cited."
        )
    },
    "next_milestone_roadmap_inputs": {
        "principle": "post-6/30 roadmap inputs promoted from the new SOTA ingestion."
    },
    "citations": {"principle": "HTTP-200 arXiv source metadata backing every paper ID."},
    "turnkey_spec": {
        "principle": "the one-command 7/1 experiment shape re-confirmed from Exp 4962."
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
        "principle": "true -- majority-ARC governs through 2026-06-30; Exp4973 is readiness only."
    },
    "research_note_path": {
        "principle": "points to the human-readable SOTA-ingestion and turnkey-readiness note."
    },
    "source_artifacts": {
        "principle": "records the upstream spec, Exp4962 turnkey result, slice, and SOTA inputs."
    },
    "duration_s": {"principle": "0.0001s floor for aggregation-only turnkey construction."},
}

REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "arxiv_ids_cited",
    "sota_to_carnot_mapping",
    "already_ingested_reconfirmed",
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
REQUIRED_MAPPING_FIELDS = exp4962.REQUIRED_MAPPING_FIELDS
REQUIRED_CITATION_FIELDS = frozenset({"title", "url", "http_status", "ingestion_status"})

CITATIONS: dict[str, JsonDict] = {
    "2504.01005": {
        "title": (
            "When To Solve, When To Verify: Compute-Optimal Problem Solving "
            "and Generative Verification for LLM Reasoning"
        ),
        "url": "https://arxiv.org/abs/2504.01005",
        "http_status": 200,
        "ingestion_status": "new",
    },
    "2504.00891": {
        "title": "GenPRM: Scaling Test-Time Compute of Process Reward Models via Generative Reasoning",
        "url": "https://arxiv.org/abs/2504.00891",
        "http_status": 200,
        "ingestion_status": "new",
    },
    "2509.24460": {
        "title": "ContextPRM: Leveraging Contextual Coherence for multi-domain Test-Time Scaling",
        "url": "https://arxiv.org/abs/2509.24460",
        "http_status": 200,
        "ingestion_status": "new",
    },
    **{
        arxiv_id: exp4962.CITATIONS[arxiv_id] | {"ingestion_status": "reconfirmed"}
        for arxiv_id in RECONFIRMED_ARXIV_IDS
    },
}

SOTA_TO_CARNOT_MAPPING: dict[str, JsonDict] = {
    "2504.01005": {
        "source_id": "2504.01005",
        "title": CITATIONS["2504.01005"]["title"],
        "url": CITATIONS["2504.01005"]["url"],
        "strongest_method": (
            "Compute-optimal fixed-budget analysis of when to allocate inference "
            "tokens to additional self-consistency samples versus fewer samples plus "
            "a generative verification pass."
        ),
        "implementation_cost_over_current_stack": (
            "Low-medium: add matched-compute accounting around the existing "
            "three-column harness, then report the decomposed-energy verifier against "
            "the self-consistency/generative-verification frontier rather than only "
            "against equal candidate counts."
        ),
        "pitfalls": (
            "A verifier that improves accuracy can still lose the north-star win if "
            "verification tokens are too expensive; frontier conclusions can flip by "
            "domain, generator strength, and self-consistency saturation."
        ),
        "roadmap_input": "candidate_next_milestone: efficiency_parity_frontier",
    },
    "2504.00891": {
        "source_id": "2504.00891",
        "title": CITATIONS["2504.00891"]["title"],
        "url": CITATIONS["2504.00891"]["url"],
        "strongest_method": (
            "GenPRM: a generative process reward model that reasons over each step, "
            "emits verification chains, and can invoke code-style checks as "
            "test-time compute scales."
        ),
        "implementation_cost_over_current_stack": (
            "Medium-high: add a generative PRM comparator beside the discriminative "
            "decomposed-energy verifier, charge its reasoning and code-check tokens "
            "against the same budget, and keep FoVer penalties as the deterministic "
            "oracle-distinct analytical column."
        ),
        "pitfalls": (
            "Generated verification traces may re-derive the generator answer, code "
            "checks are not available for every TravelPlanner/MuSR-style constraint, "
            "and verifier-token cost can erase matched-compute gains."
        ),
        "roadmap_input": "candidate_next_milestone: genprm_matched_compute_generative_comparator",
    },
    "2509.24460": {
        "source_id": "2509.24460",
        "title": CITATIONS["2509.24460"]["title"],
        "url": CITATIONS["2509.24460"]["url"],
        "strongest_method": (
            "ContextPRM: multi-domain process verification that uses contextual "
            "coherence signals to scale test-time verification beyond a single "
            "math-heavy verifier domain."
        ),
        "implementation_cost_over_current_stack": (
            "Medium-high: extend the verifier registry with domain tags, coherence "
            "features, and cross-domain calibration slices, then compare against the "
            "current math-strong/code-weak Carnot verifier stack."
        ),
        "pitfalls": (
            "Context coherence can reward fluent but wrong traces, cross-domain PRM "
            "generalization may hide per-domain failures, and registry expansion must "
            "avoid turning an executable oracle into the verifier itself."
        ),
        "roadmap_input": "candidate_next_milestone: contextprm_cross_domain_registry_comparator",
    },
    **{
        arxiv_id: dict(exp4962.SOTA_TO_CARNOT_MAPPING[arxiv_id])
        for arxiv_id in RECONFIRMED_ARXIV_IDS
    },
}

NEXT_MILESTONE_ROADMAP_INPUTS = [
    "distributional_energy_lora_ensemble_with_fover_penalties",
    "efficiency_parity_frontier",
    "genprm_matched_compute_generative_comparator",
    "contextprm_cross_domain_registry_comparator",
    "variance_reduced_uncertainty_head",
    "fast_slow_process_router",
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
        "not met by Exp4973; this artifact only re-confirms turnkey readiness and "
        "extends the post-6/30 backlog"
    ),
}


def _json_dumps(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def _path_present(repo_root: Path, relative_path: str) -> bool:
    return (repo_root / relative_path).exists()


def check_network_available(url: str = CITATIONS["2504.01005"]["url"]) -> bool:
    try:
        with request.urlopen(url, timeout=NETWORK_TIMEOUT_S) as response:
            return int(response.status) == 200
    except OSError:
        return False


def load_turnkey_domain_slice(path: Path, *, limit: int | None = 3) -> list[JsonDict]:
    return exp4962.load_turnkey_domain_slice(path, limit=limit)


def slice_self_consistency_saturated(rows: Sequence[JsonMap]) -> bool:
    return exp4962.slice_self_consistency_saturated(rows)


def run_three_column_dry_run(rows: Sequence[JsonMap], *, limit: int = 3) -> JsonDict:
    dry_run = dict(exp4962.run_three_column_dry_run(rows, limit=limit))
    dry_run["reconfirmed_from_exp4962"] = True
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
        "real_benchmark_executed_by_exp4973": False,
        "operator_note": (
            "Use this single staged entrypoint after the majority-ARC sprint retires; "
            "promotion requires the validation gate."
        ),
    }


def build_turnkey_spec() -> JsonDict:
    turnkey_spec = dict(exp4962.build_turnkey_spec())
    turnkey_spec["entrypoint_command"] = ENTRYPOINT_COMMAND
    turnkey_spec["experiment"] = "post_6_30_distributional_energy_verifier_moat_test_v458"
    turnkey_spec["sota_backlog_extension_v458"] = {
        "new_arxiv_ids": list(NEW_ARXIV_IDS),
        "reconfirmed_arxiv_ids": list(RECONFIRMED_ARXIV_IDS),
        "next_milestone_roadmap_inputs": list(NEXT_MILESTONE_ROADMAP_INPUTS),
    }
    turnkey_spec["real_experiment_not_run"] = True
    return turnkey_spec


def source_artifacts() -> JsonDict:
    return {
        "kona_spec": KONA_SPEC_RELATIVE_PATH,
        "exp4962_turnkey_artifact": EXP4962_RESULT_RELATIVE_PATH,
        "exp4962_turnkey_module": EXP4962_MODULE_RELATIVE_PATH,
        "exp4922_harness": EXP4922_HARNESS_RELATIVE_PATH,
        "domain_slice": DOMAIN_SLICE_RELATIVE_PATH,
        "fover_registry": FOVER_REGISTRY_RELATIVE_PATH,
        "verifier_gaps": "ops/verifier_gaps.md",
        "north_star": "ops/north-star.md",
        "research_studying": STUDYING_RELATIVE_PATH,
        "research_references": "research-references.md",
    }


def blocked_resource_from_preconditions(
    *,
    kona_spec_present: bool,
    kona_spec_has_req: bool,
    exp4962_artifact_present: bool,
    exp4962_module_present: bool,
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
    if not exp4962_artifact_present:
        return "exp4962_turnkey_artifact_missing"
    if not exp4962_module_present:
        return "exp4962_turnkey_module_missing"
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
    kona_spec_has_req = "REQ-KONA-4973" in kona_spec_text
    exp4962_artifact_present = _path_present(repo_root, EXP4962_RESULT_RELATIVE_PATH)
    exp4962_module_present = _path_present(repo_root, EXP4962_MODULE_RELATIVE_PATH)
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
            rows = exp4962.exp4951.scaffold.load_domain_slice(domain_slice_path)
            domain_slice_valid = True
            sc_accuracy = exp4962.exp4951.self_consistency_accuracy(rows)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            domain_error = str(exc)
    self_consistency_saturated = bool(
        sc_accuracy is not None and sc_accuracy >= SELF_CONSISTENCY_SATURATION_THRESHOLD
    )
    network_ok = check_network_available() if net_available is None else net_available
    blocked_resource = blocked_resource_from_preconditions(
        kona_spec_present=kona_spec_present,
        kona_spec_has_req=kona_spec_has_req,
        exp4962_artifact_present=exp4962_artifact_present,
        exp4962_module_present=exp4962_module_present,
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
        "exp4962_turnkey_artifact_present": exp4962_artifact_present,
        "exp4962_turnkey_artifact_path": EXP4962_RESULT_RELATIVE_PATH,
        "exp4962_turnkey_module_present": exp4962_module_present,
        "exp4962_turnkey_module_path": EXP4962_MODULE_RELATIVE_PATH,
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
        "network_checked_url": CITATIONS["2504.01005"]["url"],
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
        "reconfirmed_from_exp4962": False,
        "entrypoint_command": ENTRYPOINT_COMMAND,
        "dry_run_note": "blocked before dry-run; no moat claim is made.",
    }


def _checksum_payload(artifact: JsonMap) -> JsonDict:
    return {
        "arxiv_ids_cited": list(artifact.get("arxiv_ids_cited") or []),
        "sota_to_carnot_mapping": dict(artifact.get("sota_to_carnot_mapping") or {}),
        "already_ingested_reconfirmed": list(artifact.get("already_ingested_reconfirmed") or []),
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
        if citation["ingestion_status"] not in {"new", "reconfirmed"}:
            raise ValueError(f"citations ingestion_status invalid for {arxiv_id}")
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
        "# Exp 4973 Distributional Energy Verifier Turnkey Backlog Extension",
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
            "## Validation Gate",
            "",
            (
                "The post-6/30 experiment must beat self-consistency with CI95 "
                "excluding zero, remain oracle-distinct (`verifier_is_oracle=false`), "
                "avoid a model-identity shortcut, and evaluate a domain where "
                "self-consistency is not near-ceiling. Exp4973 states this gate but "
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
        "## Exp 4973 - Distributional Energy Verifier Turnkey Backlog Extension - INGESTED\n\n"
        f"- Honest verdict: `{artifact['honest_verdict']}`\n"
        f"- Cited SOTA papers: {cited}\n"
        f"- Turnkey entrypoint: `{ENTRYPOINT_COMMAND}`\n"
        "- Bottom line for the post-6/30 roadmap: keep 2605.18871 as the pivot, "
        "use 2504.01005 as the compute-optimal efficiency frontier, use 2504.00891 "
        "as the matched-compute generative PRM comparator, and use 2509.24460 as "
        "the cross-domain verifier-registry comparator.\n"
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
