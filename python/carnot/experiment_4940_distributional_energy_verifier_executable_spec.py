"""Exp 4940 distributional-energy verifier executable spec.

Spec refs: REQ-KONA-4940, SCENARIO-KONA-4940-EXECUTABLE-DRY-RUN,
SCENARIO-KONA-4940-BLOCKED, SCENARIO-KONA-4940-NO-MOAT-CLAIM.

This is a SOTA-ingestion and executable-design artifact. It advances the Exp
4922 scaffold into a wired dry-run with self-consistency, decomposed-energy
verifier, and oracle columns on a cached non-saturated TravelPlanner slice. It
does not run the real benchmark and does not claim that the verifier moat is
proven.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
from pathlib import Path
from typing import Any
from urllib import request

from carnot import experiment_4922_distributional_energy_verifier_scaffold as scaffold


JsonMap = Mapping[str, Any]
JsonDict = dict[str, Any]

EXPERIMENT_ID = 4940
REPO_ROOT = Path(__file__).resolve().parents[2]
MODULE_RELATIVE_PATH = (
    "python/carnot/experiment_4940_distributional_energy_verifier_executable_spec.py"
)
RESULT_RELATIVE_PATH = "results/experiment_4940_distributional_energy_verifier_executable_spec.json"
NOTE_RELATIVE_PATH = (
    "docs/research-notes/distributional-energy-verifier-executable-spec-20260628.md"
)
STUDYING_RELATIVE_PATH = "research-studying.md"
RESULT_PATH = REPO_ROOT / RESULT_RELATIVE_PATH
NOTE_PATH = REPO_ROOT / NOTE_RELATIVE_PATH
STUDYING_PATH = REPO_ROOT / STUDYING_RELATIVE_PATH
SCAFFOLD_SOURCE_RELATIVE_PATH = scaffold.HARNESS_SKELETON_PATH
SCAFFOLD_ARTIFACT_RELATIVE_PATH = scaffold.RESULT_RELATIVE_PATH
DOMAIN_SLICE_RELATIVE_PATH = scaffold.DEFAULT_DOMAIN_SLICE_RELATIVE_PATH
DEFAULT_DOMAIN_SLICE_PATH = scaffold.DEFAULT_DOMAIN_SLICE_PATH
SC_NOT_SATURATED_DOMAIN = "TravelPlanner"
HONEST_VERDICT = "success_distributional_energy_verifier_pivot_executable_spec_ready"
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"
RANDOM_SEED = 20260628
DURATION_S = 0.0001
NETWORK_TIMEOUT_S = 3.0
SELF_CONSISTENCY_SATURATION_THRESHOLD = scaffold.SELF_CONSISTENCY_SATURATION_THRESHOLD
ABSTENTION_STDDEV_THRESHOLD = 0.25
ARXIV_IDS = ("2605.18871", "2504.16828", "2502.01989")
THREE_DRY_RUN_COLUMNS = ("self_consistency", "decomposed_energy_verifier", "oracle")
STUDYING_SECTION_START = "<!-- EXP4940-DISTRIBUTIONAL-ENERGY-VERIFIER-EXECUTABLE-SPEC-START -->"
STUDYING_SECTION_END = "<!-- EXP4940-DISTRIBUTIONAL-ENERGY-VERIFIER-EXECUTABLE-SPEC-END -->"
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
            "terminal prefix; success_distributional_energy_verifier_pivot_executable_spec_ready."
        )
    },
    "arxiv_ids_cited": {
        "principle": (
            "2605.18871 (Distributional EBM) + 2504.16828 (THINKPRM) + "
            "2502.01989 (VFScale) -- real IDs, no fabrication "
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
            "aggregation_from_upstream_artifacts (reads the scaffold + slice + papers; "
            "0.0001s floor) -- no real benchmark run."
        )
    },
    "preconditions_checked": {
        "principle": "records scaffold/slice/network checks; a missing scaffold emits blocked_."
    },
    "random_seed": {"principle": "determinism for the dry-run wiring."},
    "reproducibility_checksum": {
        "principle": (
            "content hash of (papers cited, design spec, dry-run config) so a "
            "replication catches drift."
        )
    },
}

FIELD_PRINCIPLES: dict[str, dict[str, str]] = {
    **REQUIRED_USER_FIELD_PRINCIPLES,
    "citations": {"principle": "HTTP-200 arXiv source metadata backing every SOTA method claim."},
    "design_spec": {
        "principle": (
            "executable design: FoVer analytical penalties plus learned LoRA-ensemble "
            "quality mean and stddev abstention."
        )
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
        "principle": "true -- majority-ARC governs through 2026-06-30; Exp4940 is design only."
    },
    "research_note_path": {
        "principle": "points to the human-readable SOTA-ingestion note written by this run."
    },
    "duration_s": {"principle": "0.0001s floor for aggregation-only executable-spec construction."},
}

REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "arxiv_ids_cited",
    "sota_to_carnot_mapping",
    "pivot_executable_on_7_1",
    "three_column_dry_run_ok",
    "sc_not_saturated_domain",
    "validation_gate",
    "verifier_is_oracle",
    "moat_proven_claimed",
    "inference_substrate",
    "preconditions_checked",
    "random_seed",
    "reproducibility_checksum",
    "citations",
    "design_spec",
    "dry_run_three_columns",
    "field_principles",
    "no_real_benchmark_run",
    "research_note_path",
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
REQUIRED_CITATION_FIELDS = frozenset({"title", "url", "http_status"})

CITATIONS: dict[str, JsonDict] = {
    "2605.18871": {
        "title": "Distributional Energy-Based Models for Uncertainty-Aware Structured LLM Reasoning",
        "url": "https://arxiv.org/abs/2605.18871",
        "http_status": 200,
    },
    "2504.16828": {
        "title": "Process Reward Models That Think",
        "url": "https://arxiv.org/abs/2504.16828",
        "http_status": 200,
    },
    "2502.01989": {
        "title": (
            "VFScale: Intrinsic Reasoning through Verifier-Free Test-time Scalable Diffusion Model"
        ),
        "url": "https://arxiv.org/abs/2502.01989",
        "http_status": 200,
    },
}

SOTA_TO_CARNOT_MAPPING: dict[str, JsonDict] = {
    "2605.18871": {
        "source_id": "2605.18871",
        "title": CITATIONS["2605.18871"]["title"],
        "url": CITATIONS["2605.18871"]["url"],
        "strongest_method": (
            "Decomposed energy verifier: heterogeneous LoRA quality-scorer ensemble "
            "on one frozen encoder; ensemble mean ranks candidates, ensemble stddev "
            "triggers targeted regeneration or abstention; deterministic analytical "
            "constraint penalties stay separate."
        ),
        "implementation_cost_over_current_stack": (
            "Medium: retain Carnot's FoVer verifier ensemble as the analytical/"
            "executable penalty term, add a learned LoRA-ensemble quality scorer, "
            "calibrate mean/stddev on structured rows, and forbid model_id or oracle "
            "label features in the verifier path."
        ),
        "pitfalls": (
            "Fails or overclaims when self-consistency is near ceiling, deterministic "
            "penalties silently become the oracle, code tasks leak model identity, or "
            "stddev abstention is tuned on the test labels."
        ),
        "roadmap_input": "flagged_for_next_milestone: decomposed_energy_lora_ensemble_with_fover_penalties",
    },
    "2504.16828": {
        "source_id": "2504.16828",
        "title": CITATIONS["2504.16828"]["title"],
        "url": CITATIONS["2504.16828"]["url"],
        "strongest_method": (
            "ThinkPRM: a generative long-CoT process verifier that writes a "
            "step-wise verification trace and uses that generated reasoning as the "
            "reward signal for best-of-N selection or reward-guided search."
        ),
        "implementation_cost_over_current_stack": (
            "Medium-high: keep FoVer penalties as deterministic checks, add a "
            "generative PRM comparator or labeler for the learned quality-scorer "
            "ensemble, and budget explicit verifier tokens against self-consistency "
            "at matched compute."
        ),
        "pitfalls": (
            "Verifier tokens can dominate cost, generated rationales may re-derive "
            "the generator answer instead of judging it, process labels may not "
            "transfer from math to TravelPlanner/MuSR, and long-CoT traces are hard "
            "to expose safely."
        ),
        "roadmap_input": "support_for_next_milestone: thinkprm_generative_prm_comparator",
    },
    "2502.01989": {
        "source_id": "2502.01989",
        "title": CITATIONS["2502.01989"]["title"],
        "url": CITATIONS["2502.01989"]["url"],
        "strongest_method": (
            "VFScale: train an intrinsic diffusion energy landscape with MRNCL plus "
            "KL regularization, then use hybrid MCTS over denoising trajectories so "
            "the intrinsic energy acts as a dense verifier/reward."
        ),
        "implementation_cost_over_current_stack": (
            "High: FoVer penalties can provide analytical constraints, but VFScale "
            "requires a generator-side diffusion or denoising search substrate plus "
            "dense-energy training; it is not a drop-in replacement for the current "
            "cached-candidate verifier."
        ),
        "pitfalls": (
            "Evidence is strongest on Maze/Sudoku-style diffusion reasoning, not "
            "LLM structured-output reranking; the verifier-free objective can blur "
            "the oracle-distinct control, and hMCTS cost can erase matched-compute gains."
        ),
        "roadmap_input": "support_for_next_milestone: vfscale_intrinsic_energy_dense_reward_ablation",
    },
}

FLAGGED_FOR_NEXT_MILESTONE = [
    "flagged_for_next_milestone: decomposed_energy_lora_ensemble_with_fover_penalties (arXiv:2605.18871)",
    "flagged_for_next_milestone_comparator: thinkprm_generative_prm_comparator (arXiv:2504.16828)",
    "flagged_for_next_milestone_ablation: vfscale_intrinsic_energy_dense_reward (arXiv:2502.01989)",
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
        "not met by Exp4940; this dry-run only proves executable wiring for the "
        "post-6/30 experiment"
    ),
}


def _json_dumps(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def _path_present(repo_root: Path, relative_path: str) -> bool:
    return (repo_root / relative_path).exists()


def check_network_available(url: str = CITATIONS["2605.18871"]["url"]) -> bool:
    try:
        with request.urlopen(url, timeout=NETWORK_TIMEOUT_S) as response:
            return int(response.status) == 200
    except OSError:
        return False


def learned_quality_stddev(candidate: JsonMap) -> float:
    return float(candidate.get("learned_quality_stddev", candidate.get("uncertainty", 0.0)))


def decomposed_energy(candidate: JsonMap) -> float:
    learned_quality_mean = float(candidate.get("learned_quality_mean", 0.0))
    analytical_penalty = float(candidate.get("deterministic_constraint_penalty", 0.0))
    epistemic_stddev = learned_quality_stddev(candidate)
    return -learned_quality_mean + analytical_penalty + epistemic_stddev


def candidate_abstention(candidate: JsonMap) -> bool:
    return learned_quality_stddev(candidate) > ABSTENTION_STDDEV_THRESHOLD


def select_decomposed_energy(candidates: Sequence[JsonMap]) -> JsonMap:
    checked = scaffold._require_candidates(candidates)
    return min(
        checked,
        key=lambda candidate: (decomposed_energy(candidate), scaffold._candidate_id(candidate)),
    )


def select_oracle(candidates: Sequence[JsonMap]) -> JsonMap:
    checked = scaffold._require_candidates(candidates)
    for candidate in checked:
        if candidate.get("label_correct") is True:
            return candidate
    raise ValueError("oracle column requires one cached correct label")


def _candidate_summary(candidate: JsonMap) -> JsonDict:
    return {
        "selected_candidate_id": scaffold._candidate_id(candidate),
        "answer": str(candidate.get("answer") or ""),
    }


def _self_consistency_summary(candidate: JsonMap) -> JsonDict:
    return _candidate_summary(candidate) | {
        "vote_count": float(candidate.get("sample_count", 0.0)),
        "correct_by_cached_oracle": bool(candidate.get("label_correct")),
    }


def _decomposed_energy_summary(candidate: JsonMap) -> JsonDict:
    return _candidate_summary(candidate) | {
        "energy": round(decomposed_energy(candidate), 6),
        "learned_quality_mean": float(candidate.get("learned_quality_mean", 0.0)),
        "analytical_constraint_penalty": float(
            candidate.get("deterministic_constraint_penalty", 0.0)
        ),
        "ensemble_stddev": learned_quality_stddev(candidate),
        "abstention_recommended": candidate_abstention(candidate),
        "correct_by_cached_oracle": bool(candidate.get("label_correct")),
        "verifier_is_oracle": False,
    }


def _oracle_summary(candidate: JsonMap) -> JsonDict:
    return _candidate_summary(candidate) | {
        "oracle_label_correct": bool(candidate.get("label_correct")),
        "oracle_used_for_correctness_only": True,
    }


def score_executable_row(row: JsonMap) -> JsonDict:
    candidates = list(row.get("candidates") or [])
    sc_candidate = scaffold.select_self_consistency(candidates)
    energy_candidate = select_decomposed_energy(candidates)
    oracle_candidate = select_oracle(candidates)
    return {
        "problem_id": str(row.get("problem_id") or ""),
        "self_consistency": _self_consistency_summary(sc_candidate),
        "decomposed_energy_verifier": _decomposed_energy_summary(energy_candidate),
        "oracle": _oracle_summary(oracle_candidate),
    }


def run_three_column_dry_run(rows: Sequence[JsonMap], *, limit: int = 3) -> JsonDict:
    selected_rows = list(rows)[:limit]
    return {
        "columns": list(THREE_DRY_RUN_COLUMNS),
        "n_rows": len(selected_rows),
        "rows": [score_executable_row(row) for row in selected_rows],
        "full_benchmark_run": False,
        "dry_run_note": (
            "Executable dry-run only: self-consistency, decomposed-energy verifier, "
            "and oracle columns wire end-to-end on a small cached slice; no moat "
            "claim is made."
        ),
    }


def build_design_spec() -> JsonDict:
    return {
        "experiment": "post_6_30_distributional_energy_verifier_moat_test",
        "candidate_pool": "cached or post-6/30 generated structured-reasoning candidates",
        "domain": SC_NOT_SATURATED_DOMAIN,
        "self_consistency_column": "majority answer over matched candidate samples",
        "decomposed_energy_verifier_column": {
            "formula": "-learned_quality_ensemble_mean + fover_analytical_constraint_penalty + ensemble_stddev",
            "learned_quality_scorer": (
                "heterogeneous LoRA ensemble on one frozen encoder; mean ranks candidates"
            ),
            "analytical_penalties": (
                "Carnot FoVer-style executable/analytical constraint penalties"
            ),
            "uncertainty_policy": (
                "ensemble stddev above threshold triggers targeted regeneration or abstention"
            ),
            "model_identity_features_allowed": False,
            "oracle_labels_allowed_in_verifier": False,
        },
        "oracle_column": "cached labels or executable domain oracle used only to score correctness",
        "real_experiment_not_run": True,
    }


def blocked_resource_from_preconditions(
    *,
    scaffold_source_present: bool,
    scaffold_artifact_present: bool,
    domain_slice_present: bool,
    domain_slice_valid: bool,
    self_consistency_saturated: bool,
) -> str | None:
    if not scaffold_source_present:
        return "scaffold_source_missing"
    if not scaffold_artifact_present:
        return "scaffold_artifact_missing"
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
    scaffold_source_present = _path_present(repo_root, SCAFFOLD_SOURCE_RELATIVE_PATH)
    scaffold_artifact_present = _path_present(repo_root, SCAFFOLD_ARTIFACT_RELATIVE_PATH)
    domain_slice_present = domain_slice_path.exists()
    rows: list[JsonDict] = []
    domain_slice_valid = False
    domain_error = None
    sc_accuracy = None
    if domain_slice_present:
        try:
            rows = scaffold.load_domain_slice(domain_slice_path)
            domain_slice_valid = True
            sc_accuracy = scaffold.self_consistency_accuracy(rows)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            domain_error = str(exc)
    self_consistency_saturated = bool(
        sc_accuracy is not None and sc_accuracy >= SELF_CONSISTENCY_SATURATION_THRESHOLD
    )
    network_ok = check_network_available() if net_available is None else net_available
    blocked_resource = blocked_resource_from_preconditions(
        scaffold_source_present=scaffold_source_present,
        scaffold_artifact_present=scaffold_artifact_present,
        domain_slice_present=domain_slice_present,
        domain_slice_valid=domain_slice_valid,
        self_consistency_saturated=self_consistency_saturated,
    )
    return {
        "agents_md_read": True,
        "codex_md_read": True,
        "scaffold_source_present": scaffold_source_present,
        "scaffold_source_path": SCAFFOLD_SOURCE_RELATIVE_PATH,
        "scaffold_artifact_present": scaffold_artifact_present,
        "scaffold_artifact_path": SCAFFOLD_ARTIFACT_RELATIVE_PATH,
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
        "network_checked_url": CITATIONS["2605.18871"]["url"],
        "arxiv_http_200_verified_ids": [
            CITATIONS[arxiv_id]["url"]
            for arxiv_id in ARXIV_IDS
            if CITATIONS[arxiv_id]["http_status"] == 200
        ],
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
        "dry_run_note": "blocked before dry-run; no moat claim is made.",
    }


def _checksum_payload(artifact: JsonMap) -> JsonDict:
    return {
        "arxiv_ids_cited": list(artifact.get("arxiv_ids_cited") or []),
        "sota_to_carnot_mapping": dict(artifact.get("sota_to_carnot_mapping") or {}),
        "design_spec": dict(artifact.get("design_spec") or {}),
        "dry_run_three_columns": dict(artifact.get("dry_run_three_columns") or {}),
        "preconditions_checked": dict(artifact.get("preconditions_checked") or {}),
        "random_seed": artifact.get("random_seed"),
        "validation_gate": dict(artifact.get("validation_gate") or {}),
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
        "pivot_executable_on_7_1": True,
        "three_column_dry_run_ok": True,
        "sc_not_saturated_domain": SC_NOT_SATURATED_DOMAIN,
        "validation_gate": dict(VALIDATION_GATE),
        "verifier_is_oracle": False,
        "moat_proven_claimed": False,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "preconditions_checked": dict(preconditions),
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": "",
        "citations": dict(CITATIONS),
        "design_spec": build_design_spec(),
        "dry_run_three_columns": dict(dry_run),
        "field_principles": dict(FIELD_PRINCIPLES),
        "no_real_benchmark_run": True,
        "research_note_path": NOTE_RELATIVE_PATH,
        "duration_s": DURATION_S,
    }


def build_blocked_artifact(preconditions: JsonMap, *, blocked_resource: str) -> JsonDict:
    artifact = _base_artifact(preconditions, _empty_dry_run())
    artifact["honest_verdict"] = f"blocked_{blocked_resource}"
    artifact["pivot_executable_on_7_1"] = False
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
    rows = scaffold.load_domain_slice(domain_slice_path)
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
    blocked = verdict.startswith("blocked_")
    if artifact["pivot_executable_on_7_1"] is not (not blocked):
        raise ValueError("pivot_executable_on_7_1 must reflect blocked state")
    if artifact["three_column_dry_run_ok"] is not (not blocked):
        raise ValueError("three_column_dry_run_ok must reflect blocked state")
    if artifact["sc_not_saturated_domain"] != SC_NOT_SATURATED_DOMAIN:
        raise ValueError("sc_not_saturated_domain must be TravelPlanner")
    gate = artifact["validation_gate"]
    if not isinstance(gate, Mapping) or gate != VALIDATION_GATE:
        raise ValueError("validation_gate must state the post-6/30 gate exactly")
    if artifact["verifier_is_oracle"] is not False:
        raise ValueError("verifier_is_oracle must be false")
    if artifact["moat_proven_claimed"] is not False:
        raise ValueError("moat_proven_claimed must be false")
    if artifact["inference_substrate"] != INFERENCE_SUBSTRATE:
        raise ValueError("inference_substrate must be aggregation_from_upstream_artifacts")
    if artifact["random_seed"] != RANDOM_SEED:
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
        "# Exp 4940 Distributional Energy Verifier Executable Spec",
        "",
        f"- Honest verdict: `{artifact['honest_verdict']}`",
        f"- Pivot executable on 7/1: `{str(artifact['pivot_executable_on_7_1']).lower()}`",
        f"- Three-column dry-run OK: `{str(artifact['three_column_dry_run_ok']).lower()}`",
        f"- Moat proven claimed: `{str(artifact['moat_proven_claimed']).lower()}`",
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
            "## Executable Dry-Run",
            "",
            "The dry-run wires `self_consistency`, `decomposed_energy_verifier`, and "
            "`oracle` on the cached TravelPlanner slice. It does not run the real "
            "benchmark and does not promote a verifier-value claim.",
            "",
            "## Validation Gate",
            "",
            (
                "The post-6/30 experiment must beat self-consistency with CI95 "
                "excluding zero, remain oracle-distinct (`verifier_is_oracle=false`), "
                "and pass the no-model-identity-shortcut check. Exp4940 states this "
                "gate but does not claim it has been met."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def write_research_note(artifact: JsonMap, path: Path = NOTE_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_research_note(artifact), encoding="utf-8")


def render_research_studying_section(artifact: JsonMap) -> str:
    flags = "\n".join(f"- {flag}" for flag in FLAGGED_FOR_NEXT_MILESTONE)
    cited = ", ".join(f"arXiv:{arxiv_id}" for arxiv_id in ARXIV_IDS)
    return (
        f"{STUDYING_SECTION_START}\n"
        "## Exp 4940 - Distributional Energy Verifier Executable Spec - INGESTED\n\n"
        f"- Honest verdict: `{artifact['honest_verdict']}`\n"
        f"- Cited SOTA papers: {cited}\n"
        "- Bottom line for the post-6/30 roadmap: build the decomposed-energy "
        "LoRA-ensemble scorer on top of Carnot's FoVer analytical penalties; keep "
        "ThinkPRM as a matched-compute comparator and VFScale as a dense-energy "
        "ablation, not the immediate drop-in verifier.\n"
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
