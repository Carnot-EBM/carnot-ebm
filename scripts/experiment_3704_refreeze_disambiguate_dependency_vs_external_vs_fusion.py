#!/usr/bin/env python3
"""Exp 3704: disambiguate the FoVer re-freeze candidate.

Spec: REQ-PUBLISH-3704, SCENARIO-PUBLISH-3704.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import importlib
import importlib.util
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from carnot.verify import dependency_aware_dual_condition_integrity as exp3680
from carnot.verify import dependency_aware_weighting_clean as exp3667
from carnot.verify import external_comparator_dependency_vs_deentangled as exp3693
from carnot.verify import weaver_peer_comparison_v3 as exp3644


REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_REL_PATH = Path(
    "results/experiment_3704_refreeze_disambiguate_dependency_vs_external_vs_fusion.json"
)
NORTH_STAR_REL_PATH = Path("ops/north-star.md")
CI_WORKFLOW_REL_PATH = Path(".github/workflows/reproduce-fover-headline.yml")
FROZEN_HEADLINE_AUROC = 0.9131
DEFAULT_RANDOM_SEED = 3704
DEFAULT_RANDOM_SEEDS = exp3680.DEFAULT_RANDOM_SEEDS
DEFAULT_BOOTSTRAP_REPS = exp3680.DEFAULT_BOOTSTRAP_REPS
DEFAULT_CROSSFIT_FOLDS = exp3667.DEFAULT_CROSSFIT_FOLDS
INFERENCE_SUBSTRATE = (
    "verifier_ensemble_against_cached_candidates (principle: scores cached "
    "FoVer outputs; no LLM load; no compute-bound marker)."
)

SUCCESS_TEMPLATE = "complete: refreeze_disambiguated_winner_"
SUCCESS_SUFFIX = "_beats_frozen_clean_package_reemitted_frozen_headline_unchanged"
NO_CANDIDATE_VERDICT = "complete: refreeze_disambiguated_no_candidate_beats_frozen_headline_stays_0_9131"
BLOCKED_VERDICT = "complete: blocked_fover_corpus_or_baselines_unavailable"
TERMINAL_VERDICTS = (NO_CANDIDATE_VERDICT, BLOCKED_VERDICT)
CANDIDATE_FIELDS = {
    "dependency_aware": "dependency_aware_auroc",
    "external": "external_comparator_auroc",
    "fusion": "fusion_auroc",
}

REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "inference_substrate",
    "dependency_aware_auroc",
    "external_comparator_auroc",
    "fusion_auroc",
    "carnot_current_auroc",
    "strongest_candidate",
    "strongest_candidate_auroc",
    "winner_vs_runnerup_delta_ci",
    "strongest_candidate_beats_frozen",
    "refreeze_package_reemitted_for_winner",
    "operator_checklist",
    "adversarial_verify_clean",
    "north_star_unmodified_assert",
    "frozen_headline_unchanged_assert",
    "n_seeds",
    "n_examples",
    "random_seed",
    "reproducibility_checksum",
    "duration_s",
)

FIELD_PRINCIPLES = {
    "honest_verdict": "Terminal prefix for reconciler classification.",
    "inference_substrate": INFERENCE_SUBSTRATE,
    "dependency_aware_auroc": (
        "Carnot's label-conditional dependency-aware candidate under the frozen "
        "dual-condition protocol -- one field, no alias."
    ),
    "external_comparator_auroc": (
        "The published de-entangled/CIG baseline (arXiv:2604.07650) under the "
        "identical protocol -- a DISTINCT measurement."
    ),
    "fusion_auroc": (
        "The dependency-aware + external FUSION candidate (arXiv:2502.20379) -- "
        "the third DISTINCT candidate."
    ),
    "carnot_current_auroc": (
        "Carnot-current under the identical protocol -- must reproduce the "
        "frozen 0.9131 as the protocol sanity check."
    ),
    "strongest_candidate": (
        "dependency_aware / external / fusion -- which single candidate has the "
        "highest pooled AUROC."
    ),
    "strongest_candidate_auroc": (
        "The winning candidate's pooled AUROC object -- source field plus the "
        "number the operator would re-freeze to without creating a second "
        "top-level AUROC alias."
    ),
    "winner_vs_runnerup_delta_ci": (
        "Paired delta + bootstrap CI + DeLong p of winner vs runner-up -- a "
        "point estimate cannot decide the ranking."
    ),
    "strongest_candidate_beats_frozen": (
        "BARE bool. True iff the winning candidate's AUROC > frozen 0.9131 "
        "with the delta CI excluding 0 -- whether ANY candidate is a defensible "
        "re-freeze. STORE AS BARE true/false."
    ),
    "refreeze_package_reemitted_for_winner": (
        "True iff a clean operator re-freeze package was re-emitted for the "
        "WINNING candidate (not the prior dependency-aware-only package)."
    ),
    "operator_checklist": (
        "The ordered OPERATOR-ACTION steps (north-star sec-1 edit, CI-workflow "
        "update, trigger run) for the winner -- the re-freeze is operator-only."
    ),
    "adversarial_verify_clean": (
        "True iff the artifact passes adversarial_verify with no critical flag."
    ),
    "north_star_unmodified_assert": (
        "Asserts ops/north-star.md was NOT edited (operator-curated)."
    ),
    "frozen_headline_unchanged_assert": (
        "Asserts the publication gate still reads 0.9131 and paper_ready is "
        "unchanged -- no silent substitution."
    ),
    "n_seeds": "Replication count (>=5).",
    "n_examples": "Sample-size rigor (FoVer n>=1000).",
    "random_seed": "Determinism precondition.",
    "reproducibility_checksum": "Drift detection.",
    "duration_s": "Plausibility floor.",
}


@dataclass(frozen=True)
class FusionResult:
    """Cross-fit fusion scores and the fold alphas that produced them."""

    scores: np.ndarray
    mean_alpha: float
    fold_alphas: list[float]


@dataclass(frozen=True)
class CandidateScorePanel:
    """One seed's production and architecture-only candidate score vectors."""

    seed: int
    labels: Sequence[int]
    dependency_scores: Sequence[float]
    external_scores: Sequence[float]
    fusion_scores: Sequence[float]
    carnot_current_scores: Sequence[float]
    dependency_architecture_scores: Sequence[float]
    external_architecture_scores: Sequence[float]
    fusion_architecture_scores: Sequence[float]
    subset_sha256: str | None = None
    fusion_alpha_mean: float | None = None


def fusion_crossfit_scores(
    *,
    labels: Sequence[int] | np.ndarray,
    dependency_scores: Sequence[float] | np.ndarray,
    external_scores: Sequence[float] | np.ndarray,
    random_seed: int,
    n_folds: int,
) -> FusionResult:
    """Combine dependency-aware and external signals with cross-fit weights."""

    labels_arr, dependency_arr = exp3667.checked_label_scores(labels, dependency_scores)
    _, external_arr = exp3667.checked_label_scores(labels, external_scores)
    _require_binary_labels(labels_arr)
    dependency_norm = _minmax01(dependency_arr)
    external_norm = _minmax01(external_arr)
    fold_ids = _stratified_fold_ids(labels_arr, n_folds=n_folds, random_seed=random_seed)
    scores = np.zeros(len(labels_arr), dtype=np.float64)
    alphas: list[float] = []
    for fold in sorted(set(fold_ids.tolist())):
        train_idx = np.where(fold_ids != fold)[0]
        test_idx = np.where(fold_ids == fold)[0]
        alpha = _best_fusion_alpha(
            labels_arr[train_idx],
            dependency_norm[train_idx],
            external_norm[train_idx],
        )
        scores[test_idx] = alpha * dependency_norm[test_idx] + (1.0 - alpha) * external_norm[test_idx]
        alphas.append(_round_metric(alpha))
    return FusionResult(
        scores=scores,
        mean_alpha=_round_metric(float(np.mean(alphas))),
        fold_alphas=alphas,
    )


def panel_from_condition_row(row: exp3680.ConditionScoreRows) -> CandidateScorePanel:
    """Compute dependency-aware, external, fusion, and current scores for one row panel."""

    names = list(exp3644.VERIFIER_NAMES)
    labels = np.asarray(row.labels, dtype=np.int64)
    production_matrix = exp3693.exp3656.score_matrix(row.production_scores_by_verifier, names)
    architecture_matrix = exp3693.exp3656.score_matrix(row.architecture_scores_by_verifier, names)
    production = exp3667.score_weighting_panel(
        labels=labels,
        score_matrix=production_matrix,
        verifier_names=names,
        random_seed=int(row.seed),
    )
    architecture = exp3667.score_weighting_panel(
        labels=labels,
        score_matrix=architecture_matrix,
        verifier_names=names,
        random_seed=int(row.seed),
    )
    external = exp3693.cig_deentangled_crossfit_scores(
        labels=labels,
        score_matrix=production_matrix,
        verifier_names=names,
        random_seed=int(row.seed),
        n_folds=DEFAULT_CROSSFIT_FOLDS,
    )
    external_arch = exp3693.cig_deentangled_crossfit_scores(
        labels=labels,
        score_matrix=architecture_matrix,
        verifier_names=names,
        random_seed=int(row.seed),
        n_folds=DEFAULT_CROSSFIT_FOLDS,
    )
    fusion = fusion_crossfit_scores(
        labels=labels,
        dependency_scores=production["dependency_aware_proper"],
        external_scores=external.scores,
        random_seed=int(row.seed),
        n_folds=DEFAULT_CROSSFIT_FOLDS,
    )
    fusion_arch = fusion_crossfit_scores(
        labels=labels,
        dependency_scores=architecture["dependency_aware_proper"],
        external_scores=external_arch.scores,
        random_seed=int(row.seed),
        n_folds=DEFAULT_CROSSFIT_FOLDS,
    )
    return CandidateScorePanel(
        seed=int(row.seed),
        labels=labels,
        dependency_scores=np.asarray(production["dependency_aware_proper"], dtype=np.float64),
        external_scores=np.asarray(external.scores, dtype=np.float64),
        fusion_scores=np.asarray(fusion.scores, dtype=np.float64),
        carnot_current_scores=np.asarray(production["carnot_current"], dtype=np.float64),
        dependency_architecture_scores=np.asarray(
            architecture["dependency_aware_proper"],
            dtype=np.float64,
        ),
        external_architecture_scores=np.asarray(external_arch.scores, dtype=np.float64),
        fusion_architecture_scores=np.asarray(fusion_arch.scores, dtype=np.float64),
        subset_sha256=row.subset_sha256,
        fusion_alpha_mean=fusion.mean_alpha,
    )


def build_artifact_from_panels(
    *,
    repo_root: Path,
    panels: Sequence[CandidateScorePanel],
    started_s: float,
    now_s: float | None,
    preconditions: Sequence[Mapping[str, Any]],
    publication_gate_before: Mapping[str, Any],
    publication_gate_after: Mapping[str, Any],
    north_star_hash_before: str,
    north_star_hash_after: str,
    ci_workflow_hash_before: str,
    ci_workflow_hash_after: str,
    github_run_triggered: bool,
    adversarial_verify_clean: bool,
    random_seed: int = DEFAULT_RANDOM_SEED,
    bootstrap_seeds: Sequence[int] = DEFAULT_RANDOM_SEEDS,
    n_bootstrap: int = DEFAULT_BOOTSTRAP_REPS,
) -> dict[str, Any]:
    """Assemble the final Exp 3704 artifact from scored candidate panels."""

    rows = list(panels)
    if not rows:
        raise ValueError("at least one candidate panel is required")
    labels = np.concatenate([np.asarray(row.labels, dtype=np.int64) for row in rows])
    score_vectors = {
        "dependency_aware": np.concatenate(
            [np.asarray(row.dependency_scores, dtype=np.float64) for row in rows]
        ),
        "external": np.concatenate([np.asarray(row.external_scores, dtype=np.float64) for row in rows]),
        "fusion": np.concatenate([np.asarray(row.fusion_scores, dtype=np.float64) for row in rows]),
        "carnot_current": np.concatenate(
            [np.asarray(row.carnot_current_scores, dtype=np.float64) for row in rows]
        ),
    }
    _assert_score_lengths(labels, score_vectors)
    aurocs = {
        "dependency_aware": exp3644.tie_aware_auroc(labels, score_vectors["dependency_aware"]),
        "external": exp3644.tie_aware_auroc(labels, score_vectors["external"]),
        "fusion": exp3644.tie_aware_auroc(labels, score_vectors["fusion"]),
        "carnot_current": exp3644.tie_aware_auroc(labels, score_vectors["carnot_current"]),
    }
    ranking = sorted(
        ("dependency_aware", "external", "fusion"),
        key=lambda name: (-float(aurocs[name]), name),
    )
    winner, runner_up = ranking[0], ranking[1]
    winner_vs_runner = _paired_delta_with_delong(
        labels,
        score_vectors[winner],
        score_vectors[runner_up],
        seeds=bootstrap_seeds,
        n_bootstrap=n_bootstrap,
        first=winner,
        second=runner_up,
    )
    winner_vs_frozen = _paired_delta_with_delong(
        labels,
        score_vectors[winner],
        score_vectors["carnot_current"],
        seeds=bootstrap_seeds,
        n_bootstrap=n_bootstrap,
        first=winner,
        second="frozen_0_9131_carnot_current_vector",
    )
    winner_beats_frozen = bool(
        aurocs[winner] > FROZEN_HEADLINE_AUROC
        and list(winner_vs_frozen.get("ci95") or [0.0])[0] > 0.0
        and max(aurocs[name] for name in ("dependency_aware", "external", "fusion")) < 0.99
    )
    north_star_unmodified = north_star_hash_before == north_star_hash_after
    ci_workflow_unmodified = (
        ci_workflow_hash_before == ci_workflow_hash_after and not bool(github_run_triggered)
    )
    frozen_headline_unchanged = (
        publication_gate_before.get("paper_ready") == publication_gate_after.get("paper_ready")
        and publication_gate_reads_frozen_0_9131(publication_gate_after)
    )
    refreeze = bool(
        winner_beats_frozen
        and adversarial_verify_clean
        and north_star_unmodified
        and ci_workflow_unmodified
        and frozen_headline_unchanged
    )
    artifact = {
        "artifact": "experiment_3704_refreeze_disambiguate_dependency_vs_external_vs_fusion",
        "schema": "carnot.refreeze_disambiguation.v1",
        "honest_verdict": classify_honest_verdict(winner, refreeze),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "dependency_aware_auroc": _round_metric(aurocs["dependency_aware"]),
        "external_comparator_auroc": _round_metric(aurocs["external"]),
        "fusion_auroc": _round_metric(aurocs["fusion"]),
        "carnot_current_auroc": _round_metric(aurocs["carnot_current"]),
        "frozen_headline_auroc": FROZEN_HEADLINE_AUROC,
        "candidate_auroc_ci95": {
            "dependency_aware": exp3667.bootstrap_auroc_ci(
                labels,
                score_vectors["dependency_aware"],
                seeds=bootstrap_seeds,
                n_bootstrap=n_bootstrap,
            ),
            "external": exp3667.bootstrap_auroc_ci(
                labels,
                score_vectors["external"],
                seeds=bootstrap_seeds,
                n_bootstrap=n_bootstrap,
            ),
            "fusion": exp3667.bootstrap_auroc_ci(
                labels,
                score_vectors["fusion"],
                seeds=bootstrap_seeds,
                n_bootstrap=n_bootstrap,
            ),
        },
        "strongest_candidate": winner,
        "strongest_candidate_auroc": {
            "candidate": winner,
            "source_field": CANDIDATE_FIELDS[winner],
            "value": _round_metric(aurocs[winner]),
        },
        "runner_up_candidate": runner_up,
        "winner_vs_runnerup_delta_ci": winner_vs_runner,
        "winner_vs_frozen_delta_ci": winner_vs_frozen,
        "strongest_candidate_beats_frozen": winner_beats_frozen,
        "refreeze_package_reemitted_for_winner": refreeze,
        "operator_checklist": build_operator_checklist(winner, artifact_path=OUTPUT_REL_PATH)
        if refreeze
        else [],
        "adversarial_verify_clean": bool(adversarial_verify_clean),
        "north_star_unmodified_assert": bool(north_star_unmodified),
        "ci_workflow_unmodified_assert": bool(ci_workflow_unmodified),
        "frozen_headline_unchanged_assert": bool(frozen_headline_unchanged),
        "github_actions_run_triggered": bool(github_run_triggered),
        "publication_gate_paper_ready_before": publication_gate_before.get("paper_ready"),
        "publication_gate_paper_ready_after": publication_gate_after.get("paper_ready"),
        "n_seeds": int(len(rows)),
        "n_examples": int(min(len(row.labels) for row in rows)),
        "n_pooled_examples": int(len(labels)),
        "random_seed": int(random_seed),
        "random_seeds_used": [int(row.seed) for row in rows],
        "bootstrap_seeds": [int(seed) for seed in bootstrap_seeds],
        "n_bootstrap_per_seed": int(n_bootstrap),
        "reproducibility_checksum": reproducibility_checksum(
            panels=rows,
            score_vectors=score_vectors,
            random_seed=random_seed,
            bootstrap_seeds=bootstrap_seeds,
            gate_after=publication_gate_after,
        ),
        "duration_s": _round_metric(_duration(started_s, now_s)),
        "leak_free": bool(max(aurocs[name] for name in ("dependency_aware", "external", "fusion")) < 0.99),
        "candidate_ranking": [
            {"candidate": name, "pooled_auroc": _round_metric(aurocs[name])} for name in ranking
        ],
        "architecture_only_aurocs": architecture_aurocs(rows),
        "per_seed_results": per_seed_results(rows),
        "score_vector_checksums": {
            "dependency_aware": vector_checksum(score_vectors["dependency_aware"]),
            "external_comparator": vector_checksum(score_vectors["external"]),
            "fusion": vector_checksum(score_vectors["fusion"]),
            "carnot_current": vector_checksum(score_vectors["carnot_current"]),
        },
        "operator_package_note": (
            "The frozen 0.9131 stays the headline until the operator completes "
            "the checklist and the CI reproducer re-runs green on the new number."
        ),
        "acceptance_gate": {
            "condition": (
                "dependency_aware_auroc present AND external_comparator_auroc present "
                "AND fusion_auroc present AND winner_vs_runnerup_delta_ci present AND "
                "adversarial_verify_clean == true AND north_star_unmodified_assert == true "
                "AND frozen_headline_unchanged_assert == true"
            ),
            "principle": (
                "A trustworthy re-freeze disambiguation requires all three "
                "candidates measured under the identical frozen protocol with "
                "the paired ranking CI, adversarial-clean, and the "
                "operator-curated headline + frozen number untouched -- otherwise "
                "the operator re-freezes to a candidate that is not actually the strongest."
            ),
            "passed": bool(
                adversarial_verify_clean
                and north_star_unmodified
                and frozen_headline_unchanged
                and all(aurocs[name] is not None for name in ("dependency_aware", "external", "fusion"))
                and winner_vs_runner is not None
            ),
        },
        "preconditions_checked": [dict(item) for item in preconditions],
        "field_principles": dict(FIELD_PRINCIPLES),
    }
    validate_artifact(artifact)
    return artifact


def blocked_artifact(
    *,
    started_s: float,
    now_s: float | None,
    random_seed: int,
    preconditions: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Return the closed blocked artifact when required inputs are unavailable."""

    payload = json.dumps(
        {"preconditions": [dict(item) for item in preconditions], "random_seed": random_seed},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    artifact = {
        "artifact": "experiment_3704_refreeze_disambiguate_dependency_vs_external_vs_fusion",
        "schema": "carnot.refreeze_disambiguation.v1",
        "honest_verdict": BLOCKED_VERDICT,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "dependency_aware_auroc": None,
        "external_comparator_auroc": None,
        "fusion_auroc": None,
        "carnot_current_auroc": None,
        "frozen_headline_auroc": FROZEN_HEADLINE_AUROC,
        "strongest_candidate": "blocked",
        "strongest_candidate_auroc": None,
        "winner_vs_runnerup_delta_ci": None,
        "winner_vs_frozen_delta_ci": None,
        "strongest_candidate_beats_frozen": False,
        "refreeze_package_reemitted_for_winner": False,
        "operator_checklist": [],
        "adversarial_verify_clean": False,
        "north_star_unmodified_assert": False,
        "ci_workflow_unmodified_assert": False,
        "frozen_headline_unchanged_assert": False,
        "github_actions_run_triggered": False,
        "n_seeds": 0,
        "n_examples": 0,
        "random_seed": int(random_seed),
        "reproducibility_checksum": hashlib.sha256(payload).hexdigest(),
        "duration_s": _round_metric(_duration(started_s, now_s)),
        "preconditions_checked": [dict(item) for item in preconditions],
        "acceptance_gate": {"condition": "blocked before candidate scoring", "passed": False},
        "field_principles": dict(FIELD_PRINCIPLES),
    }
    validate_artifact(artifact)
    return artifact


def score_candidate_panels(repo_root: Path) -> list[CandidateScorePanel]:
    """Run all five frozen-protocol seeds and convert them to candidate panels."""

    root = Path(repo_root)
    state_files = exp3680.discover_fr11_state_files(root)
    rows = [
        exp3680.score_dual_condition_rows(
            root,
            seed=int(seed),
            n_examples=exp3680.DEFAULT_N_EXAMPLES,
            state_files=state_files,
        )
        for seed in DEFAULT_RANDOM_SEEDS
    ]
    return [panel_from_condition_row(row) for row in rows]


def probe_preconditions(repo_root: Path) -> list[dict[str, Any]]:
    """Check FoVer, baseline, external comparator, and reproducer availability."""

    checks = list(exp3693.probe_preconditions(Path(repo_root), n_examples=exp3680.DEFAULT_N_EXAMPLES))
    try:
        importlib.import_module("scripts.reproduce_fover_headline")
        reproducer_available = True
        detail = "scripts.reproduce_fover_headline importable"
    except ImportError as exc:
        reproducer_available = False
        detail = f"{type(exc).__name__}: {exc}"
    checks.append(
        {
            "resource": "fover_headline_reproducer_importable",
            "available": reproducer_available,
            "detail": detail,
        }
    )
    return [dict(item) for item in checks]


def evaluate_publication_gate(repo_root: Path) -> dict[str, Any]:
    """Evaluate the stable publication gate without editing files."""

    script = Path(repo_root) / "scripts" / "publication_gate.py"
    spec = importlib.util.spec_from_file_location("carnot_publication_gate_exp3704", script)
    if spec is None or spec.loader is None:
        return {"paper_ready": None, "error": f"could not import {script}"}
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.PROJECT_ROOT = Path(repo_root)
    module.STATE_PATH = Path(repo_root) / "ops" / "publication_gate_state.json"
    module.TECH_REPORT = Path(repo_root) / "docs" / "technical-report.md"
    module.PAPER_TEX = Path(repo_root) / "docs" / "arxiv-paper" / "main.tex"
    return dict(module.evaluate())


def write_artifact(
    repo_root: Path = REPO_ROOT,
    *,
    output_path: Path | str = OUTPUT_REL_PATH,
    started_s: float | None = None,
    now_s: float | None = None,
) -> Path:
    """Build, adversarially verify, and write the Exp 3704 terminal artifact."""

    root = Path(repo_root)
    start = time.time() if started_s is None else float(started_s)
    north_before = _sha256_file(root / NORTH_STAR_REL_PATH)
    workflow_before = _sha256_file(root / CI_WORKFLOW_REL_PATH)
    gate_before = evaluate_publication_gate(root)
    preconditions = probe_preconditions(root)
    if not all(bool(item.get("available")) for item in preconditions):
        artifact = blocked_artifact(
            started_s=start,
            now_s=now_s,
            random_seed=DEFAULT_RANDOM_SEED,
            preconditions=preconditions,
        )
    else:
        try:
            panels = score_candidate_panels(root)
            gate_after = evaluate_publication_gate(root)
            artifact = build_artifact_from_panels(
                repo_root=root,
                panels=panels,
                started_s=start,
                now_s=now_s,
                preconditions=preconditions,
                publication_gate_before=gate_before,
                publication_gate_after=gate_after,
                north_star_hash_before=north_before,
                north_star_hash_after=_sha256_file(root / NORTH_STAR_REL_PATH),
                ci_workflow_hash_before=workflow_before,
                ci_workflow_hash_after=_sha256_file(root / CI_WORKFLOW_REL_PATH),
                github_run_triggered=False,
                adversarial_verify_clean=False,
                random_seed=DEFAULT_RANDOM_SEED,
            )
        except Exception as exc:  # noqa: BLE001 - terminal artifact must fail closed.
            failed = [
                *preconditions,
                {
                    "resource": "candidate_panel_scoring",
                    "available": False,
                    "detail": f"{type(exc).__name__}: {exc}",
                },
            ]
            artifact = blocked_artifact(
                started_s=start,
                now_s=now_s,
                random_seed=DEFAULT_RANDOM_SEED,
                preconditions=failed,
            )
    target = root / output_path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if artifact["honest_verdict"] != BLOCKED_VERDICT:
        report = run_adversarial_verify_report(target)
        artifact = finalize_adversarial_fields(artifact, report)
        validate_artifact(artifact)
        target.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return target


def finalize_adversarial_fields(
    artifact: Mapping[str, Any],
    report: Mapping[str, Any],
) -> dict[str, Any]:
    """Update clean-package fields after the artifact verifier runs."""

    updated = dict(artifact)
    clean = adversarial_report_is_clean(report)
    updated["adversarial_verify_clean"] = clean
    refreeze = bool(
        clean
        and updated.get("strongest_candidate_beats_frozen") is True
        and updated.get("north_star_unmodified_assert") is True
        and updated.get("ci_workflow_unmodified_assert") is True
        and updated.get("frozen_headline_unchanged_assert") is True
    )
    updated["refreeze_package_reemitted_for_winner"] = refreeze
    updated["operator_checklist"] = (
        build_operator_checklist(str(updated["strongest_candidate"]), artifact_path=OUTPUT_REL_PATH)
        if refreeze
        else []
    )
    updated["honest_verdict"] = classify_honest_verdict(str(updated["strongest_candidate"]), refreeze)
    updated["acceptance_gate"] = {
        **dict(updated.get("acceptance_gate") or {}),
        "passed": bool(
            clean
            and updated.get("north_star_unmodified_assert") is True
            and updated.get("frozen_headline_unchanged_assert") is True
            and updated.get("winner_vs_runnerup_delta_ci") is not None
        ),
    }
    updated["adversarial_verify_report"] = {
        "flag_count": int(report.get("flag_count", 0)),
        "max_severity": report.get("max_severity"),
        "flags": list(report.get("flags") or []),
    }
    return updated


def run_adversarial_verify_report(path: Path) -> dict[str, Any]:
    """Run the repository adversarial verifier and return its structured report."""

    script_path = REPO_ROOT / "scripts" / "adversarial_verify.py"
    spec = importlib.util.spec_from_file_location("carnot_adversarial_verify_exp3704", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not import {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return dict(module.verify_artifact(Path(path)))


def adversarial_report_is_clean(report: Mapping[str, Any]) -> bool:
    """True when adversarial verification emitted no critical flag."""

    for flag in list(report.get("flags") or []):
        item = dict(flag)
        if str(item.get("severity")) == "critical" or str(item.get("kind")) == "TAUTOLOGY":
            return False
    return True


def build_operator_checklist(winner: str, *, artifact_path: Path) -> list[str]:
    """Build the operator-only package for the winning candidate."""

    label = winner.replace("_", "-")
    return [
        (
            "OPERATOR-ACTION: Edit ops/north-star.md Section 1 methods headline "
            f"from FoVer AUROC 0.9131 to the {label} winning candidate, using "
            f"strongest_candidate_auroc and CI from {artifact_path.as_posix()}."
        ),
        (
            "OPERATOR-ACTION: Update .github/workflows/reproduce-fover-headline.yml "
            f"assertion bounds to the {label} winner's recorded CI from "
            f"{artifact_path.as_posix()}."
        ),
        (
            "OPERATOR-ACTION: Trigger the GitHub Actions FoVer headline reproducer "
            "run and record the green run. The frozen 0.9131 stays the headline "
            "until this checklist is complete and the CI reproducer re-runs green "
            "on the new number."
        ),
    ]


def classify_honest_verdict(winner: str, refreeze_reemitted: bool) -> str:
    """Return the closed terminal verdict for the disambiguation outcome."""

    if refreeze_reemitted:
        return f"{SUCCESS_TEMPLATE}{winner}{SUCCESS_SUFFIX}"
    return NO_CANDIDATE_VERDICT


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Validate the Exp 3704 schema, non-edit assertions, and anti-copy checks."""

    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    if missing:
        raise ValueError(f"missing required artifact fields: {missing}")
    verdict = str(artifact.get("honest_verdict"))
    if verdict != BLOCKED_VERDICT and verdict != NO_CANDIDATE_VERDICT:
        if not (verdict.startswith(SUCCESS_TEMPLATE) and verdict.endswith(SUCCESS_SUFFIX)):
            raise ValueError(f"unsupported honest_verdict: {verdict!r}")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        raise ValueError("inference_substrate must use the cached-verifier sentinel")
    principles = artifact.get("field_principles")
    if not isinstance(principles, Mapping):
        raise ValueError("field_principles must be present")
    missing_principles = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in principles]
    if missing_principles:
        raise ValueError(f"missing field principles: {missing_principles}")
    for field in (
        "strongest_candidate_beats_frozen",
        "refreeze_package_reemitted_for_winner",
        "adversarial_verify_clean",
        "north_star_unmodified_assert",
        "frozen_headline_unchanged_assert",
    ):
        if type(artifact.get(field)) is not bool:
            raise ValueError(f"{field} must be a bare boolean")
    checklist = artifact.get("operator_checklist")
    if not isinstance(checklist, list):
        raise ValueError("operator_checklist must be a list")
    if artifact.get("refreeze_package_reemitted_for_winner"):
        if not checklist or not all(str(step).startswith("OPERATOR-ACTION:") for step in checklist):
            raise ValueError("operator_checklist must contain OPERATOR-ACTION steps")
        if artifact.get("north_star_unmodified_assert") is not True:
            raise ValueError("north_star_unmodified_assert must remain true for a package")
        if artifact.get("frozen_headline_unchanged_assert") is not True:
            raise ValueError("frozen_headline_unchanged_assert must remain true for a package")
    elif any(str(step).startswith("OPERATOR-ACTION:") for step in checklist):
        raise ValueError("operator_checklist cannot contain OPERATOR-ACTION steps without a package")
    if verdict == BLOCKED_VERDICT:
        return
    if int(artifact.get("n_seeds", 0)) < 5:
        raise ValueError("n_seeds must be at least 5")
    if int(artifact.get("n_examples", 0)) < exp3680.DEFAULT_N_EXAMPLES:
        raise ValueError(f"n_examples must be at least {exp3680.DEFAULT_N_EXAMPLES}")
    for field in (
        "dependency_aware_auroc",
        "external_comparator_auroc",
        "fusion_auroc",
        "carnot_current_auroc",
        "frozen_headline_auroc",
    ):
        _validate_auroc_field(artifact, field)
    _validate_strongest_candidate_auroc(artifact)
    _validate_candidate_distinctness(artifact)
    _validate_delta_ci(artifact.get("winner_vs_runnerup_delta_ci"), "winner_vs_runnerup_delta_ci")
    _validate_delta_ci(artifact.get("winner_vs_frozen_delta_ci"), "winner_vs_frozen_delta_ci")
    if artifact.get("refreeze_package_reemitted_for_winner") and artifact.get(
        "adversarial_verify_clean"
    ) is not True:
        raise ValueError("clean re-freeze package requires adversarial_verify_clean")


def architecture_aurocs(rows: Sequence[CandidateScorePanel]) -> dict[str, float]:
    """Return architecture-only AUROCs for all three candidates."""

    labels = np.concatenate([np.asarray(row.labels, dtype=np.int64) for row in rows])
    return {
        "dependency_aware": _round_metric(
            exp3644.tie_aware_auroc(
                labels,
                np.concatenate(
                    [np.asarray(row.dependency_architecture_scores, dtype=np.float64) for row in rows]
                ),
            )
        ),
        "external": _round_metric(
            exp3644.tie_aware_auroc(
                labels,
                np.concatenate(
                    [np.asarray(row.external_architecture_scores, dtype=np.float64) for row in rows]
                ),
            )
        ),
        "fusion": _round_metric(
            exp3644.tie_aware_auroc(
                labels,
                np.concatenate(
                    [np.asarray(row.fusion_architecture_scores, dtype=np.float64) for row in rows]
                ),
            )
        ),
    }


def per_seed_results(rows: Sequence[CandidateScorePanel]) -> list[dict[str, Any]]:
    """Return per-seed production and architecture-only AUROC diagnostics."""

    results: list[dict[str, Any]] = []
    for row in rows:
        labels = np.asarray(row.labels, dtype=np.int64)
        dep = exp3644.tie_aware_auroc(labels, row.dependency_scores)
        ext = exp3644.tie_aware_auroc(labels, row.external_scores)
        fusion = exp3644.tie_aware_auroc(labels, row.fusion_scores)
        dep_arch = exp3644.tie_aware_auroc(labels, row.dependency_architecture_scores)
        ext_arch = exp3644.tie_aware_auroc(labels, row.external_architecture_scores)
        fusion_arch = exp3644.tie_aware_auroc(labels, row.fusion_architecture_scores)
        results.append(
            {
                "seed": int(row.seed),
                "n_examples": int(len(labels)),
                "dependency_aware_auroc": _round_metric(dep),
                "external_comparator_auroc": _round_metric(ext),
                "fusion_auroc": _round_metric(fusion),
                "carnot_current_auroc": _round_metric(
                    exp3644.tie_aware_auroc(labels, row.carnot_current_scores)
                ),
                "architecture_dependency_aware_auroc": _round_metric(dep_arch),
                "architecture_external_comparator_auroc": _round_metric(ext_arch),
                "architecture_fusion_auroc": _round_metric(fusion_arch),
                "learning_contribution_dependency_aware": _round_metric(dep - dep_arch),
                "learning_contribution_external": _round_metric(ext - ext_arch),
                "learning_contribution_fusion": _round_metric(fusion - fusion_arch),
                "fusion_alpha_mean": row.fusion_alpha_mean,
                "subset_sha256": row.subset_sha256,
            }
        )
    return results


def reproducibility_checksum(
    *,
    panels: Sequence[CandidateScorePanel],
    score_vectors: Mapping[str, np.ndarray],
    random_seed: int,
    bootstrap_seeds: Sequence[int],
    gate_after: Mapping[str, Any],
) -> str:
    """Hash labels, score vectors, seed config, and frozen-gate evidence."""

    digest = hashlib.sha256()
    for panel in panels:
        digest.update(str(int(panel.seed)).encode("ascii"))
        digest.update(np.ascontiguousarray(panel.labels, dtype=np.int64).tobytes())
    for name in sorted(score_vectors):
        digest.update(name.encode("utf-8"))
        digest.update(np.ascontiguousarray(score_vectors[name], dtype=np.float64).tobytes())
    digest.update(str(int(random_seed)).encode("ascii"))
    digest.update(json.dumps([int(seed) for seed in bootstrap_seeds]).encode("ascii"))
    digest.update(json.dumps(gate_after, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    return digest.hexdigest()


def vector_checksum(values: Sequence[float] | np.ndarray) -> str:
    """Stable checksum for one full score vector."""

    return hashlib.sha256(np.ascontiguousarray(values, dtype=np.float64).tobytes()).hexdigest()


def publication_gate_reads_frozen_0_9131(gate: Mapping[str, Any]) -> bool:
    """True when the gate evidence still names the frozen headline."""

    blob = json.dumps(gate, sort_keys=True)
    return "0.9131" in blob and "0.9253" not in blob and "0.9287" not in blob


def _paired_delta_with_delong(
    labels: np.ndarray,
    first_scores: np.ndarray,
    second_scores: np.ndarray,
    *,
    seeds: Sequence[int],
    n_bootstrap: int,
    first: str,
    second: str,
) -> dict[str, Any]:
    delta = exp3667.paired_delta_ci(
        labels,
        first_scores,
        second_scores,
        seeds=seeds,
        n_bootstrap=n_bootstrap,
    )
    delong = exp3667.paired_delong_test(labels, first_scores, second_scores)
    return {
        **dict(delta),
        "winner": first,
        "comparison": second,
        "delong_p": delong["p_value"],
        "delong": {
            "method": delong["method"],
            "auc_first": delong["auc_dependency_aware_proper"],
            "auc_second": delong["auc_carnot_current"],
            "auc_difference": delong["auc_difference"],
            "standard_error": delong["standard_error"],
            "z_value": delong["z_value"],
            "p_value": delong["p_value"],
        },
    }


def _best_fusion_alpha(labels: np.ndarray, dependency: np.ndarray, external: np.ndarray) -> float:
    best_alpha = 0.5
    best_auc = -math.inf
    for alpha in np.linspace(0.05, 0.95, 19):
        scores = alpha * dependency + (1.0 - alpha) * external
        auc = exp3644.tie_aware_auroc(labels, scores)
        if auc > best_auc + 1e-12 or (
            abs(auc - best_auc) <= 1e-12 and abs(alpha - 0.5) < abs(best_alpha - 0.5)
        ):
            best_alpha = float(alpha)
            best_auc = float(auc)
    return best_alpha


def _minmax01(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    low = float(np.min(arr))
    high = float(np.max(arr))
    if high - low <= 1e-12:
        return np.full_like(arr, 0.5, dtype=np.float64)
    return (arr - low) / (high - low)


def _stratified_fold_ids(labels: np.ndarray, *, n_folds: int, random_seed: int) -> np.ndarray:
    min_class = min(int(np.sum(labels == 0)), int(np.sum(labels == 1)))
    folds = max(2, min(int(n_folds), min_class))
    rng = np.random.default_rng(int(random_seed))
    fold_ids = np.zeros(len(labels), dtype=np.int64)
    for label in (0, 1):
        idx = np.where(labels == label)[0]
        rng.shuffle(idx)
        for offset, row_idx in enumerate(idx):
            fold_ids[row_idx] = offset % folds
    return fold_ids


def _assert_score_lengths(labels: np.ndarray, score_vectors: Mapping[str, np.ndarray]) -> None:
    for name, values in score_vectors.items():
        if len(values) != len(labels):
            raise ValueError(f"{name} scores and labels must have the same length")
        if not np.isfinite(values).all():
            raise ValueError(f"{name} scores must be finite")
    _require_binary_labels(labels)


def _require_binary_labels(labels: np.ndarray) -> None:
    if set(np.asarray(labels, dtype=np.int64).tolist()) != {0, 1}:
        raise ValueError("labels must contain both binary classes 0 and 1")


def _validate_candidate_distinctness(artifact: Mapping[str, Any]) -> None:
    candidate_values = [
        ("dependency_aware_auroc", artifact.get("dependency_aware_auroc")),
        ("external_comparator_auroc", artifact.get("external_comparator_auroc")),
        ("fusion_auroc", artifact.get("fusion_auroc")),
        ("frozen_headline_auroc", artifact.get("frozen_headline_auroc")),
    ]
    seen: list[tuple[str, float]] = []
    for field, value in candidate_values:
        numeric = float(value)
        for prior_field, prior_value in seen:
            if numeric == prior_value:
                raise ValueError(f"distinct AUROC fields cannot be bit-identical: {prior_field}, {field}")
        seen.append((field, numeric))
    checksums = dict(artifact.get("score_vector_checksums") or {})
    for left, right in (
        ("dependency_aware", "external_comparator"),
        ("dependency_aware", "fusion"),
        ("external_comparator", "fusion"),
    ):
        if checksums.get(left) == checksums.get(right):
            raise ValueError(f"distinct AUROC score vectors cannot be copied: {left}, {right}")


def _validate_strongest_candidate_auroc(artifact: Mapping[str, Any]) -> None:
    value = artifact.get("strongest_candidate_auroc")
    if not isinstance(value, Mapping):
        raise ValueError("strongest_candidate_auroc must be an object")
    candidate = str(value.get("candidate"))
    if candidate not in CANDIDATE_FIELDS:
        raise ValueError("strongest_candidate_auroc candidate is unsupported")
    source_field = str(value.get("source_field"))
    if source_field != CANDIDATE_FIELDS[candidate]:
        raise ValueError("strongest_candidate_auroc source_field does not match candidate")
    numeric = value.get("value")
    if not _is_finite_number(numeric) or float(numeric) != float(artifact[source_field]):
        raise ValueError("strongest_candidate_auroc value must match its source field")


def _validate_delta_ci(value: Any, field: str) -> None:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be an object")
    point = value.get("point")
    ci = value.get("ci95")
    if not _is_finite_number(point) or not isinstance(ci, list) or len(ci) != 2:
        raise ValueError(f"{field} must include point and ci95")
    if not all(_is_finite_number(item) for item in ci):
        raise ValueError(f"{field} bounds must be finite")
    if not float(ci[0]) <= float(point) <= float(ci[1]):
        raise ValueError(f"{field} must contain its point estimate")
    delong_p = value.get("delong_p")
    if not _is_finite_number(delong_p) or not 0.0 <= float(delong_p) <= 1.0:
        raise ValueError(f"{field} must include a DeLong p in [0, 1]")


def _validate_auroc_field(artifact: Mapping[str, Any], field: str) -> None:
    value = artifact.get(field)
    if not _is_finite_number(value) or not 0.0 <= float(value) <= 1.0:
        raise ValueError(f"{field} must be finite and in [0, 1]")
    if field in CANDIDATE_FIELDS.values() and float(value) >= 0.99:
        raise ValueError(f"{field} triggers the AUROC>=0.99 leak guard")


def _is_finite_number(value: Any) -> bool:
    return not isinstance(value, bool) and isinstance(value, int | float) and math.isfinite(float(value))


def _round_metric(value: float | int | None, digits: int = 6) -> float | None:
    if value is None:
        return None
    return round(float(value), digits)


def _duration(started_s: float, now_s: float | None) -> float:
    end = time.time() if now_s is None else float(now_s)
    return max(1.0, end - float(started_s))


def _sha256_file(path: Path) -> str:
    if not path.exists():
        return "missing"
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    """CLI entrypoint."""

    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    output = write_artifact(REPO_ROOT)
    artifact = json.loads(output.read_text(encoding="utf-8"))
    print(artifact["honest_verdict"])
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
