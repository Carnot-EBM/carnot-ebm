"""Exp 4246 second-corpus replication for the oracle-distinct code selector.

Spec refs: REQ-VERIFY-4246, SCENARIO-VERIFY-4246,
SCENARIO-VERIFY-4246-BLOCKED.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from carnot.reporting.oracle_distinct_code_beats_vote_4233 import (
    BOOTSTRAP_RESAMPLES,
    FEATURE_NAMES,
    CandidatePool,
    FeatureRow,
    PoolSpec,
    _clean_adversarial_report,
    _load_rows_for_spec,
    _model_specs as _base_model_specs,
    _round_metric,
    _run_adversarial_verify,
    _sha256_file,
    build_feature_matrix,
    measure_gate,
    train_oof_predictor,
)


RANDOM_SEED = 4246
OUTPUT_REL = Path("results/experiment_4246_code_oracle_distinct_replication.json")
EXP4233_REL = Path("results/experiment_4233_oracle_distinct_code_beats_vote.json")
INFERENCE_SUBSTRATE = "cached_second_code_candidates_oracle_distinct_sklearn_cpu"
BLOCKED_VERDICT = "blocked_code_second_corpus_missing"

SPEC_REFS = [
    "REQ-VERIFY-4246",
    "SCENARIO-VERIFY-4246",
    "SCENARIO-VERIFY-4246-BLOCKED",
]

FIELD_PRINCIPLES = {
    "honest_verdict": (
        "Terminal-prefixed. A replication win, a corpus-specific tie, or an "
        "honest no-pool/no-headroom is COMPLETE -- each calibrates how robust "
        "the .392 code win is."
    ),
    "code_replication_beats_vote": (
        "BARE bool: predictor@1 - vote@1 CI95 excludes 0 AND delta>0 AND "
        "headroom on the SECOND corpus -- robustness of the oracle-distinct "
        "code win; NOT the circular execution result."
    ),
    "code_predictor_minus_vote_delta": (
        "predictor@1 - vote@1 on the 2nd held-out code corpus -- the "
        "oracle-distinct lift; compare to .392's +0.03125."
    ),
    "code_predictor_minus_vote_ci95": (
        "Task-level bootstrap CI95 of the 2nd-corpus delta -- excluding 0 "
        "confirms the win is not corpus-specific noise."
    ),
    "oracle_at_k": (
        "2nd-corpus positive-control ceiling (any candidate passes) -- if "
        "~=vote the null is uninformative."
    ),
    "held_out_task_n": "BARE int: the 2nd-corpus gate's N.",
    "replication_read": (
        "replicates / corpus_specific / no_headroom -- the robustness read on "
        "the .392 code oracle-distinct win, anchoring the .393 headline "
        "alongside the ARC A3 result."
    ),
    "verifier_is_oracle": (
        "BARE bool=false -- the predictor scores code WITHOUT executing tests "
        "at inference (hidden-test label is the training target only); keeps "
        "the result oracle-distinct, not circular."
    ),
    "model_specs": (
        "The pass-predictor architecture + oracle-distinct code feature set + "
        "calibrated loss + the 2nd corpus id; required methodology."
    ),
    "random_seed": "Determinism precondition; fold split + init seeded.",
    "reproducibility_checksum": (
        "Hash of the 2nd code candidate pool + fold split; lets a third party re-run."
    ),
}

REQUIRED_FIELDS = (
    "honest_verdict",
    "code_replication_beats_vote",
    "code_predictor_minus_vote_delta",
    "code_predictor_minus_vote_ci95",
    "oracle_at_k",
    "held_out_task_n",
    "replication_read",
    "verifier_is_oracle",
    "model_specs",
    "random_seed",
    "reproducibility_checksum",
    "field_principles",
    "spec_refs",
    "acceptance_gate",
)

DEFAULT_POOL_SPECS = (
    PoolSpec("exp2837_mbpp_ensemble_eval", (Path("results/experiment_2837_mbpp_ensemble_eval.json"),)),
    PoolSpec(
        "exp2838_2830_humaneval_disjoint",
        (
            Path("results/experiment_2838_humaneval_full_ensemble_eval.json"),
            Path("results/experiment_2830_humaneval_full_ensemble_eval.json"),
        ),
    ),
    PoolSpec("exp1607_dsl_humaneval", (Path("results/experiment_1607_dsl_humaneval.json"),)),
    PoolSpec(
        "verifier_reward_3arm_lora_rft_91b7244bb09edd32",
        (
            Path(
                "results/verifier_reward_3arm_lora_rft/"
                "code_verifier_reward_lora_rft_91b7244bb09edd32/corpora/arm_A.jsonl"
            ),
            Path(
                "results/verifier_reward_3arm_lora_rft/"
                "code_verifier_reward_lora_rft_91b7244bb09edd32/corpora/arm_B.jsonl"
            ),
            Path(
                "results/verifier_reward_3arm_lora_rft/"
                "code_verifier_reward_lora_rft_91b7244bb09edd32/corpora/arm_C.jsonl"
            ),
        ),
    ),
)


class BlockedSecondCorpus(RuntimeError):
    """Expected precondition failure for a missing distinct second corpus."""

    def __init__(
        self,
        reason: str,
        attempted_sources: list[dict[str, Any]],
        exp4233_source: dict[str, Any],
    ) -> None:
        super().__init__(reason)
        self.reason = reason
        self.attempted_sources = attempted_sources
        self.exp4233_source = exp4233_source


def _resolve_path(repo_root: Path, path_text: str) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else repo_root / path


def _source_hashes(paths: list[Path]) -> dict[str, str]:
    return {str(path): _sha256_file(path) for path in paths if path.exists()}


def _exp4233_source(repo_root: Path, exp4233_rel: Path = EXP4233_REL) -> dict[str, Any]:
    path = exp4233_rel if exp4233_rel.is_absolute() else repo_root / exp4233_rel
    if not path.exists():
        return {
            "artifact_path": str(path),
            "artifact_exists": False,
            "source_id": "",
            "source_paths": [],
            "source_sha256": {},
            "source_hash_values": [],
        }
    payload = json.loads(path.read_text(encoding="utf-8"))
    pool = payload.get("candidate_pool") if isinstance(payload, dict) else {}
    source_paths_raw = pool.get("source_paths") if isinstance(pool, dict) else []
    source_paths = [
        _resolve_path(repo_root, item)
        for item in source_paths_raw
        if isinstance(item, str) and item
    ]
    source_sha = _source_hashes(source_paths)
    return {
        "artifact_path": str(path),
        "artifact_exists": True,
        "source_id": pool.get("source_id", "") if isinstance(pool, dict) else "",
        "source_paths": [str(item) for item in source_paths],
        "source_sha256": source_sha,
        "source_hash_values": sorted(source_sha.values()),
    }


def _spec_existing_paths(repo_root: Path, spec: PoolSpec) -> list[Path]:
    paths = []
    for rel_path in spec.paths:
        path = rel_path if rel_path.is_absolute() else repo_root / rel_path
        if path.exists():
            paths.append(path)
    return paths


def _is_viable_report(report: dict[str, Any]) -> bool:
    return bool(
        report.get("viable_candidate_rows", 0) > 0
        and report.get("task_n", 0) > 0
        and report.get("positive_n", 0) > 0
        and report.get("has_both_labels") is True
    )


def load_second_candidate_pool(
    repo_root: Path | str,
    pool_specs: tuple[PoolSpec, ...] = DEFAULT_POOL_SPECS,
    *,
    exp4233_rel: Path = EXP4233_REL,
) -> CandidatePool:
    root = Path(repo_root)
    exp4233_source = _exp4233_source(root, exp4233_rel)
    exp4233_hashes = set(exp4233_source["source_hash_values"])
    attempted_sources: list[dict[str, Any]] = []
    if not exp4233_hashes:
        raise BlockedSecondCorpus(BLOCKED_VERDICT, attempted_sources, exp4233_source)

    for spec in pool_specs:
        rows, report = _load_rows_for_spec(root, spec)
        source_paths = _spec_existing_paths(root, spec)
        source_sha = _source_hashes(source_paths)
        overlap = sorted(set(source_sha.values()) & exp4233_hashes)
        distinct = bool(source_sha) and not overlap
        report.update(
            {
                "source_sha256": source_sha,
                "distinct_from_exp4233": distinct,
                "exp4233_source_overlap_sha256": overlap,
            }
        )
        if _is_viable_report(report) and not distinct:
            report["skip_reason"] = "candidate_source_not_distinct_from_exp4233"
        elif not _is_viable_report(report):
            report["skip_reason"] = "missing_viable_multicandidate_hidden_label_rows"
        attempted_sources.append(report)
        if not _is_viable_report(report) or not distinct:
            continue
        return CandidatePool(
            source_id=spec.source_id,
            rows=rows,
            source_paths=source_paths,
            source_sha256=source_sha,
            task_n=len({row.task_id for row in rows}),
            candidate_n=len(rows),
            positive_n=sum(1 for row in rows if row.passes_hidden_tests),
            pass_rate=sum(1 for row in rows if row.passes_hidden_tests) / float(len(rows)),
            attempted_sources=attempted_sources,
            vote_signature_source="normalized_code_text_signature",
        )
    raise BlockedSecondCorpus(BLOCKED_VERDICT, attempted_sources, exp4233_source)


def _replication_read(metrics: dict[str, Any]) -> tuple[str, str]:
    if not metrics.get("headroom_exists"):
        return "no_headroom", "complete: code_oracle_distinct_replication_no_headroom"
    if metrics.get("code_oracle_distinct_beats_vote") is True:
        return "replicates", "complete: code_oracle_distinct_replication_replicates"
    return "corpus_specific", "complete: code_oracle_distinct_replication_corpus_specific"


def _model_specs(feature_names: list[str], second_corpus_id: str) -> dict[str, Any]:
    specs = _base_model_specs(feature_names)
    specs["second_corpus_id"] = second_corpus_id
    specs["source_distinctness_gate"] = "source file sha256 must not overlap exp4233 source corpus"
    return specs


def reproducibility_checksum(
    pool: CandidatePool,
    folds: list[list[str]],
    feature_names: list[str],
    random_seed: int,
    exp4233_source: dict[str, Any],
) -> str:
    payload = {
        "candidate_ids": [row.candidate_id for row in pool.rows],
        "folds": folds,
        "feature_names": feature_names,
        "random_seed": int(random_seed),
        "second_source_id": pool.source_id,
        "second_source_sha256": pool.source_sha256,
        "exp4233_source_hash_values": exp4233_source.get("source_hash_values", []),
        "vote_signature_source": pool.vote_signature_source,
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _blocked_checksum(
    reason: str,
    random_seed: int,
    attempted_sources: list[dict[str, Any]],
    exp4233_source: dict[str, Any],
) -> str:
    payload = {
        "reason": reason,
        "random_seed": int(random_seed),
        "attempted_sources": attempted_sources,
        "exp4233_source": exp4233_source,
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _blocked_artifact(
    attempted_sources: list[dict[str, Any]],
    exp4233_source: dict[str, Any],
    *,
    random_seed: int,
    duration_s: float,
) -> dict[str, Any]:
    return {
        "experiment": "experiment_4246_code_oracle_distinct_replication",
        "schema": "carnot.code_oracle_distinct_replication_4246.v1",
        "status": "complete",
        "honest_verdict": BLOCKED_VERDICT,
        "code_replication_beats_vote": False,
        "code_predictor_minus_vote_delta": 0.0,
        "code_predictor_minus_vote_ci95": [0.0, 0.0],
        "oracle_at_k": 0.0,
        "held_out_task_n": 0,
        "replication_read": BLOCKED_VERDICT,
        "verifier_is_oracle": False,
        "model_specs": _model_specs(list(FEATURE_NAMES), ""),
        "random_seed": int(random_seed),
        "reproducibility_checksum": _blocked_checksum(
            BLOCKED_VERDICT,
            random_seed,
            attempted_sources,
            exp4233_source,
        ),
        "field_principles": FIELD_PRINCIPLES,
        "spec_refs": SPEC_REFS,
        "acceptance_gate": True,
        "off_fold_auroc": 0.0,
        "pass_rates": {
            "predictor_at_1": 0.0,
            "vote_at_1": 0.0,
            "matched_control_at_1": 0.0,
        },
        "matched_control_delta": 0.0,
        "oracle_minus_vote": 0.0,
        "headroom_exists": False,
        "ci95_excludes_zero": False,
        "bootstrap_resamples": 0,
        "candidate_pool": {
            "source_id": "",
            "candidate_n": 0,
            "task_n": 0,
            "positive_n": 0,
            "pass_rate": 0.0,
        },
        "attempted_candidate_sources": attempted_sources,
        "exp4233_source": exp4233_source,
        "vote_signature_source": "normalized_code_text_signature",
        "task_rows": [],
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": round(duration_s, 6),
        "methodology_note": (
            "Blocked before training because no cached second code candidate pool "
            "was both hidden-label viable and source-distinct from Exp 4233."
        ),
        "adversarial_verify": {"status": "pending"},
    }


def _complete_artifact(
    pool: CandidatePool,
    metrics: dict[str, Any],
    *,
    exp4233_source: dict[str, Any],
    off_fold_auroc: float,
    folds: list[list[str]],
    feature_names: list[str],
    random_seed: int,
    duration_s: float,
) -> dict[str, Any]:
    replication_read, honest_verdict = _replication_read(metrics)
    return {
        "experiment": "experiment_4246_code_oracle_distinct_replication",
        "schema": "carnot.code_oracle_distinct_replication_4246.v1",
        "status": "complete",
        "honest_verdict": honest_verdict,
        "code_replication_beats_vote": bool(metrics["code_oracle_distinct_beats_vote"]),
        "code_predictor_minus_vote_delta": metrics["code_predictor_minus_vote_delta"],
        "code_predictor_minus_vote_ci95": metrics["code_predictor_minus_vote_ci95"],
        "oracle_at_k": metrics["oracle_at_k"],
        "held_out_task_n": metrics["held_out_task_n"],
        "replication_read": replication_read,
        "verifier_is_oracle": False,
        "model_specs": _model_specs(feature_names, pool.source_id),
        "random_seed": int(random_seed),
        "reproducibility_checksum": reproducibility_checksum(
            pool,
            folds,
            feature_names,
            random_seed,
            exp4233_source,
        ),
        "field_principles": FIELD_PRINCIPLES,
        "spec_refs": SPEC_REFS,
        "acceptance_gate": True,
        "off_fold_auroc": _round_metric(off_fold_auroc),
        "pass_rates": metrics["pass_rates"],
        "matched_control_delta": metrics["matched_control_delta"],
        "oracle_minus_vote": metrics["oracle_minus_vote"],
        "headroom_exists": metrics["headroom_exists"],
        "ci95_excludes_zero": metrics["ci95_excludes_zero"],
        "bootstrap_resamples": metrics["bootstrap_resamples"],
        "candidate_pool": {
            "source_id": pool.source_id,
            "candidate_n": pool.candidate_n,
            "task_n": pool.task_n,
            "positive_n": pool.positive_n,
            "pass_rate": _round_metric(pool.pass_rate),
            "source_paths": [str(path) for path in pool.source_paths],
        },
        "attempted_candidate_sources": pool.attempted_sources,
        "exp4233_source": exp4233_source,
        "vote_signature_source": pool.vote_signature_source,
        "fold_task_ids": folds,
        "task_rows": metrics["task_rows"],
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": round(duration_s, 6),
        "methodology_note": (
            "Hidden-test pass labels are supervised targets only. Candidate scoring "
            "uses code text and cross-candidate signature features without executing "
            "candidate code or tests at inference."
        ),
        "adversarial_verify": {"status": "pending"},
    }


def validate_artifact(artifact: dict[str, Any]) -> None:
    missing = [field for field in REQUIRED_FIELDS if field not in artifact]
    if missing:
        raise ValueError(f"missing required fields: {missing}")
    verdict = artifact["honest_verdict"]
    if not isinstance(verdict, str) or not (
        verdict.startswith("complete:") or verdict == BLOCKED_VERDICT
    ):
        raise ValueError("honest_verdict must be terminal-prefixed or a blocked verdict")
    if type(artifact["code_replication_beats_vote"]) is not bool:
        raise ValueError("code_replication_beats_vote must be a bare bool")
    for field in ("code_predictor_minus_vote_delta", "oracle_at_k"):
        if isinstance(artifact[field], bool) or not isinstance(artifact[field], (int, float)):
            raise ValueError(f"{field} must be a bare float")
    ci95 = artifact["code_predictor_minus_vote_ci95"]
    if (
        not isinstance(ci95, list)
        or len(ci95) != 2
        or any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in ci95)
    ):
        raise ValueError("code_predictor_minus_vote_ci95 must be a two-number ci95")
    if type(artifact["held_out_task_n"]) is not int:
        raise ValueError("held_out_task_n must be a bare int")
    if artifact["replication_read"] not in {
        "replicates",
        "corpus_specific",
        "no_headroom",
        BLOCKED_VERDICT,
    }:
        raise ValueError("replication_read has an unknown value")
    if artifact["code_replication_beats_vote"] and artifact["replication_read"] != "replicates":
        raise ValueError("winning artifacts must set replication_read=replicates")
    if artifact["verifier_is_oracle"] is not False:
        raise ValueError("verifier_is_oracle must be the bare bool false")
    if type(artifact["random_seed"]) is not int:
        raise ValueError("random_seed must be a bare int")
    if not isinstance(artifact["model_specs"], dict):
        raise ValueError("model_specs must be present")
    if artifact["model_specs"].get("verifier_is_oracle") is not False:
        raise ValueError("model_specs must preserve verifier_is_oracle=false")
    if artifact["field_principles"] != FIELD_PRINCIPLES:
        raise ValueError("field_principles do not match REQ-VERIFY-4246")
    if artifact["spec_refs"] != SPEC_REFS:
        raise ValueError("spec_refs do not match REQ-VERIFY-4246")


def run(
    repo_root: Path | str = Path("."),
    *,
    random_seed: int = RANDOM_SEED,
    bootstrap_resamples: int = BOOTSTRAP_RESAMPLES,
    pool_specs: tuple[PoolSpec, ...] = DEFAULT_POOL_SPECS,
    exp4233_rel: Path = EXP4233_REL,
    adversarial_runner: Callable[[Path], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    start = time.perf_counter()
    root = Path(repo_root)
    output_path = root / OUTPUT_REL
    try:
        pool = load_second_candidate_pool(root, pool_specs, exp4233_rel=exp4233_rel)
        exp4233_source = _exp4233_source(root, exp4233_rel)
        feature_rows, feature_names = build_feature_matrix(pool.rows)
        scored_rows, folds, off_fold_auroc = train_oof_predictor(
            feature_rows,
            feature_names,
            random_seed=random_seed,
        )
        metrics = measure_gate(
            scored_rows,
            random_seed=random_seed,
            bootstrap_resamples=bootstrap_resamples,
        )
        artifact = _complete_artifact(
            pool,
            metrics,
            exp4233_source=exp4233_source,
            off_fold_auroc=off_fold_auroc,
            folds=folds,
            feature_names=feature_names,
            random_seed=random_seed,
            duration_s=time.perf_counter() - start,
        )
    except BlockedSecondCorpus as exc:
        artifact = _blocked_artifact(
            exc.attempted_sources,
            exc.exp4233_source,
            random_seed=random_seed,
            duration_s=time.perf_counter() - start,
        )
    validate_artifact(artifact)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    raw_report = (
        adversarial_runner(output_path)
        if adversarial_runner is not None
        else _run_adversarial_verify(root, output_path)
    )
    artifact["adversarial_verify"] = _clean_adversarial_report(raw_report)
    validate_artifact(artifact)
    output_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return artifact
