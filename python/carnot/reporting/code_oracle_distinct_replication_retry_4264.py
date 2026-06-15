"""Exp 4264 retry for the oracle-distinct code replication gate.

Spec refs: REQ-VERIFY-4264, SCENARIO-VERIFY-4264.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any

from carnot.reporting.code_oracle_distinct_replication_4246 import (
    DEFAULT_POOL_SPECS as EXP4246_POOL_SPECS,
    _exp4233_source,
    _is_viable_report,
    _source_hashes,
    _spec_existing_paths,
    reproducibility_checksum as _pool_reproducibility_checksum,
)
from carnot.reporting.oracle_distinct_code_beats_vote_4233 import (
    BOOTSTRAP_RESAMPLES,
    FEATURE_NAMES,
    CandidatePool,
    CodeCandidate,
    PoolSpec,
    _clean_adversarial_report,
    _load_rows_for_spec as _load_flat_rows_for_spec,
    _model_specs as _base_model_specs,
    _round_metric,
    _run_adversarial_verify,
    build_feature_matrix,
    measure_gate,
    normalized_code_signature,
    train_oof_predictor,
)


RANDOM_SEED = 4264
OUTPUT_REL = Path("results/experiment_4264_code_oracle_distinct_replication_retry.json")
EXP4233_REL = Path("results/experiment_4233_oracle_distinct_code_beats_vote.json")
INFERENCE_SUBSTRATE = "deterministic_verifier_plus_replay"
BLOCKED_MODEL_VERDICT = "blocked_code_gen_model_not_cached"
RETIRED_VERDICT = "complete: code_replication_retired"

SPEC_REFS = [
    "REQ-VERIFY-4264",
    "SCENARIO-VERIFY-4264",
]

FIELD_PRINCIPLES = {
    "honest_verdict": (
        "Terminal-prefixed. A replication win, a corpus-specific tie, a "
        "no-headroom, AND a retire are ALL COMPLETE -- each calibrates the "
        "robustness of the .392 code win."
    ),
    "code_replication_beats_vote": (
        "BARE bool: predictor@1 - vote@1 CI95-excl-0 AND headroom on the "
        "SECOND corpus -- robustness of the oracle-distinct code win; NOT the "
        "circular execution result."
    ),
    "code_predictor_minus_vote_delta": (
        "predictor@1 - vote@1 on the 2nd corpus -- compare to .392's +0.03125."
    ),
    "code_predictor_minus_vote_ci95": (
        "Task-level bootstrap CI95 of the 2nd-corpus delta -- excluding 0 "
        "confirms the win is not corpus-specific noise."
    ),
    "oracle_at_k": (
        "2nd-corpus positive-control ceiling (any candidate passes) -- if ~=vote "
        "the null is uninformative."
    ),
    "held_out_task_n": "BARE int: the 2nd-corpus gate's N.",
    "replication_read": (
        "replicates / corpus_specific / no_headroom -- the robustness read "
        "anchoring the .394 headline alongside the ARC result."
    ),
    "code_replication_retired": (
        "BARE bool: true iff a fresh distinct pool cannot be built in-window -> "
        "the .392 single-corpus +3.1pp stands (no 3rd block)."
    ),
    "verifier_is_oracle": (
        "BARE bool=false -- the predictor scores code WITHOUT executing tests at "
        "inference (label is the training target only)."
    ),
    "random_seed": "Determinism precondition; gen + fold split seeded.",
    "reproducibility_checksum": (
        "Hash of the 2nd code pool + fold split; lets a third party re-run."
    ),
    "model_specs": (
        "The gen model (SOTA GGUF) + pass-predictor + oracle-distinct feature "
        "set + calibrated loss; required methodology."
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
    "code_replication_retired",
    "verifier_is_oracle",
    "model_specs",
    "random_seed",
    "reproducibility_checksum",
    "field_principles",
    "spec_refs",
    "acceptance_gate",
)

DEFAULT_POOL_SPECS = (
    PoolSpec(
        "offarc_power_sync_gemma12b_evalplus_k5_checkpoint",
        (Path("results/offarc_power_sync_gemma12b_evalplus_k5.checkpoint.json"),),
    ),
) + EXP4246_POOL_SPECS

SOTA_GGUF_CANDIDATES = (
    (
        "unsloth/gemma-4-12B-it-GGUF",
        Path("models--unsloth--gemma-4-12B-it-GGUF"),
    ),
    (
        "unsloth/Qwen3.6-35B-A3B-GGUF",
        Path("models--unsloth--Qwen3.6-35B-A3B-GGUF"),
    ),
    (
        "unsloth/gemma-4-31B-it-GGUF",
        Path("models--unsloth--gemma-4-31B-it-GGUF"),
    ),
)


class MissingDistinctPool(RuntimeError):
    """Expected retry precondition failure for no viable distinct cached pool."""

    def __init__(
        self,
        attempted_sources: list[dict[str, Any]],
        exp4233_source: dict[str, Any],
    ) -> None:
        super().__init__("missing distinct second code corpus")
        self.attempted_sources = attempted_sources
        self.exp4233_source = exp4233_source


def _as_hidden_label(row: dict[str, Any]) -> bool | None:
    values = row.get("hidden_passes")
    if isinstance(values, list):
        return bool(values) and all(value is True for value in values)
    value = row.get("hidden_pass")
    if isinstance(value, bool):
        return value
    return None


def _candidate_from_evalplus_row(
    source_id: str,
    source_path: Path,
    task_id: str,
    row: dict[str, Any],
    row_index: int,
) -> CodeCandidate | None:
    code = row.get("code")
    label = _as_hidden_label(row)
    if not isinstance(code, str) or not code or label is None:
        return None
    draw_index = row.get("draw_index", row_index)
    if isinstance(draw_index, bool) or not isinstance(draw_index, int):
        draw_index = row_index
    signature = normalized_code_signature(code)
    raw_id = {
        "source_id": source_id,
        "source_schema": "evalplus_checkpoint",
        "task_id": task_id,
        "draw_index": draw_index,
        "row_index": row_index,
        "signature": signature,
    }
    candidate_hash = hashlib.sha256(
        json.dumps(raw_id, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return CodeCandidate(
        source_id=source_id,
        source_path=str(source_path),
        row_index=row_index,
        task_id=task_id,
        candidate_id=f"{source_id}:{task_id}:{draw_index}:{candidate_hash[:12]}",
        candidate_index=draw_index,
        code=code,
        prompt="",
        passes_hidden_tests=label,
        vote_signature=signature,
    )


def _load_evalplus_checkpoint_rows(
    repo_root: Path,
    spec: PoolSpec,
) -> tuple[list[CodeCandidate], dict[str, Any]]:
    rows: list[CodeCandidate] = []
    path_reports: list[dict[str, Any]] = []
    source_models: list[str] = []
    for rel_path in spec.paths:
        path = rel_path if rel_path.is_absolute() else repo_root / rel_path
        if not path.exists():
            path_reports.append({"path": str(rel_path), "exists": False, "candidate_rows": 0})
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            path_reports.append(
                {
                    "path": str(rel_path),
                    "exists": True,
                    "candidate_rows": 0,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            continue
        evaluations = payload.get("evaluations_by_task") if isinstance(payload, dict) else None
        if not isinstance(evaluations, dict):
            path_reports.append({"path": str(rel_path), "exists": True, "candidate_rows": 0})
            continue
        model_specs = payload.get("model_specs")
        if isinstance(model_specs, dict) and isinstance(model_specs.get("local_generator"), str):
            source_models.append(model_specs["local_generator"])
        start = len(rows)
        for task_id, task_rows in sorted(evaluations.items()):
            if not isinstance(task_rows, list):
                continue
            for row_index, row in enumerate(task_rows):
                if not isinstance(row, dict):
                    continue
                candidate = _candidate_from_evalplus_row(
                    spec.source_id,
                    path,
                    str(task_id),
                    row,
                    row_index,
                )
                if candidate is not None:
                    rows.append(candidate)
        path_reports.append(
            {
                "path": str(rel_path),
                "exists": True,
                "candidate_rows": len(rows) - start,
            }
        )
    task_counts = Counter(row.task_id for row in rows)
    viable_task_ids = {task_id for task_id, count in task_counts.items() if count >= 2}
    viable_rows = [row for row in rows if row.task_id in viable_task_ids]
    labels = [row.passes_hidden_tests for row in viable_rows]
    generation_model = sorted(set(source_models))[0] if source_models else "unsloth/gemma-4-12B-it-GGUF"
    report = {
        "source_id": spec.source_id,
        "source_schema": "evalplus_checkpoint",
        "paths": path_reports,
        "candidate_rows": len(rows),
        "viable_candidate_rows": len(viable_rows),
        "task_n": len(viable_task_ids),
        "positive_n": sum(1 for row in viable_rows if row.passes_hidden_tests),
        "has_both_labels": len(set(labels)) == 2,
        "generation_model": {"hf_id": generation_model, "available": True},
    }
    return viable_rows, report


def _load_rows_for_spec(repo_root: Path, spec: PoolSpec) -> tuple[list[CodeCandidate], dict[str, Any]]:
    rows, report = _load_evalplus_checkpoint_rows(repo_root, spec)
    if report["candidate_rows"] > 0:
        return rows, report
    rows, report = _load_flat_rows_for_spec(repo_root, spec)
    report["source_schema"] = "flat_records"
    report.setdefault("generation_model", {"hf_id": "", "available": None})
    return rows, report


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
        raise MissingDistinctPool(attempted_sources, exp4233_source)

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
    raise MissingDistinctPool(attempted_sources, exp4233_source)


def resolve_cached_sota_gguf(repo_root: Path | str) -> dict[str, Any] | None:
    del repo_root
    cache_root = Path.home() / ".cache" / "huggingface" / "hub"
    for hf_id, cache_dir_name in SOTA_GGUF_CANDIDATES:
        cache_dir = cache_root / cache_dir_name
        if not cache_dir.exists():
            continue
        ggufs = sorted(
            path
            for path in cache_dir.rglob("*.gguf")
            if ".no_exist" not in path.parts and path.is_file()
        )
        if ggufs:
            return {
                "hf_id": hf_id,
                "model_path": str(ggufs[0]),
                "cache_dir": str(cache_dir),
                "available": True,
                "loader": "llama_cpp_model_path",
            }
    return None


def _replication_read(metrics: dict[str, Any]) -> tuple[str, str]:
    if not metrics.get("headroom_exists"):
        return "no_headroom", "complete: code_oracle_distinct_replication_no_headroom"
    if metrics.get("code_oracle_distinct_beats_vote") is True:
        return "replicates", "complete: code_oracle_distinct_replication_replicates"
    return "corpus_specific", "complete: code_oracle_distinct_replication_corpus_specific"


def _selected_report(pool: CandidatePool) -> dict[str, Any]:
    for report in reversed(pool.attempted_sources):
        if report.get("source_id") == pool.source_id:
            return report
    return {}


def _pool_generation_model(pool: CandidatePool) -> dict[str, Any]:
    report = _selected_report(pool)
    value = report.get("generation_model")
    if isinstance(value, dict):
        return dict(value)
    return {"hf_id": "", "available": None}


def _model_specs(
    feature_names: list[str],
    second_corpus_id: str,
    generation_model: dict[str, Any],
) -> dict[str, Any]:
    specs = _base_model_specs(feature_names)
    specs["second_corpus_id"] = second_corpus_id
    specs["generation_model"] = generation_model
    specs["source_distinctness_gate"] = "source file sha256 must not overlap exp4233 source corpus"
    specs["retry_policy"] = (
        "use cached distinct hidden-label pool first; if absent, require SOTA GGUF "
        "cache before any generation; retire rather than fabricate if no fresh pool "
        "can be built in-window"
    )
    return specs


def _blocked_checksum(
    verdict: str,
    random_seed: int,
    attempted_sources: list[dict[str, Any]],
    exp4233_source: dict[str, Any],
    generation_model: dict[str, Any],
) -> str:
    payload = {
        "verdict": verdict,
        "random_seed": int(random_seed),
        "attempted_sources": attempted_sources,
        "exp4233_source": exp4233_source,
        "generation_model": generation_model,
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _retired_artifact(
    *,
    honest_verdict: str,
    attempted_sources: list[dict[str, Any]],
    exp4233_source: dict[str, Any],
    generation_model: dict[str, Any],
    random_seed: int,
    duration_s: float,
) -> dict[str, Any]:
    blocked = honest_verdict == BLOCKED_MODEL_VERDICT
    return {
        "experiment": "experiment_4264_code_oracle_distinct_replication_retry",
        "schema": "carnot.code_oracle_distinct_replication_retry_4264.v1",
        "status": "complete",
        "honest_verdict": honest_verdict,
        "code_replication_beats_vote": False,
        "code_predictor_minus_vote_delta": 0.0,
        "code_predictor_minus_vote_ci95": [0.0, 0.0],
        "oracle_at_k": 0.0,
        "held_out_task_n": 0,
        "replication_read": "code_replication_retired",
        "code_replication_retired": True,
        "verifier_is_oracle": False,
        "model_specs": _model_specs(list(FEATURE_NAMES), "", generation_model),
        "random_seed": int(random_seed),
        "reproducibility_checksum": _blocked_checksum(
            honest_verdict,
            random_seed,
            attempted_sources,
            exp4233_source,
            generation_model,
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
        "controls": {
            "matched_control_delta": 0.0,
            "oracle_minus_vote": 0.0,
        },
        "oracle_minus_vote": 0.0,
        "headroom_exists": False,
        "ci95_excludes_zero": False,
        "bootstrap_resamples": 0,
        "candidate_pool": {
            "source_id": "",
            "source_schema": "",
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
            "Blocked before generation because no accepted SOTA GGUF was cached."
            if blocked
            else "Retired after no source-distinct cached pool was viable and no fresh "
            "second corpus could be built inside the task window."
        ),
        "adversarial_verify": {"status": "pending"},
    }


def _complete_artifact(
    pool: CandidatePool,
    metrics: dict[str, Any],
    *,
    exp4233_source: dict[str, Any],
    generation_model: dict[str, Any],
    off_fold_auroc: float,
    folds: list[list[str]],
    feature_names: list[str],
    random_seed: int,
    duration_s: float,
) -> dict[str, Any]:
    replication_read, honest_verdict = _replication_read(metrics)
    selected_report = _selected_report(pool)
    return {
        "experiment": "experiment_4264_code_oracle_distinct_replication_retry",
        "schema": "carnot.code_oracle_distinct_replication_retry_4264.v1",
        "status": "complete",
        "honest_verdict": honest_verdict,
        "code_replication_beats_vote": bool(metrics["code_oracle_distinct_beats_vote"]),
        "code_predictor_minus_vote_delta": metrics["code_predictor_minus_vote_delta"],
        "code_predictor_minus_vote_ci95": metrics["code_predictor_minus_vote_ci95"],
        "oracle_at_k": metrics["oracle_at_k"],
        "held_out_task_n": metrics["held_out_task_n"],
        "replication_read": replication_read,
        "code_replication_retired": False,
        "verifier_is_oracle": False,
        "model_specs": _model_specs(feature_names, pool.source_id, generation_model),
        "random_seed": int(random_seed),
        "reproducibility_checksum": _pool_reproducibility_checksum(
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
        "controls": {
            "matched_control_delta": metrics["matched_control_delta"],
            "oracle_minus_vote": metrics["oracle_minus_vote"],
        },
        "oracle_minus_vote": metrics["oracle_minus_vote"],
        "headroom_exists": metrics["headroom_exists"],
        "ci95_excludes_zero": metrics["ci95_excludes_zero"],
        "bootstrap_resamples": metrics["bootstrap_resamples"],
        "candidate_pool": {
            "source_id": pool.source_id,
            "source_schema": str(selected_report.get("source_schema", "")),
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
            "uses code text, AST/lexical statistics, normalized-code agreement, and "
            "self-consistency margins without executing candidate tests at inference."
        ),
        "adversarial_verify": {"status": "pending"},
    }


def validate_artifact(artifact: dict[str, Any]) -> None:
    missing = [field for field in REQUIRED_FIELDS if field not in artifact]
    if missing:
        raise ValueError(f"missing required fields: {missing}")
    verdict = artifact["honest_verdict"]
    if not isinstance(verdict, str) or not (
        verdict.startswith("complete:") or verdict == BLOCKED_MODEL_VERDICT
    ):
        raise ValueError("honest_verdict must be terminal-prefixed or the model blocker")
    for field in ("code_replication_beats_vote", "code_replication_retired"):
        if type(artifact[field]) is not bool:
            raise ValueError(f"{field} must be a bare bool")
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
        "code_replication_retired",
    }:
        raise ValueError("replication_read has an unknown value")
    if artifact["code_replication_beats_vote"] and artifact["replication_read"] != "replicates":
        raise ValueError("winning artifacts must set replication_read=replicates")
    if artifact["code_replication_retired"] and artifact["replication_read"] != "code_replication_retired":
        raise ValueError("retired artifacts must set replication_read=code_replication_retired")
    if artifact["verifier_is_oracle"] is not False:
        raise ValueError("verifier_is_oracle must be the bare bool false")
    if type(artifact["random_seed"]) is not int:
        raise ValueError("random_seed must be a bare int")
    if not isinstance(artifact["model_specs"], dict):
        raise ValueError("model_specs must be present")
    if artifact["model_specs"].get("verifier_is_oracle") is not False:
        raise ValueError("model_specs must preserve verifier_is_oracle=false")
    if artifact["field_principles"] != FIELD_PRINCIPLES:
        raise ValueError("field_principles do not match REQ-VERIFY-4264")
    if artifact["spec_refs"] != SPEC_REFS:
        raise ValueError("spec_refs do not match REQ-VERIFY-4264")


def _score_pool(
    pool: CandidatePool,
    *,
    exp4233_source: dict[str, Any],
    generation_model: dict[str, Any],
    random_seed: int,
    bootstrap_resamples: int,
    started_at: float,
) -> dict[str, Any]:
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
    return _complete_artifact(
        pool,
        metrics,
        exp4233_source=exp4233_source,
        generation_model=generation_model,
        off_fold_auroc=off_fold_auroc,
        folds=folds,
        feature_names=feature_names,
        random_seed=random_seed,
        duration_s=time.perf_counter() - started_at,
    )


def run(
    repo_root: Path | str = Path("."),
    *,
    random_seed: int = RANDOM_SEED,
    bootstrap_resamples: int = BOOTSTRAP_RESAMPLES,
    pool_specs: tuple[PoolSpec, ...] = DEFAULT_POOL_SPECS,
    exp4233_rel: Path = EXP4233_REL,
    gguf_resolver: Callable[[Path], dict[str, Any] | None] | None = None,
    generation_runner: Callable[[Path, dict[str, Any], int], CandidatePool | None] | None = None,
    adversarial_runner: Callable[[Path], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    start = time.perf_counter()
    root = Path(repo_root)
    output_path = root / OUTPUT_REL
    resolver = gguf_resolver or resolve_cached_sota_gguf
    try:
        pool = load_second_candidate_pool(root, pool_specs, exp4233_rel=exp4233_rel)
        resolved_model = resolver(root)
        generation_model = _pool_generation_model(pool)
        if resolved_model is not None:
            generation_model.update(resolved_model)
        exp4233_source = _exp4233_source(root, exp4233_rel)
        artifact = _score_pool(
            pool,
            exp4233_source=exp4233_source,
            generation_model=generation_model,
            random_seed=random_seed,
            bootstrap_resamples=bootstrap_resamples,
            started_at=start,
        )
    except MissingDistinctPool as exc:
        generation_model = resolver(root)
        if generation_model is None:
            generation_model = {"hf_id": "", "available": False, "model_path": None}
            artifact = _retired_artifact(
                honest_verdict=BLOCKED_MODEL_VERDICT,
                attempted_sources=exc.attempted_sources,
                exp4233_source=exc.exp4233_source,
                generation_model=generation_model,
                random_seed=random_seed,
                duration_s=time.perf_counter() - start,
            )
        else:
            generated_pool = (
                generation_runner(root, generation_model, random_seed)
                if generation_runner is not None
                else None
            )
            if generated_pool is not None:  # pragma: no cover - live generation hook.
                artifact = _score_pool(
                    generated_pool,
                    exp4233_source=exc.exp4233_source,
                    generation_model=generation_model,
                    random_seed=random_seed,
                    bootstrap_resamples=bootstrap_resamples,
                    started_at=start,
                )
            else:
                artifact = _retired_artifact(
                    honest_verdict=RETIRED_VERDICT,
                    attempted_sources=exc.attempted_sources,
                    exp4233_source=exc.exp4233_source,
                    generation_model=generation_model,
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
