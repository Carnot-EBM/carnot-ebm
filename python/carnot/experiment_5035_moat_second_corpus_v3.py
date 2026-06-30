"""Exp 5035: D4 v3 second-corpus verifier-moat generalization check.

Spec refs: REQ-VERIFY-5035, SCENARIO-VERIFY-5035.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import json
import math
from pathlib import Path
import re
import sys
import time
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "python") not in sys.path:  # pragma: no cover - direct script execution
    sys.path.insert(0, str(REPO_ROOT / "python"))

from carnot import experiment_5021_moat_second_corpus_v2 as base5021  # noqa: E402
from carnot import experiment_5032_uprm_replication_v3 as exp5032  # noqa: E402
from carnot import experiment_5033_ebrm_uncertainty_verifier_v3 as exp5033  # noqa: E402
from carnot import moat_benchmark_harness as harness  # noqa: E402
from carnot.moat_benchmark_harness import (  # noqa: E402
    DEFAULT_RANDOM_SEED,
    OracleDistinctnessError,
    evaluate_verifier,
)


JsonDict = dict[str, Any]
JsonMap = Mapping[str, Any]
AuditRunner = Callable[[Path], JsonDict]
SummaryRunner = Callable[[Path], int]
Clock = Callable[[], float]
CorpusLoader = Callable[[int], list[JsonDict]]
CandidateRowsLoader = Callable[..., tuple[list[JsonDict], Path] | list[JsonDict]]

PreconditionCheck = base5021.PreconditionCheck
VerifierSelection = base5021.VerifierSelection
CorpusAttempt = base5021.CorpusAttempt
SecondCorpusUnavailable = base5021.SecondCorpusUnavailable

EXPERIMENT_ID = 5035
EXPERIMENT_NAME = "experiment_5035_moat_second_corpus_v3"
RESULT_RELATIVE_PATH = "results/experiment_5035_moat_second_corpus_v3.json"
D1_ARTIFACT_RELATIVE_PATH = "results/experiment_5031_lora_ebm_scorer_musr_v3.json"
D2_ARTIFACT_RELATIVE_PATH = "results/experiment_5032_uprm_replication_v3.json"
D3_ARTIFACT_RELATIVE_PATH = "results/experiment_5033_ebrm_uncertainty_verifier_v3.json"
MODEL_NAME = "gemma-4-12B-it-GGUF"
MODEL_HF_ID = "unsloth/gemma-4-12B-it-GGUF"
SPEC_REFS = ["REQ-VERIFY-5035", "SCENARIO-VERIFY-5035"]
RANDOM_SEED = DEFAULT_RANDOM_SEED
DEFAULT_LIMIT = 200
DEFAULT_K = 5
DEFAULT_SERVER_PORT = 8919

FIELD_PRINCIPLES: dict[str, dict[str, str]] = {
    "honest_verdict": {
        "principle": (
            "terminal prefix; a win is success_moat_generalizes_<corpus>_<delta>, "
            "a scoped result is complete_moat_musr_scoped_<corpus>_no_confirm, "
            "no verifier is blocked_no_best_verifier."
        )
    },
    "verifier_is_oracle": {
        "principle": (
            "false -- the best verifier scores reasoning quality, never the answer's "
            "executable correctness (must pass check_circular_moat_overclaim)."
        )
    },
    "headroom_present": {
        "principle": (
            "true required on the 2nd corpus vs the GENUINE tuned-SC "
            "(FALSE_NEGATIVE_RISK guard); if false, the corpus is excluded from "
            "the moat claim."
        )
    },
    "best_verifier_from": {
        "principle": ("which arm (D1/D2/D3) provided the best verifier by MuSR delta_vs_tuned_sc.")
    },
    "second_corpus": {
        "principle": (
            "the chosen confirmed-cached headroom-present oracle-distinct corpus "
            "(GPQA/MMLU-Pro-hard/MATH-500-hard)."
        )
    },
    "second_corpus_accuracy": {
        "principle": "the best verifier's oracle-distinct accuracy on the 2nd corpus."
    },
    "genuine_tuned_sc_accuracy_second": {
        "principle": ("the GENUINE K-way tuned-SC on the 2nd corpus (headroom-control).")
    },
    "delta_vs_tuned_sc_second": {
        "principle": (
            "the cross-corpus moat lift (signed); CI95-excl-0 is the generalization confirmation."
        )
    },
    "paired_ci95_second": {"principle": "paired bootstrap CI95 of the 2nd-corpus delta."},
    "n_questions": {"principle": ">=200 (sample-size rigor)."},
    "model_specs": {"principle": "the generator + the best verifier -- the methodology stamp."},
    "inference_substrate": {
        "principle": (
            "verifier_ensemble_against_cached_candidates if candidates are reused "
            "(1s floor); live_llm_inference only if generated fresh (>=60s)."
        )
    },
    "random_seed": {"principle": "determinism for the bootstrap."},
    "preconditions_checked": {
        "principle": ("records verifier/corpus/headroom checks; a missing resource emits blocked_.")
    },
}

REQUIRED_ARTIFACT_FIELDS = tuple(FIELD_PRINCIPLES) + (
    "schema",
    "experiment",
    "experiment_id",
    "spec_refs",
    "result_path",
    "deliverable_stage",
    "oracle_distinctness_enforced",
    "oracle_at_k_second",
    "mcnemar_p_second",
    "candidate_cache_path",
    "non_degenerate",
    "adversarial_verify_clean",
    "adversarial_verify_flags",
    "summarize_artifact_exit_code",
    "duration_s",
    "field_principles",
    "reproducibility_checksum",
)

_number = base5021._number
_slug_corpus = base5021._slug_corpus
_format_delta = base5021._format_delta
_ci_excludes_zero_positive = base5021._ci_excludes_zero_positive
_compact_adversarial_flags = base5021._compact_adversarial_flags
_audit_is_clean = base5021._audit_is_clean
_normalise_harness_evaluation = base5021._normalise_harness_evaluation
_load_corpus_rows = base5021._load_corpus_rows
_read_jsonl = base5021._read_jsonl
write_json = base5021.write_json


def _precondition_dicts(checks: Sequence[PreconditionCheck | JsonMap]) -> list[JsonDict]:
    return [
        check.as_dict() if isinstance(check, PreconditionCheck) else dict(check) for check in checks
    ]


def _read_json_object(path: Path) -> JsonDict | None:
    return base5021._read_json_object(path)


def _d3_non_degenerate(payload: JsonMap) -> bool:
    abstention_rate = _number(payload.get("abstention_rate"))
    if abstention_rate is not None and abstention_rate > harness.ABSTENTION_DEGENERACY_THRESHOLD:
        return False
    guard = payload.get("degeneracy_guard")
    if isinstance(guard, Mapping) and guard.get("degeneracy_flag") is True:
        return False
    calibration = payload.get("uncertainty_calibration")
    if isinstance(calibration, Mapping) and calibration.get("degeneracy_flag") is True:
        return False
    return True


def select_best_verifier(
    root: Path = REPO_ROOT,
) -> tuple[VerifierSelection | None, list[PreconditionCheck]]:
    """Select the best usable Exp 5031/5032/5033 verifier without fallback."""

    root = Path(root)
    specs: tuple[tuple[str, str, str, str, Callable[[JsonMap], bool] | None], ...] = (
        (
            "D1",
            "artifact_energy",
            D1_ARTIFACT_RELATIVE_PATH,
            "trained_scorer_accuracy",
            lambda payload: payload.get("scorer_trained") is True,
        ),
        (
            "D2",
            "uprm_process_score",
            D2_ARTIFACT_RELATIVE_PATH,
            "uprm_selection_accuracy",
            lambda payload: (
                payload.get("scoring_path")
                in {"uprm_logprob", "lc_erd_consistency", "self_supervised_frozen"}
            ),
        ),
        (
            "D3",
            "ebrm_uncertainty_v3",
            D3_ARTIFACT_RELATIVE_PATH,
            "ebrm_selection_accuracy",
            _d3_non_degenerate,
        ),
    )
    candidates: list[VerifierSelection] = []
    checks: list[PreconditionCheck] = []
    for arm, scorer_kind, relative_path, accuracy_field, extra_usable in specs:
        candidate, check = base5021._verifier_from_artifact(
            arm=arm,
            scorer_kind=scorer_kind,
            path=root / relative_path,
            accuracy_field=accuracy_field,
            extra_usable=extra_usable,
        )
        checks.append(check)
        if candidate is not None:
            candidates.append(candidate)
    if not candidates:
        return None, checks
    return max(candidates, key=lambda item: (item.delta_vs_tuned_sc, item.arm)), checks


def candidate_cache_relative_path(corpus: str) -> str:
    return f"results/experiment_5035_candidates_{_slug_corpus(corpus)}.jsonl"


def shared_b2_candidate_cache_relative_path(corpus: str) -> str:
    return f"results/experiment_5029_shared_logprob_candidate_cache_v2_{_slug_corpus(corpus)}.jsonl"


def prior_v2_candidate_cache_relative_path(corpus: str) -> str:
    return f"results/experiment_5021_candidates_{_slug_corpus(corpus)}.jsonl"


def legacy_candidate_cache_relative_path(corpus: str) -> str:
    return f"results/experiment_5006_candidates_{_slug_corpus(corpus)}.jsonl"


def _candidate_cache_paths(root: Path, corpus: str) -> list[Path]:
    return [
        root / shared_b2_candidate_cache_relative_path(corpus),
        root / prior_v2_candidate_cache_relative_path(corpus),
        root / legacy_candidate_cache_relative_path(corpus),
        root / candidate_cache_relative_path(corpus),
    ]


def default_corpus_loaders() -> list[tuple[str, CorpusLoader]]:  # pragma: no cover - cache boundary
    return [
        ("GPQA", lambda limit: harness.load_gpqa_cached(limit=limit)),
        ("MMLU-Pro-hard", lambda limit: harness.load_mmlu_pro_hard_cached(limit=limit)),
        ("MATH-500-hard", lambda limit: harness.load_math_500_hard_cached(limit=limit)),
    ]


def default_candidate_rows_loader(
    *,
    root: Path,
    corpus: str,
    corpus_rows: Sequence[JsonMap],
    candidate_cache_path: Path,
    limit: int,
    min_questions: int,
    k_candidates: int,
    random_seed: int,
    server_port: int,
) -> tuple[list[JsonDict], Path]:
    del candidate_cache_path, k_candidates, random_seed, server_port
    required = min(min_questions, len(corpus_rows))
    for path in _candidate_cache_paths(root, corpus):
        rows = _read_jsonl(path)
        if len(rows) >= required:
            return rows[:limit], path
    paths = ", ".join(path.as_posix() for path in _candidate_cache_paths(root, corpus))
    raise SecondCorpusUnavailable(f"no cached candidate rows >= {required}: {paths}")


def _rows_have_uprm_scores(rows: Sequence[JsonMap]) -> bool:
    candidates = [candidate for row in rows for candidate in row.get("candidates", [])]
    return bool(candidates) and all(
        _number(candidate.get("uprm_process_score")) is not None for candidate in candidates
    )


def _uprm_energy(candidate: Mapping[str, Any]) -> float:
    score = _number(candidate.get("uprm_process_score"))
    if score is not None:
        return -score
    process_score = _number(candidate.get("process_score"))
    if process_score is not None:
        return -process_score
    raise SecondCorpusUnavailable("uPRM candidate lacks process score")


def evaluate_rows_with_verifier(
    rows: Sequence[JsonMap],
    *,
    verifier: VerifierSelection,
    seed: int,
    bootstrap_samples: int,
) -> JsonDict:
    """Score second-corpus candidate rows with the selected v3 verifier."""

    rows_list = [dict(row) for row in rows if row.get("candidates")]
    if verifier.scorer_kind == "uprm_process_score":
        prepared_rows = (
            rows_list
            if _rows_have_uprm_scores(rows_list)
            else exp5032.prepare_rows_with_process_scores(
                rows_list, scoring_path="self_supervised_frozen"
            )
        )
        return _normalise_harness_evaluation(
            evaluate_verifier(
                prepared_rows,
                scorer=_uprm_energy,
                seed=seed,
                bootstrap_samples=bootstrap_samples,
                headroom_threshold=harness.HEADROOM_THRESHOLD,
            )
        )
    if verifier.scorer_kind == "ebrm_uncertainty_v3":
        prepared = exp5033.prepare_rows_with_ebrm_distributions(rows_list)
        evaluation = exp5033.evaluate_ebrm_rows(
            prepared,
            threshold=verifier.ebrm_threshold,
            seed=seed,
            bootstrap_samples=bootstrap_samples,
        )
        tuned_sc = dict(evaluation.get("tuned_self_consistency") or {})
        return {
            "n_rows": int(evaluation["n_rows"]),
            "accuracy": float(evaluation["ebrm_selection_accuracy"]),
            "genuine_tuned_sc_accuracy": float(tuned_sc["accuracy"]),
            "delta": float(evaluation["delta_vs_tuned_sc"]),
            "paired_ci95": [float(value) for value in evaluation["paired_ci95"]],
            "mcnemar_p": float(evaluation["mcnemar_p"]),
            "oracle_at_k": float(evaluation["oracle_at_k"]),
            "headroom_present": bool(evaluation["headroom_present"]),
            "n_flips_possible": int(evaluation["n_flips_possible"]),
            "non_degenerate": not bool(evaluation.get("degeneracy_flag")),
            "raw": dict(evaluation),
        }
    return _normalise_harness_evaluation(
        evaluate_verifier(
            rows_list,
            scorer=base5021._finite_candidate_energy,
            seed=seed,
            bootstrap_samples=bootstrap_samples,
            headroom_threshold=harness.HEADROOM_THRESHOLD,
        )
    )


def _oracle_distinctness_enforced(rows: Sequence[JsonMap]) -> bool:
    try:
        evaluate_verifier(rows, scorer=lambda candidate: candidate["gold"], bootstrap_samples=8)
    except OracleDistinctnessError:
        return True
    return False  # pragma: no cover - indicates shared harness regression


def _candidate_loader_result(
    loaded: tuple[list[JsonDict], Path] | list[JsonDict],
    fallback_path: Path,
) -> tuple[list[JsonDict], Path]:
    if isinstance(loaded, tuple):
        return loaded
    return loaded, fallback_path


def select_second_corpus_attempt(
    *,
    root: Path,
    verifier: VerifierSelection,
    corpus_loaders: Sequence[tuple[str, CorpusLoader]],
    candidate_rows_loader: CandidateRowsLoader,
    limit: int,
    min_questions: int,
    k_candidates: int,
    random_seed: int,
    server_port: int,
    bootstrap_samples: int,
) -> tuple[CorpusAttempt | None, list[PreconditionCheck]]:
    checks: list[PreconditionCheck] = []
    scored_attempts: list[CorpusAttempt] = []
    for name, loader in corpus_loaders:
        try:
            corpus_rows = _load_corpus_rows(loader, limit=limit, min_questions=min_questions)
        except Exception as exc:
            checks.append(
                PreconditionCheck(
                    f"second_corpus_{_slug_corpus(name)}",
                    False,
                    f"{type(exc).__name__}: {exc}",
                )
            )
            continue
        corpus_check = PreconditionCheck(
            f"second_corpus_{_slug_corpus(name)}",
            True,
            f"{len(corpus_rows)} cached row(s), required >= {min_questions}",
        )
        candidate_cache_path = root / candidate_cache_relative_path(name)
        try:
            loaded = candidate_rows_loader(
                root=root,
                corpus=name,
                corpus_rows=corpus_rows,
                candidate_cache_path=candidate_cache_path,
                limit=limit,
                min_questions=min_questions,
                k_candidates=k_candidates,
                random_seed=random_seed,
                server_port=server_port,
            )
            candidate_rows, used_cache_path = _candidate_loader_result(loaded, candidate_cache_path)
            candidate_rows = [dict(row) for row in candidate_rows[:limit] if row.get("candidates")]
            if len(candidate_rows) < min_questions:
                raise SecondCorpusUnavailable(
                    f"only {len(candidate_rows)} candidate row(s), required {min_questions}"
                )
            if not _oracle_distinctness_enforced(candidate_rows):
                raise OracleDistinctnessError("shared harness did not block gold access")
            evaluation = evaluate_rows_with_verifier(
                candidate_rows,
                verifier=verifier,
                seed=random_seed,
                bootstrap_samples=bootstrap_samples,
            )
        except Exception as exc:
            checks.extend(
                [
                    corpus_check,
                    PreconditionCheck(
                        f"candidate_cache_{_slug_corpus(name)}",
                        False,
                        f"{type(exc).__name__}: {exc}",
                        candidate_cache_path.as_posix(),
                    ),
                ]
            )
            continue
        attempt_checks = [
            corpus_check,
            PreconditionCheck(
                f"candidate_cache_{_slug_corpus(name)}",
                True,
                f"{len(candidate_rows)} cached candidate row(s), required >= {min_questions}",
                used_cache_path.as_posix(),
            ),
            PreconditionCheck(
                f"headroom_{_slug_corpus(name)}",
                bool(evaluation["headroom_present"]),
                (
                    f"oracle@K={evaluation['oracle_at_k']:.6f}; "
                    f"genuine_tuned_sc={evaluation['genuine_tuned_sc_accuracy']:.6f}; "
                    f"flips={evaluation['n_flips_possible']}"
                ),
            ),
        ]
        checks.extend(attempt_checks)
        attempt = CorpusAttempt(
            corpus=name,
            rows=candidate_rows,
            candidate_cache_path=used_cache_path,
            evaluation=evaluation,
            checks=attempt_checks,
        )
        scored_attempts.append(attempt)
        if evaluation["headroom_present"]:
            return attempt, checks
    return (scored_attempts[0] if scored_attempts else None), checks


def _base_artifact(
    *,
    honest_verdict: str,
    best_verifier: VerifierSelection | None,
    preconditions_checked: Sequence[PreconditionCheck | JsonMap],
    root: Path,
    duration_s: float,
    deliverable_stage: str,
) -> JsonDict:
    return {
        "schema": "carnot.experiment_5035_moat_second_corpus_v3.v1",
        "experiment": EXPERIMENT_NAME,
        "experiment_id": EXPERIMENT_ID,
        "spec_refs": list(SPEC_REFS),
        "result_path": str(root / RESULT_RELATIVE_PATH),
        "deliverable_stage": deliverable_stage,
        "honest_verdict": honest_verdict,
        "verifier_is_oracle": False,
        "headroom_present": False,
        "best_verifier_from": best_verifier.arm if best_verifier else None,
        "second_corpus": None,
        "second_corpus_accuracy": None,
        "genuine_tuned_sc_accuracy_second": None,
        "delta_vs_tuned_sc_second": None,
        "paired_ci95_second": None,
        "n_questions": 0,
        "model_specs": {
            "generator_model": MODEL_NAME,
            "generator_hf_id": MODEL_HF_ID,
            "generator_gpu": 0,
            "best_verifier": best_verifier.arm if best_verifier else None,
            "best_verifier_scorer_kind": best_verifier.scorer_kind if best_verifier else None,
            "best_verifier_musr_delta": best_verifier.delta_vs_tuned_sc if best_verifier else None,
            "best_verifier_artifact_path": best_verifier.artifact_path.as_posix()
            if best_verifier
            else None,
            "best_verifier_model_specs": best_verifier.model_specs if best_verifier else {},
        },
        "inference_substrate": "precondition_check_only",
        "random_seed": RANDOM_SEED,
        "preconditions_checked": _precondition_dicts(preconditions_checked),
        "oracle_distinctness_enforced": False,
        "oracle_at_k_second": None,
        "mcnemar_p_second": None,
        "candidate_cache_path": None,
        "non_degenerate": False,
        "adversarial_verify_clean": False,
        "adversarial_verify_flags": [],
        "summarize_artifact_exit_code": None,
        "duration_s": round(float(duration_s), 6),
        "field_principles": dict(FIELD_PRINCIPLES),
        "reproducibility_checksum": "",
    }


def reproducibility_checksum(payload: JsonMap) -> str:
    return base5021.reproducibility_checksum(payload)


def _checksum(payload: JsonMap) -> str:
    basis = {
        "experiment_id": payload.get("experiment_id"),
        "honest_verdict": payload.get("honest_verdict"),
        "best_verifier_from": payload.get("best_verifier_from"),
        "second_corpus": payload.get("second_corpus"),
        "delta_vs_tuned_sc_second": payload.get("delta_vs_tuned_sc_second"),
        "random_seed": payload.get("random_seed"),
    }
    return reproducibility_checksum(basis)


def build_skeleton_artifact(
    *,
    best_verifier: VerifierSelection | None,
    preconditions_checked: Sequence[PreconditionCheck | JsonMap],
    root: Path = REPO_ROOT,
    duration_s: float = 0.0,
) -> JsonDict:
    artifact = _base_artifact(
        honest_verdict="running_moat_second_corpus_v3_schema_skeleton",
        best_verifier=best_verifier,
        preconditions_checked=preconditions_checked,
        root=root,
        duration_s=duration_s,
        deliverable_stage="schema_skeleton",
    )
    artifact["reproducibility_checksum"] = _checksum(artifact)
    return artifact


def build_blocked_artifact(
    *,
    missing_resource: str,
    best_verifier: VerifierSelection | None,
    preconditions_checked: Sequence[PreconditionCheck | JsonMap],
    root: Path = REPO_ROOT,
    duration_s: float,
    blocked_error: str | None = None,
) -> JsonDict:
    artifact = _base_artifact(
        honest_verdict=f"blocked_{missing_resource}",
        best_verifier=best_verifier,
        preconditions_checked=preconditions_checked,
        root=root,
        duration_s=duration_s,
        deliverable_stage="blocked_precondition",
    )
    artifact["blocked_error"] = blocked_error
    artifact["reproducibility_checksum"] = _checksum(artifact)
    return artifact


def build_complete_artifact(
    *,
    attempt: CorpusAttempt,
    best_verifier: VerifierSelection,
    preconditions_checked: Sequence[PreconditionCheck | JsonMap],
    root: Path = REPO_ROOT,
    duration_s: float,
    inference_substrate: str = "verifier_ensemble_against_cached_candidates",
) -> JsonDict:
    evaluation = attempt.evaluation
    delta = float(evaluation["delta"])
    ci95 = [float(value) for value in evaluation["paired_ci95"]]
    corpus_slug = _slug_corpus(attempt.corpus)
    success = (
        bool(evaluation["headroom_present"])
        and bool(evaluation["non_degenerate"])
        and delta > 0.0
        and _ci_excludes_zero_positive(ci95)
    )
    verdict = (
        f"success_moat_generalizes_{corpus_slug}_{_format_delta(delta)}"
        if success
        else f"complete_moat_musr_scoped_{corpus_slug}_no_confirm"
    )
    artifact = _base_artifact(
        honest_verdict=verdict,
        best_verifier=best_verifier,
        preconditions_checked=preconditions_checked,
        root=root,
        duration_s=duration_s,
        deliverable_stage="complete",
    )
    artifact.update(
        {
            "headroom_present": bool(evaluation["headroom_present"]),
            "second_corpus": attempt.corpus,
            "second_corpus_accuracy": round(float(evaluation["accuracy"]), 6),
            "genuine_tuned_sc_accuracy_second": round(
                float(evaluation["genuine_tuned_sc_accuracy"]), 6
            ),
            "delta_vs_tuned_sc_second": round(delta, 6),
            "paired_ci95_second": ci95,
            "n_questions": int(evaluation["n_rows"]),
            "model_specs": {
                **artifact["model_specs"],
                "candidate_cache_path": attempt.candidate_cache_path.as_posix(),
                "tuned_self_consistency_config": evaluation["raw"]
                .get("tuned_self_consistency", {})
                .get("config"),
            },
            "inference_substrate": inference_substrate,
            "oracle_distinctness_enforced": True,
            "oracle_at_k_second": round(float(evaluation["oracle_at_k"]), 6),
            "mcnemar_p_second": round(float(evaluation["mcnemar_p"]), 6),
            "candidate_cache_path": attempt.candidate_cache_path.as_posix(),
            "non_degenerate": bool(evaluation["non_degenerate"]),
            "evaluation": evaluation,
        }
    )
    artifact["reproducibility_checksum"] = _checksum(artifact)
    return artifact


def _default_audit_runner(path: Path) -> JsonDict:  # pragma: no cover - external script hook
    return base5021._default_audit_runner(path)


def _default_summary_runner(path: Path) -> int:  # pragma: no cover - external script hook
    return base5021._default_summary_runner(path)


def _finalize_artifact(
    artifact: JsonDict,
    artifact_path: Path,
    *,
    audit_runner: AuditRunner,
    summary_runner: SummaryRunner,
    write: bool,
) -> JsonDict:
    if write:
        write_json(artifact_path, artifact)
    audit_report = audit_runner(artifact_path)
    artifact["adversarial_verify_flags"] = _compact_adversarial_flags(audit_report)
    artifact["adversarial_verify_clean"] = _audit_is_clean(audit_report)
    artifact["adversarial_verify_report"] = audit_report
    artifact["summarize_artifact_exit_code"] = int(summary_runner(artifact_path))
    artifact["reproducibility_checksum"] = _checksum(artifact)
    if write:
        write_json(artifact_path, artifact)
    return artifact


def run(
    *,
    root: Path = REPO_ROOT,
    artifact_path: Path | None = None,
    corpus_loaders: Sequence[tuple[str, CorpusLoader]] | None = None,
    candidate_rows_loader: CandidateRowsLoader = default_candidate_rows_loader,
    audit_runner: AuditRunner | None = None,
    summary_runner: SummaryRunner | None = None,
    min_questions: int = DEFAULT_LIMIT,
    limit: int = DEFAULT_LIMIT,
    k_candidates: int = DEFAULT_K,
    bootstrap_samples: int = 2000,
    random_seed: int = RANDOM_SEED,
    server_port: int = DEFAULT_SERVER_PORT,
    now: Clock = time.perf_counter,
    write: bool = True,
) -> JsonDict:
    root = Path(root)
    artifact_path = Path(artifact_path) if artifact_path else root / RESULT_RELATIVE_PATH
    audit = audit_runner or _default_audit_runner
    summarize = summary_runner or _default_summary_runner
    start = float(now())
    if write:
        write_json(
            artifact_path,
            build_skeleton_artifact(
                best_verifier=None,
                preconditions_checked=[],
                root=root,
                duration_s=0.0,
            ),
        )

    best_verifier, verifier_checks = select_best_verifier(root)
    if best_verifier is None:
        return _finalize_artifact(
            build_blocked_artifact(
                missing_resource="no_best_verifier",
                best_verifier=None,
                preconditions_checked=verifier_checks,
                root=root,
                duration_s=float(now()) - start,
                blocked_error="D1/D2/D3 v3 artifacts are all blocked or unusable",
            ),
            artifact_path,
            audit_runner=audit,
            summary_runner=summarize,
            write=write,
        )

    loaders = list(corpus_loaders) if corpus_loaders is not None else default_corpus_loaders()
    attempt, corpus_checks = select_second_corpus_attempt(
        root=root,
        verifier=best_verifier,
        corpus_loaders=loaders,
        candidate_rows_loader=candidate_rows_loader,
        limit=limit,
        min_questions=min_questions,
        k_candidates=k_candidates,
        random_seed=random_seed,
        server_port=server_port,
        bootstrap_samples=bootstrap_samples,
    )
    checks = [*verifier_checks, *corpus_checks]
    if attempt is None:
        return _finalize_artifact(
            build_blocked_artifact(
                missing_resource="second_corpus_unavailable",
                best_verifier=best_verifier,
                preconditions_checked=checks,
                root=root,
                duration_s=float(now()) - start,
                blocked_error="no priority second corpus had enough cached headroom candidate rows",
            ),
            artifact_path,
            audit_runner=audit,
            summary_runner=summarize,
            write=write,
        )

    return _finalize_artifact(
        build_complete_artifact(
            attempt=attempt,
            best_verifier=best_verifier,
            preconditions_checked=checks,
            root=root,
            duration_s=float(now()) - start,
        ),
        artifact_path,
        audit_runner=audit,
        summary_runner=summarize,
        write=write,
    )


def artifact_schema_errors(artifact: JsonMap) -> list[str]:
    errors: list[str] = []
    for field in REQUIRED_ARTIFACT_FIELDS:
        if field not in artifact:
            errors.append(field)
    if artifact.get("spec_refs") != SPEC_REFS:
        errors.append("spec_refs")
    if artifact.get("verifier_is_oracle") is not False:
        errors.append("verifier_is_oracle")
    if artifact.get("field_principles") != FIELD_PRINCIPLES:
        errors.append("field_principles")
    for field in (
        "headroom_present",
        "oracle_distinctness_enforced",
        "non_degenerate",
        "adversarial_verify_clean",
    ):
        if not isinstance(artifact.get(field), bool):
            errors.append(field)
    ci95 = artifact.get("paired_ci95_second")
    if ci95 is not None and (
        not isinstance(ci95, list)
        or len(ci95) != 2
        or not all(_number(value) is not None for value in ci95)
    ):
        errors.append("paired_ci95_second")
    for field in (
        "second_corpus_accuracy",
        "genuine_tuned_sc_accuracy_second",
        "oracle_at_k_second",
        "mcnemar_p_second",
    ):
        value = artifact.get(field)
        if value is not None and not (
            isinstance(value, (int, float)) and 0.0 <= float(value) <= 1.0
        ):
            errors.append(field)
    if artifact.get("delta_vs_tuned_sc_second") is not None and not isinstance(
        artifact.get("delta_vs_tuned_sc_second"), (int, float)
    ):
        errors.append("delta_vs_tuned_sc_second")
    if not isinstance(artifact.get("preconditions_checked"), list):
        errors.append("preconditions_checked")
    if not isinstance(artifact.get("model_specs"), dict):
        errors.append("model_specs")
    if not isinstance(artifact.get("duration_s"), (int, float)):
        errors.append("duration_s")
    if not str(artifact.get("honest_verdict", "")).startswith(
        ("blocked_", "running_", "complete_", "success_")
    ):
        errors.append("honest_verdict")
    return sorted(set(errors))


def main() -> int:  # pragma: no cover - requested script entrypoint
    artifact = run()
    errors = artifact_schema_errors(artifact)
    path = REPO_ROOT / RESULT_RELATIVE_PATH
    print(f"{path}: {artifact.get('honest_verdict')}")
    if errors:
        print(f"schema_errors={errors}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
