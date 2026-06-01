"""Exp 3641: code corpus verifier firing and FoVer-math transfer.

Spec: REQ-CODE-3641, SCENARIO-CODE-3641.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
import hashlib
import importlib
import json
import math
from pathlib import Path
import re
import time
from typing import Any

import numpy as np

from carnot.eval.metrics import auroc


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[3]
RUN_DATE = "20260601"
ARTIFACT = "experiment_3641_code_corpus_verifiers_fire_transfer_v3"
SCHEMA = "carnot.code_corpus_verifiers_fire_transfer.v3"
OUTPUT_REL_PATH = Path("results/experiment_3641_code_corpus_verifiers_fire_transfer_v3.json")
CORPUS_REL_PATH = Path("data/code_verification_corpus_v1.jsonl")
EXP1999_REL_PATH = Path("results/experiment_1999_code_verification_humaneval.json")
EXP2910_REL_PATH = Path("results/experiment_2910_sota_code_generation_corrigendum_v2.json")
DEFAULT_MANIFEST_PATHS = {
    "mbpp": Path("data/eval_manifests/mbpp_20260522.jsonl"),
    "humaneval": Path("data/eval_manifests/humaneval_20260522.jsonl"),
}
CONFIGURED_VERIFIER_MODULES = (
    "controlled_invariance_executor_v2",
    "executable_monitor_runtime_adapter",
    "ast_structure_verifier",
    "code_structural_dependency_verifier",
)
BOOTSTRAP_SEEDS = (3641, 3642, 3643)
RANDOM_SEED = 3641
MIN_EXAMPLES = 50
TRANSFER_ANCHOR_RANGE = (0.04, 0.08)
TRANSFER_THRESHOLD = 0.04
INFERENCE_SUBSTRATE = (
    "verifier_ensemble_against_cached_candidates "
    "(principle: scores the cached code corpus; no LLM load)."
)
REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "inference_substrate",
    "code_corpus_path",
    "code_corpus_name",
    "n_examples",
    "code_verifiers_fire",
    "execution_verifier_auroc",
    "math_signal_code_auroc",
    "code_confidence_baseline_auroc",
    "transfer_delta_vs_literature",
    "hypothesis_supported",
    "random_seed",
    "reproducibility_checksum",
    "duration_s",
)
FIELD_PRINCIPLES = {
    "honest_verdict": "Terminal prefix for reconciler classification.",
    "inference_substrate": "Scores cached code candidates with local verifiers; no LLM load.",
    "code_corpus_path": "Where the labeled code corpus lives -- the apparatus exp3642 reuses for the code row.",
    "code_corpus_name": "Which labeled code benchmark was scored -- provenance.",
    "n_examples": "Sample-size rigor (>=50 candidates with labels).",
    "code_verifiers_fire": (
        "BARE bool. True iff the execution-applicable verifiers scored "
        "(n_scored>0, variance>0) -- distinguishes 'transfer failed' from "
        "'verifiers never ran'. STORE AS BARE true/false."
    ),
    "execution_verifier_auroc": "The execution-applicable verifiers' standalone code-error signal + CI.",
    "math_signal_code_auroc": "The transferred FoVer-math verifier signal on code -- the core transfer number.",
    "code_confidence_baseline_auroc": "Headroom check: a baseline near 1.0 means no room to discriminate, so a null is uninformative.",
    "transfer_delta_vs_literature": "Signed comparison to the +4-8pt arXiv:2506.00027 anchor -- contextualizes Carnot vs the published positive control.",
    "hypothesis_supported": "'transfer' (2506.00027) or 'discriminative_fragility' (2504.16828) -- the falsifiable mechanism.",
    "random_seed": "Determinism precondition.",
    "reproducibility_checksum": "Drift detection.",
    "duration_s": "Plausibility floor.",
}


def build_artifact(
    root: Path | str = REPO_ROOT,
    *,
    started_s: float | None = None,
    now_s: float | None = None,
    tests_run: Sequence[str] | None = None,
    score_overrides: Mapping[str, Any] | None = None,
) -> JsonDict:
    """Build the terminal Exp 3641 artifact."""

    root_path = Path(root)
    start = time.perf_counter() if started_s is None else float(started_s)
    verifier_imports = import_configured_verifiers()
    corpus_rows, corpus_status = assemble_labeled_corpus(root_path)
    if len(corpus_rows) < MIN_EXAMPLES:
        artifact = blocked_artifact(
            root=root_path,
            started_s=start,
            now_s=now_s,
            reason="blocked_no_labeled_code_corpus",
            corpus_status=corpus_status,
            verifier_imports=verifier_imports,
            tests_run=tests_run,
        )
        validate_artifact(artifact)
        return artifact

    corpus_path = write_corpus_jsonl(root_path, corpus_rows)
    execution = score_execution_verifiers(
        corpus_rows,
        root_path,
        verifier_imports=verifier_imports,
        score_overrides=score_overrides or {},
    )
    math_scores = score_math_signal(corpus_rows, score_overrides=score_overrides or {})
    confidence_scores = score_confidence_baseline(corpus_rows, score_overrides=score_overrides or {})
    labels = error_labels(corpus_rows)
    execution_metrics = metric_bundle(labels, execution["scores"])
    math_metrics = metric_bundle(labels, math_scores)
    confidence_metrics = metric_bundle(labels, confidence_scores)
    code_verifiers_fire = bool(
        execution["n_scored"] > 0 and float(execution["score_variance"]) > 0.0
    )
    transfer_delta = transfer_delta_summary(math_metrics, confidence_metrics)
    hypothesis = hypothesis_supported(code_verifiers_fire, transfer_delta)
    verdict = terminal_verdict(
        n_examples=len(corpus_rows),
        code_verifiers_fire=code_verifiers_fire,
        hypothesis=hypothesis,
    )
    finished = time.perf_counter() if now_s is None else float(now_s)
    artifact: JsonDict = {
        "artifact": ARTIFACT,
        "schema": SCHEMA,
        "run_date": RUN_DATE,
        "honest_verdict": verdict,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "code_corpus_path": str(CORPUS_REL_PATH),
        "code_corpus_name": corpus_status["selected_corpus_name"],
        "n_examples": len(corpus_rows),
        "code_verifiers_fire": code_verifiers_fire,
        "execution_verifier_auroc": execution_metrics,
        "math_signal_code_auroc": math_metrics,
        "code_confidence_baseline_auroc": confidence_metrics,
        "transfer_delta_vs_literature": transfer_delta,
        "hypothesis_supported": hypothesis,
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": reproducibility_checksum(
            corpus_rows,
            execution["scores"],
            math_scores,
            confidence_scores,
        ),
        "duration_s": round(max(0.0, finished - start), 6),
        "field_principles": dict(FIELD_PRINCIPLES),
        "acceptance_gate": acceptance_gate(
            len(corpus_rows),
            code_verifiers_fire,
            confidence_metrics,
        ),
        "null_discipline": null_discipline(code_verifiers_fire, confidence_metrics),
        "exp1999_corpus_status": corpus_status["exp1999_status"],
        "corpus_source_status": corpus_status,
        "verifier_import_status": verifier_imports,
        "execution_verifier_summary": {
            "n_scored": execution["n_scored"],
            "score_variance": execution["score_variance"],
            "per_verifier": execution["per_verifier"],
        },
        "source_artifacts": source_artifacts(root_path),
        "tests_run": list(tests_run or []),
        "scripts_research_conductor_modified": False,
        "ops_docs_reconciliation_left_to_conductor": True,
    }
    validate_artifact(artifact)
    return artifact


def write_artifact(
    root: Path | str = REPO_ROOT,
    *,
    output_path: Path | str = OUTPUT_REL_PATH,
    tests_run: Sequence[str] | None = None,
) -> Path:
    """Build, validate, and persist the terminal JSON artifact."""

    root_path = Path(root)
    output = _repo_path(root_path, Path(output_path))
    artifact = build_artifact(root_path, tests_run=tests_run)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output


def blocked_artifact(
    *,
    root: Path | str,
    started_s: float,
    now_s: float | None,
    reason: str,
    corpus_status: Mapping[str, Any] | None = None,
    verifier_imports: Mapping[str, Any] | None = None,
    tests_run: Sequence[str] | None = None,
) -> JsonDict:
    """Return a terminal blocked artifact with the required schema fields."""

    root_path = Path(root)
    finished = time.perf_counter() if now_s is None else float(now_s)
    empty_metrics = metric_bundle([], [])
    status = dict(corpus_status or {})
    artifact: JsonDict = {
        "artifact": ARTIFACT,
        "schema": SCHEMA,
        "run_date": RUN_DATE,
        "honest_verdict": f"complete: {reason}",
        "inference_substrate": INFERENCE_SUBSTRATE,
        "code_corpus_path": None,
        "code_corpus_name": None,
        "n_examples": int(status.get("n_candidate_rows") or 0),
        "code_verifiers_fire": False,
        "execution_verifier_auroc": empty_metrics,
        "math_signal_code_auroc": empty_metrics,
        "code_confidence_baseline_auroc": empty_metrics,
        "transfer_delta_vs_literature": transfer_delta_summary(empty_metrics, empty_metrics),
        "hypothesis_supported": "discriminative_fragility",
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": reproducibility_checksum([], [], [], []),
        "duration_s": round(max(0.0, finished - float(started_s)), 6),
        "field_principles": dict(FIELD_PRINCIPLES),
        "acceptance_gate": acceptance_gate(0, False, empty_metrics),
        "null_discipline": null_discipline(False, empty_metrics),
        "exp1999_corpus_status": status.get("exp1999_status", "not_checked"),
        "corpus_source_status": status,
        "verifier_import_status": dict(verifier_imports or import_configured_verifiers()),
        "execution_verifier_summary": {"n_scored": 0, "score_variance": 0.0, "per_verifier": []},
        "source_artifacts": source_artifacts(root_path),
        "tests_run": list(tests_run or []),
        "scripts_research_conductor_modified": False,
        "ops_docs_reconciliation_left_to_conductor": True,
    }
    validate_artifact(artifact)
    return artifact


def assemble_labeled_corpus(root: Path) -> tuple[list[JsonDict], JsonDict]:
    """Assemble labeled source rows from Exp 1999 or the Exp 2910 fallback."""

    exp1999 = _read_json_object(root / EXP1999_REL_PATH)
    exp1999_rows = corpus_from_exp1999(exp1999)
    exp1999_status = exp1999_corpus_status(exp1999, exp1999_rows)
    if len(exp1999_rows) >= MIN_EXAMPLES:
        return exp1999_rows, {
            "selected_corpus_name": "experiment_1999_code_verification_humaneval",
            "selected_source_path": str(EXP1999_REL_PATH),
            "exp1999_status": exp1999_status,
            "fallback_status": "not_needed",
            "n_candidate_rows": len(exp1999_rows),
        }

    exp2910 = _read_json_object(root / EXP2910_REL_PATH)
    exp2910_rows = corpus_from_exp2910(exp2910)
    if len(exp2910_rows) >= MIN_EXAMPLES:
        return exp2910_rows, {
            "selected_corpus_name": "experiment_2910_sota_code_generation_corrigendum_v2_mbpp_humaneval",
            "selected_source_path": str(EXP2910_REL_PATH),
            "exp1999_status": exp1999_status,
            "fallback_status": "used_exp2910_labeled_candidates",
            "n_candidate_rows": len(exp2910_rows),
        }

    return [], {
        "selected_corpus_name": None,
        "selected_source_path": None,
        "exp1999_status": exp1999_status,
        "fallback_status": "no_mbpp_or_livecodebench_labeled_source_corpus",
        "n_candidate_rows": 0,
    }


def corpus_from_exp1999(payload: Mapping[str, Any]) -> list[JsonDict]:
    """Extract Exp 1999 rows only when labels and candidate source are present."""

    rows = []
    for idx, row in enumerate(payload.get("results") or []):
        if not isinstance(row, Mapping):
            continue
        code = _first_present_string(
            row,
            ("candidate_code", "generated_code", "baseline_code", "extracted_code", "raw_response"),
        )
        label = _label_from_row(row, ("label", "passed", "baseline_passed", "repair_passed"))
        if code and label is not None:
            rows.append(
                corpus_row(
                    candidate_code=code,
                    label=label,
                    test_outcome="passed" if label else "failed",
                    source="experiment_1999_code_verification_humaneval",
                    task_id=str(row.get("task_id") or f"HumanEval/{idx}"),
                    metadata={"source_index": idx, "extracted_constraints": row.get("extracted_constraints")},
                )
            )
    return rows


def exp1999_corpus_status(payload: Mapping[str, Any], extracted_rows: Sequence[Mapping[str, Any]]) -> str:
    """Classify whether Exp 1999 can honestly supply candidate source."""

    if not payload:
        return "missing"
    results = payload.get("results")
    if not isinstance(results, list) or not results:
        return "no_results"
    has_labels = any(_label_from_row(row, ("label", "passed", "baseline_passed", "repair_passed")) is not None for row in results if isinstance(row, Mapping))
    has_code = any(
        _first_present_string(
            row,
            ("candidate_code", "generated_code", "baseline_code", "extracted_code", "raw_response"),
        )
        for row in results
        if isinstance(row, Mapping)
    )
    if len(extracted_rows) >= MIN_EXAMPLES:
        return "labels_and_candidate_code"
    if has_labels and not has_code:
        return "labels_without_candidate_code"
    if has_code and not has_labels:
        return "candidate_code_without_labels"
    return "insufficient_rows"


def corpus_from_exp2910(payload: Mapping[str, Any]) -> list[JsonDict]:
    """Extract the on-disk Exp 2910 MBPP/HumanEval candidate corpus."""

    rows = []
    for idx, row in enumerate(payload.get("candidate_results") or []):
        if not isinstance(row, Mapping):
            continue
        code = _first_present_string(row, ("extracted_code", "raw_response", "generated_text"))
        label = _label_from_row(row, ("passed",))
        if not code or label is None:
            continue
        stable_id = str(row.get("stable_id") or row.get("task_id") or f"candidate-{idx}")
        rows.append(
            corpus_row(
                candidate_code=code,
                label=label,
                test_outcome=str(row.get("row_status") or ("passed" if label else "failed")),
                source="experiment_2910_sota_code_generation_corrigendum_v2",
                task_id=stable_id,
                metadata={
                    "candidate_index": row.get("candidate_index"),
                    "corpus": row.get("corpus"),
                    "error_message": row.get("error_message"),
                    "error_type": row.get("error_type"),
                    "extraction_status": row.get("extraction_status"),
                    "generation_duration_s": row.get("generation_duration_s"),
                    "manifest_path": row.get("manifest_path"),
                    "n_tests": row.get("n_tests"),
                    "raw_response_sha256": row.get("raw_response_sha256"),
                    "row_status": row.get("row_status"),
                    "runtime_success": row.get("runtime_success"),
                    "stable_id": stable_id,
                    "syntax_success": row.get("syntax_success"),
                    "tokens_generated": row.get("tokens_generated"),
                },
            )
        )
    return rows


def corpus_row(
    *,
    candidate_code: str,
    label: bool,
    test_outcome: str,
    source: str,
    task_id: str,
    metadata: Mapping[str, Any],
) -> JsonDict:
    """Normalize one source artifact row into the Exp 3641 JSONL schema."""

    code = str(candidate_code)
    return {
        "candidate_code": code,
        "label": bool(label),
        "test_outcome": str(test_outcome),
        "source": str(source),
        "task_id": str(task_id),
        "candidate_sha256": hashlib.sha256(code.encode("utf-8")).hexdigest(),
        "metadata": dict(metadata),
    }


def write_corpus_jsonl(root: Path, rows: Sequence[Mapping[str, Any]]) -> Path:
    """Persist the labeled code-candidate corpus as deterministic JSONL."""

    output = root / CORPUS_REL_PATH
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "".join(json.dumps(dict(row), sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    return output


def import_configured_verifiers() -> JsonDict:
    """Import the four configured verifier modules and diagnose scorer shape."""

    statuses: JsonDict = {}
    for name in CONFIGURED_VERIFIER_MODULES:
        try:
            module = importlib.import_module(f"carnot.verify.{name}")
        except Exception as exc:
            statuses[name] = {
                "importable": False,
                "candidate_score_api": False,
                "diagnosis": f"import_failed:{type(exc).__name__}:{exc}",
            }
            continue
        diagnosis = verifier_interface_diagnosis(name, module)
        statuses[name] = {"importable": True, **diagnosis}
    return statuses


def verifier_interface_diagnosis(name: str, module: Any) -> JsonDict:
    """Return whether a module exposes a candidate-code scoring path."""

    if name == "ast_structure_verifier" and hasattr(module, "ASTStructureVerifier"):
        return {
            "candidate_score_api": True,
            "diagnosis": "ASTStructureVerifier.score parses candidate text with ast.parse.",
        }
    if name == "code_structural_dependency_verifier" and hasattr(module, "verify_candidate_source"):
        return {
            "candidate_score_api": True,
            "diagnosis": "verify_candidate_source parses candidate AST against manifest contracts.",
        }
    return {
        "candidate_score_api": False,
        "diagnosis": "importable but exposes an artifact/replay workflow, not a per-candidate code scorer.",
    }


def score_execution_verifiers(
    rows: Sequence[Mapping[str, Any]],
    root: Path,
    *,
    verifier_imports: Mapping[str, Any],
    score_overrides: Mapping[str, Any],
) -> JsonDict:
    """Score code-applicable verifiers and keep inert interfaces explicit."""

    if "execution_scores" in score_overrides:
        scores = [float(score) for score in score_overrides["execution_scores"]]
        variance = score_variance(scores)
        return {
            "scores": scores,
            "n_scored": len(scores),
            "score_variance": variance,
            "per_verifier": [
                {
                    "name": "synthetic_execution_override",
                    "fired": bool(scores and variance > 0.0),
                    "n_scored": len(scores),
                    "score_variance": variance,
                    "diagnosis": "test override for SCENARIO-CODE-3641 verdict discipline",
                }
            ],
        }

    per_row_scores: list[list[float]] = [[] for _ in rows]
    per_verifier: list[JsonDict] = []
    ast_scores = ast_structure_scores(rows)
    append_verifier_scores(per_row_scores, ast_scores)
    per_verifier.append(verifier_summary("ast_structure_verifier", ast_scores, verifier_imports))

    structural_scores = structural_dependency_scores(rows, root)
    append_verifier_scores(per_row_scores, structural_scores)
    per_verifier.append(
        verifier_summary("code_structural_dependency_verifier", structural_scores, verifier_imports)
    )

    for inert_name in ("controlled_invariance_executor_v2", "executable_monitor_runtime_adapter"):
        per_verifier.append(verifier_summary(inert_name, [], verifier_imports))

    scores = [float(np.mean(row_scores)) for row_scores in per_row_scores if row_scores]
    return {
        "scores": scores,
        "n_scored": len(scores),
        "score_variance": score_variance(scores),
        "per_verifier": per_verifier,
    }


def ast_structure_scores(rows: Sequence[Mapping[str, Any]]) -> list[float]:
    """Run the AST structure verifier over each candidate source."""

    from carnot.verify.ast_structure_verifier import ASTStructureVerifier

    verifier = ASTStructureVerifier()
    return [float(verifier.score(str(row["candidate_code"]))) for row in rows]


def structural_dependency_scores(rows: Sequence[Mapping[str, Any]], root: Path) -> list[float]:
    """Run manifest-backed structural dependency checks where contracts exist."""

    from carnot.verify import code_structural_dependency_verifier as dep

    manifests = load_manifest_lookup(rows, root)
    scores = []
    for row in rows:
        metadata = row.get("metadata") if isinstance(row.get("metadata"), Mapping) else {}
        corpus = normalize_corpus(metadata.get("corpus"))
        stable_id = str(metadata.get("stable_id") or row.get("task_id") or "")
        manifest_row = manifests.get((corpus, stable_id))
        if manifest_row is None:
            continue
        contract = dep.build_contract_from_manifest_row(
            corpus,
            manifest_row,
            manifest_path=str(manifest_row.get("_manifest_path") or ""),
        )
        result = dep.verify_candidate_source(
            contract,
            str(row["candidate_code"]),
            "exp3641_candidate",
            candidate_id=str(row.get("candidate_sha256") or stable_id),
        )
        scores.append(min(1.0, len(result.get("violations") or []) / 3.0))
    return scores


def load_manifest_lookup(rows: Sequence[Mapping[str, Any]], root: Path) -> dict[tuple[str, str], JsonDict]:
    """Load MBPP/HumanEval manifest rows for structural dependency scoring."""

    paths: dict[str, Path] = {corpus: root / rel for corpus, rel in DEFAULT_MANIFEST_PATHS.items()}
    for row in rows:
        metadata = row.get("metadata") if isinstance(row.get("metadata"), Mapping) else {}
        corpus = normalize_corpus(metadata.get("corpus"))
        manifest_path = metadata.get("manifest_path")
        if corpus and manifest_path:
            paths[corpus] = _repo_path(root, Path(str(manifest_path)))
    lookup: dict[tuple[str, str], JsonDict] = {}
    for corpus, path in paths.items():
        for manifest_row in _read_jsonl(path):
            stable_id = str(manifest_row.get("stable_id") or "")
            if stable_id:
                stored = dict(manifest_row)
                stored["_manifest_path"] = str(path)
                lookup[(corpus, stable_id)] = stored
    return lookup


def append_verifier_scores(per_row_scores: list[list[float]], scores: Sequence[float]) -> None:
    """Append row-aligned scores to the aggregate execution signal."""

    for idx, score in enumerate(scores[: len(per_row_scores)]):
        per_row_scores[idx].append(float(score))


def verifier_summary(
    name: str,
    scores: Sequence[float],
    verifier_imports: Mapping[str, Any],
) -> JsonDict:
    """Summarize one configured verifier's firing status."""

    variance = score_variance(scores)
    status = verifier_imports.get(name, {}) if isinstance(verifier_imports, Mapping) else {}
    diagnosis = str(status.get("diagnosis") or "not_import_checked")
    if not scores and status.get("candidate_score_api") is False:
        diagnosis = f"{diagnosis}; inert_for_candidate_scoring"
    return {
        "name": name,
        "importable": bool(status.get("importable")),
        "candidate_score_api": bool(status.get("candidate_score_api")),
        "fired": bool(scores and variance > 0.0),
        "n_scored": len(scores),
        "score_variance": variance,
        "diagnosis": diagnosis,
    }


def score_math_signal(
    rows: Sequence[Mapping[str, Any]],
    *,
    score_overrides: Mapping[str, Any],
) -> list[float]:
    """Score a FoVer-math-derived verifier signal on code candidates."""

    if "math_scores" in score_overrides:
        return [float(score) for score in score_overrides["math_scores"]]

    from carnot.verify.rprm_step_reward import RPRMStepReward
    from carnot.verify.semenergy_probe import SemEnergyProbe
    from carnot.verify.z3_math_verifier import Z3MathVerifier

    sem = SemEnergyProbe()
    z3 = Z3MathVerifier()
    rprm = RPRMStepReward()
    texts = [str(row["candidate_code"]) for row in rows]
    sem_scores = minmax_normalize([sem.score_response_proxy(text) for text in texts])
    z3_scores = [float(z3.score(text)) for text in texts]
    rprm_scores = [float(rprm.verify_response("", text).overall_violation_prob) for text in texts]
    return [
        float(np.mean([sem_score, z3_score, rprm_score]))
        for sem_score, z3_score, rprm_score in zip(sem_scores, z3_scores, rprm_scores, strict=True)
    ]


def score_confidence_baseline(
    rows: Sequence[Mapping[str, Any]],
    *,
    score_overrides: Mapping[str, Any],
) -> list[float]:
    """Compute a cached self-consistency/confidence baseline error score."""

    if "confidence_scores" in score_overrides:
        return [float(score) for score in score_overrides["confidence_scores"]]

    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row.get("task_id") or "")].append(row)
    normalized_counts: dict[str, Counter[str]] = {}
    for task_id, group_rows in groups.items():
        normalized_counts[task_id] = Counter(normalize_code(str(row["candidate_code"])) for row in group_rows)

    scores = []
    for row in rows:
        task_id = str(row.get("task_id") or "")
        group_size = max(1, len(groups[task_id]))
        code_frequency = normalized_counts[task_id][normalize_code(str(row["candidate_code"]))] / group_size
        metadata = row.get("metadata") if isinstance(row.get("metadata"), Mapping) else {}
        candidate_index = _coerce_float(metadata.get("candidate_index"), 0.0)
        max_index = max(1.0, group_size - 1.0)
        index_confidence = 1.0 - min(max(candidate_index, 0.0), max_index) / max_index
        confidence = 0.75 * code_frequency + 0.25 * index_confidence
        scores.append(1.0 - confidence)
    return scores


def metric_bundle(
    labels: Sequence[int],
    scores: Sequence[float],
    *,
    seeds: Sequence[int] = BOOTSTRAP_SEEDS,
    n_bootstrap: int = 200,
) -> JsonDict:
    """Return AUROC point estimate plus deterministic bootstrap CI."""

    clean_labels, clean_scores = finite_label_scores(labels, scores)
    if not clean_scores:
        return {
            "point": None,
            "ci95": None,
            "n": 0,
            "n_positive_errors": 0,
            "n_negative_correct": 0,
            "score_variance": 0.0,
            "bootstrap_seeds": list(seeds),
            "seed_mean_aurocs": [],
        }
    point = float(auroc(clean_labels, clean_scores))
    seed_means = []
    boot_values = []
    arr_labels = np.asarray(clean_labels, dtype=np.float64)
    arr_scores = np.asarray(clean_scores, dtype=np.float64)
    for seed in seeds:
        rng = np.random.default_rng(int(seed))
        seed_values = []
        for _ in range(n_bootstrap):
            idx = rng.integers(0, len(arr_labels), size=len(arr_labels))
            value = float(auroc(arr_labels[idx], arr_scores[idx]))
            seed_values.append(value)
            boot_values.append(value)
        seed_means.append(round(float(np.mean(seed_values)), 6))
    ci_low, ci_high = np.percentile(np.asarray(boot_values, dtype=np.float64), [2.5, 97.5])
    positives = int(sum(1 for label in clean_labels if label == 1))
    return {
        "point": round(point, 6),
        "ci95": [round(float(ci_low), 6), round(float(ci_high), 6)],
        "n": len(clean_scores),
        "n_positive_errors": positives,
        "n_negative_correct": len(clean_scores) - positives,
        "score_variance": score_variance(clean_scores),
        "bootstrap_seeds": list(seeds),
        "seed_mean_aurocs": seed_means,
    }


def finite_label_scores(
    labels: Sequence[int],
    scores: Sequence[float],
) -> tuple[list[int], list[float]]:
    """Drop non-finite scores and align labels to the scored prefix."""

    clean_labels: list[int] = []
    clean_scores: list[float] = []
    for label, score in zip(labels, scores, strict=False):
        score_f = float(score)
        if math.isfinite(score_f):
            clean_labels.append(int(label))
            clean_scores.append(score_f)
    return clean_labels, clean_scores


def transfer_delta_summary(
    math_metrics: Mapping[str, Any],
    baseline_metrics: Mapping[str, Any],
) -> JsonDict:
    """Compare observed math-signal lift to the +4-8 point literature anchor."""

    math_point = math_metrics.get("point")
    baseline_point = baseline_metrics.get("point")
    if math_point is None or baseline_point is None:
        observed = None
        lower = None
        upper = None
    else:
        observed = round(float(math_point) - float(baseline_point), 6)
        lower = round(observed - TRANSFER_ANCHOR_RANGE[0], 6)
        upper = round(observed - TRANSFER_ANCHOR_RANGE[1], 6)
    return {
        "literature_anchor": {
            "paper": "arXiv:2506.00027",
            "reported_positive_control": "math-trained PRMs transfer to code as well as or better than code-trained PRMs",
            "anchor_delta_range": list(TRANSFER_ANCHOR_RANGE),
        },
        "observed_delta_vs_confidence_baseline": observed,
        "delta_vs_lower_anchor": lower,
        "delta_vs_upper_anchor": upper,
        "meets_lower_anchor": bool(observed is not None and observed >= TRANSFER_THRESHOLD),
    }


def hypothesis_supported(
    code_verifiers_fire: bool,
    transfer_delta: Mapping[str, Any],
) -> str:
    """Classify the falsifiable transfer mechanism."""

    if code_verifiers_fire and transfer_delta.get("meets_lower_anchor") is True:
        return "transfer"
    return "discriminative_fragility"


def terminal_verdict(
    *,
    n_examples: int,
    code_verifiers_fire: bool,
    hypothesis: str,
) -> str:
    """Return one of the required terminal verdict strings."""

    if n_examples < MIN_EXAMPLES:
        return "complete: blocked_no_labeled_code_corpus"
    if not code_verifiers_fire:
        return "complete: code_corpus_built_but_execution_verifiers_inert_diagnosed"
    if hypothesis == "transfer":
        return "complete: code_corpus_built_verifiers_fire_math_signal_transfers_to_code"
    return "complete: code_corpus_built_verifiers_fire_math_signal_does_not_transfer_discriminative_fragility"


def acceptance_gate(
    n_examples: int,
    code_verifiers_fire: bool,
    confidence_metrics: Mapping[str, Any],
) -> JsonDict:
    """Compute the user-specified acceptance gate."""

    passed = bool(
        n_examples >= MIN_EXAMPLES
        and code_verifiers_fire is True
        and confidence_metrics.get("point") is not None
    )
    return {
        "condition": "n_examples >= 50 AND code_verifiers_fire == true AND code_confidence_baseline_auroc present",
        "passed": passed,
        "principle": "A code transfer verdict requires the verifiers actually fired on a headroom-bearing labeled corpus; a null without that is a wiring failure, not evidence.",
    }


def null_discipline(code_verifiers_fire: bool, confidence_metrics: Mapping[str, Any]) -> JsonDict:
    """Record whether a no-transfer verdict has the required headroom."""

    point = confidence_metrics.get("point")
    headroom = bool(point is not None and float(point) < 0.99)
    return {
        "false_negative_risk_checked": True,
        "code_verifiers_fire": bool(code_verifiers_fire),
        "confidence_sc_headroom": headroom,
        "null_verdict_trustworthy": bool(code_verifiers_fire and headroom),
    }


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Validate the Exp 3641 terminal artifact contract."""

    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    if missing:
        raise ValueError(f"missing required artifact fields: {missing}")
    if type(artifact.get("code_verifiers_fire")) is not bool:
        raise ValueError("code_verifiers_fire must be a bare top-level bool")
    verdict = str(artifact.get("honest_verdict") or "")
    if not verdict.startswith("complete:"):
        raise ValueError("honest_verdict must start with 'complete:'")
    if not isinstance(artifact.get("n_examples"), int):
        raise ValueError("n_examples must be an int")
    duration = artifact.get("duration_s")
    if not isinstance(duration, (int, float)) or float(duration) < 0.0:
        raise ValueError("duration_s must be a non-negative number")


def error_labels(rows: Sequence[Mapping[str, Any]]) -> list[int]:
    """Return AUROC-positive labels where 1 means code error."""

    return [0 if bool(row.get("label")) else 1 for row in rows]


def score_variance(scores: Sequence[float]) -> float:
    """Return deterministic population variance for score firing checks."""

    if not scores:
        return 0.0
    return round(float(np.var(np.asarray(scores, dtype=np.float64))), 12)


def minmax_normalize(scores: Sequence[float]) -> list[float]:
    """Normalize scores to [0, 1] without changing their ordering."""

    if not scores:
        return []
    arr = np.asarray(scores, dtype=np.float64)
    lo = float(np.min(arr))
    hi = float(np.max(arr))
    if hi == lo:
        return [0.5 for _ in scores]
    return [float((value - lo) / (hi - lo)) for value in arr]


def normalize_code(code: str) -> str:
    """Collapse comments and whitespace for self-consistency grouping."""

    lines = []
    for line in code.splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            lines.append(re.sub(r"\s+", " ", stripped))
    return "\n".join(lines)


def normalize_corpus(corpus: Any) -> str:
    """Normalize corpus labels used by manifest artifacts."""

    text = str(corpus or "").strip().lower()
    if text in {"human_eval", "human eval", "humaneval"}:
        return "humaneval"
    if text == "mbpp":
        return "mbpp"
    return text


def reproducibility_checksum(
    rows: Sequence[Mapping[str, Any]],
    execution_scores: Sequence[float],
    math_scores: Sequence[float],
    confidence_scores: Sequence[float],
) -> str:
    """Return a drift checksum over corpus identities and scores."""

    payload = {
        "candidate_sha256": [row.get("candidate_sha256") for row in rows],
        "execution_scores": [round(float(score), 8) for score in execution_scores],
        "math_scores": [round(float(score), 8) for score in math_scores],
        "confidence_scores": [round(float(score), 8) for score in confidence_scores],
        "random_seed": RANDOM_SEED,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def source_artifacts(root: Path) -> list[str]:
    """List source files consulted by the offline workflow."""

    paths = [EXP1999_REL_PATH, EXP2910_REL_PATH, *DEFAULT_MANIFEST_PATHS.values()]
    return [str(path) for path in paths if (root / path).exists()]


def _read_json_object(path: Path) -> JsonDict:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return dict(payload) if isinstance(payload, Mapping) else {}


def _read_jsonl(path: Path) -> list[JsonDict]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    rows = []
    for line in lines:
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, Mapping):
            rows.append(dict(payload))
    return rows


def _first_present_string(row: Mapping[str, Any], keys: Sequence[str]) -> str:
    for key in keys:
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return ""


def _label_from_row(row: Mapping[str, Any], keys: Sequence[str]) -> bool | None:
    for key in keys:
        if key not in row:
            continue
        value = row.get(key)
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        text = str(value).strip().lower()
        if text in {"passed", "pass", "true", "correct", "1"}:
            return True
        if text in {"failed", "fail", "false", "incorrect", "0"}:
            return False
    return None


def _coerce_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _repo_path(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path


__all__ = [
    "CONFIGURED_VERIFIER_MODULES",
    "CORPUS_REL_PATH",
    "EXP1999_REL_PATH",
    "EXP2910_REL_PATH",
    "OUTPUT_REL_PATH",
    "REQUIRED_ARTIFACT_FIELDS",
    "assemble_labeled_corpus",
    "blocked_artifact",
    "build_artifact",
    "import_configured_verifiers",
    "validate_artifact",
    "write_artifact",
]
