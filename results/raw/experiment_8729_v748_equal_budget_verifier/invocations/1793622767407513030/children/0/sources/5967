"""Exp 3658: balanced second-corpus code generalization replication.

Spec: REQ-CODE-3658, SCENARIO-CODE-3658.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import ast
import hashlib
import json
from pathlib import Path
import time
from typing import Any

from carnot.verify import code_corpus_verifiers_fire_transfer_v3 as exp3641


JsonDict = dict[str, Any]
Executor = Callable[[str, dict[str, Any], float], Any]

REPO_ROOT = Path(__file__).resolve().parents[3]
RUN_DATE = "20260601"
ARTIFACT = "experiment_3658_code_generalization_second_corpus"
SCHEMA = "carnot.code_generalization_second_corpus.v1"
OUTPUT_REL_PATH = Path("results/experiment_3658_code_generalization_second_corpus.json")
CORPUS_REL_PATH = Path("data/code_verification_corpus_v2.jsonl")
HUMANEVAL_MANIFEST_REL_PATH = Path("data/eval_manifests/humaneval_20260522.jsonl")
EXP3641_REL_PATH = Path("results/experiment_3641_code_corpus_verifiers_fire_transfer_v3.json")
RANDOM_SEED = 3658
BOOTSTRAP_SEEDS = (3658, 3659, 3660)
MIN_EXAMPLES = 50
MIN_CLASS_FRACTION = 0.20
DEFAULT_TARGET_PER_CLASS = 30
TRANSFER_DELTA_THRESHOLD = 0.04
INFERENCE_SUBSTRATE = (
    "verifier_ensemble_against_cached_candidates "
    "(principle: scores the cached second code corpus; no LLM load)."
)

REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "inference_substrate",
    "second_code_corpus_name",
    "n_examples",
    "class_balance",
    "code_verifiers_fire",
    "execution_verifier_auroc",
    "math_signal_code_auroc",
    "confidence_baseline_auroc",
    "code_generalization_replicates",
    "random_seed",
    "reproducibility_checksum",
    "duration_s",
)

FIELD_PRINCIPLES = {
    "honest_verdict": "Terminal prefix for reconciler classification.",
    "inference_substrate": (
        "verifier_ensemble_against_cached_candidates (principle: scores the cached second code corpus; no LLM load)."
    ),
    "second_code_corpus_name": (
        "Which second benchmark was scored (MBPP/LiveCodeBench/HumanEval-split) -- provenance + de-risks the single-corpus result."
    ),
    "n_examples": "Sample-size rigor (>=50 with labels).",
    "class_balance": (
        "Both classes >= 20% -- an AUROC on a 296/24 split (the exp3641 weakness) is what this replication fixes."
    ),
    "code_verifiers_fire": (
        "True iff the execution-applicable verifiers scored (n_scored>0, variance>0) -- distinguishes a wiring bug from a real limitation."
    ),
    "execution_verifier_auroc": "The execution verifiers' code-error signal on the second corpus + CI.",
    "math_signal_code_auroc": (
        "The transferred FoVer-math signal on the second corpus -- the replication number."
    ),
    "confidence_baseline_auroc": "Headroom check on the balanced corpus.",
    "code_generalization_replicates": (
        "BARE bool. True iff math->code transfer holds on the balanced second corpus consistent with exp3641 -- hardens (or refutes) the code claim. STORE AS BARE true/false."
    ),
    "random_seed": "Determinism precondition.",
    "reproducibility_checksum": "Drift detection.",
    "duration_s": "Plausibility floor.",
}


def build_artifact(
    root: Path | str = REPO_ROOT,
    *,
    started_s: float | None = None,
    now_s: float | None = None,
    corpus_rows: Sequence[Mapping[str, Any]] | None = None,
    exp3641_artifact: Mapping[str, Any] | None = None,
    score_overrides: Mapping[str, Any] | None = None,
    tests_run: Sequence[str] | None = None,
    target_per_class: int = DEFAULT_TARGET_PER_CLASS,
    executor: Executor | None = None,
    n_bootstrap: int = 200,
) -> JsonDict:
    """Build the Exp 3658 artifact from a balanced second code corpus."""

    root_path = Path(root)
    start = time.perf_counter() if started_s is None else float(started_s)
    verifier_imports = exp3641.import_configured_verifiers()
    if corpus_rows is None:
        rows, corpus_status = build_second_labeled_code_corpus(
            root_path,
            target_per_class=target_per_class,
            executor=executor,
        )
    else:
        rows = [dict(row) for row in corpus_rows]
        corpus_status = {
            "selected_corpus_name": "HumanEval-split" if rows else None,
            "selected_source_path": "test_fixture_override" if rows else None,
            "n_candidate_rows": len(rows),
            "n_manifest_rows_scanned": 0,
            "fallback_status": "test_fixture_override",
        }

    balance = class_balance(rows)
    if len(rows) < MIN_EXAMPLES or not balance["balanced"]:
        artifact = blocked_artifact(
            root=root_path,
            started_s=start,
            now_s=now_s,
            reason="blocked_no_second_code_corpus",
            corpus_rows=rows,
            corpus_status=corpus_status,
            verifier_imports=verifier_imports,
            tests_run=tests_run,
        )
        validate_artifact(artifact)
        return artifact

    write_corpus_jsonl(root_path, rows)
    overrides = score_overrides or {}
    execution = exp3641.score_execution_verifiers(
        rows,
        root_path,
        verifier_imports=verifier_imports,
        score_overrides=overrides,
    )
    math_scores = exp3641.score_math_signal(rows, score_overrides=overrides)
    confidence_scores = exp3641.score_confidence_baseline(rows, score_overrides=overrides)
    labels = exp3641.error_labels(rows)
    execution_metrics = exp3641.metric_bundle(
        labels,
        execution["scores"],
        seeds=BOOTSTRAP_SEEDS,
        n_bootstrap=n_bootstrap,
    )
    math_metrics = exp3641.metric_bundle(
        labels,
        math_scores,
        seeds=BOOTSTRAP_SEEDS,
        n_bootstrap=n_bootstrap,
    )
    confidence_metrics = exp3641.metric_bundle(
        labels,
        confidence_scores,
        seeds=BOOTSTRAP_SEEDS,
        n_bootstrap=n_bootstrap,
    )
    code_verifiers_fire = bool(
        execution["n_scored"] > 0 and float(execution["score_variance"]) > 0.0
    )
    prior = (
        dict(exp3641_artifact)
        if exp3641_artifact is not None
        else _read_exp3641_artifact(root_path)
    )
    replicates = code_generalization_replicates(
        code_verifiers_fire=code_verifiers_fire,
        balance=balance,
        math_metrics=math_metrics,
        confidence_metrics=confidence_metrics,
        exp3641_artifact=prior,
    )
    finished = time.perf_counter() if now_s is None else float(now_s)
    artifact: JsonDict = {
        "artifact": ARTIFACT,
        "schema": SCHEMA,
        "run_date": RUN_DATE,
        "honest_verdict": terminal_verdict(
            n_examples=len(rows),
            balance=balance,
            code_verifiers_fire=code_verifiers_fire,
            replicates=replicates,
        ),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "second_code_corpus_name": corpus_status.get("selected_corpus_name"),
        "second_code_corpus_path": str(CORPUS_REL_PATH),
        "n_examples": len(rows),
        "class_balance": balance,
        "code_verifiers_fire": code_verifiers_fire,
        "execution_verifier_auroc": execution_metrics,
        "math_signal_code_auroc": math_metrics,
        "confidence_baseline_auroc": confidence_metrics,
        "code_generalization_replicates": replicates,
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": reproducibility_checksum(
            rows,
            execution["scores"],
            math_scores,
            confidence_scores,
            prior,
        ),
        "duration_s": round(max(0.0, finished - start), 6),
        "field_principles": dict(FIELD_PRINCIPLES),
        "acceptance_gate": acceptance_gate(len(rows), code_verifiers_fire, balance),
        "cross_corpus_consistency": cross_corpus_consistency(
            math_metrics,
            confidence_metrics,
            prior,
            replicates,
        ),
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
    corpus_rows: Sequence[Mapping[str, Any]] | None = None,
    exp3641_artifact: Mapping[str, Any] | None = None,
    score_overrides: Mapping[str, Any] | None = None,
    tests_run: Sequence[str] | None = None,
) -> Path:
    """Build, validate, and persist the Exp 3658 JSON artifact."""

    root_path = Path(root)
    output = _repo_path(root_path, Path(output_path))
    artifact = build_artifact(
        root_path,
        corpus_rows=corpus_rows,
        exp3641_artifact=exp3641_artifact,
        score_overrides=score_overrides,
        tests_run=tests_run,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output


def blocked_artifact(
    *,
    root: Path | str,
    started_s: float,
    now_s: float | None,
    reason: str,
    corpus_rows: Sequence[Mapping[str, Any]] | None = None,
    corpus_status: Mapping[str, Any] | None = None,
    verifier_imports: Mapping[str, Any] | None = None,
    tests_run: Sequence[str] | None = None,
) -> JsonDict:
    """Return a terminal blocked or pre-scoring artifact with the required fields."""

    root_path = Path(root)
    rows = [dict(row) for row in (corpus_rows or [])]
    balance = class_balance(rows)
    finished = time.perf_counter() if now_s is None else float(now_s)
    empty_metrics = exp3641.metric_bundle([], [], seeds=BOOTSTRAP_SEEDS)
    status = dict(corpus_status or {})
    artifact: JsonDict = {
        "artifact": ARTIFACT,
        "schema": SCHEMA,
        "run_date": RUN_DATE,
        "honest_verdict": f"complete: {reason}",
        "inference_substrate": INFERENCE_SUBSTRATE,
        "second_code_corpus_name": status.get("selected_corpus_name"),
        "second_code_corpus_path": None,
        "n_examples": len(rows),
        "class_balance": balance,
        "code_verifiers_fire": False,
        "execution_verifier_auroc": empty_metrics,
        "math_signal_code_auroc": empty_metrics,
        "confidence_baseline_auroc": empty_metrics,
        "code_generalization_replicates": False,
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": reproducibility_checksum(rows, [], [], [], {}),
        "duration_s": round(max(0.0, finished - float(started_s)), 6),
        "field_principles": dict(FIELD_PRINCIPLES),
        "acceptance_gate": acceptance_gate(len(rows), False, balance),
        "cross_corpus_consistency": cross_corpus_consistency(
            empty_metrics,
            empty_metrics,
            {},
            False,
        ),
        "corpus_source_status": status,
        "verifier_import_status": dict(verifier_imports or exp3641.import_configured_verifiers()),
        "execution_verifier_summary": {"n_scored": 0, "score_variance": 0.0, "per_verifier": []},
        "source_artifacts": source_artifacts(root_path),
        "tests_run": list(tests_run or []),
        "scripts_research_conductor_modified": False,
        "ops_docs_reconciliation_left_to_conductor": True,
    }
    validate_artifact(artifact)
    return artifact


def build_second_labeled_code_corpus(
    root: Path | str = REPO_ROOT,
    *,
    target_per_class: int = DEFAULT_TARGET_PER_CLASS,
    executor: Executor | None = None,
    timeout_s: float = 1.0,
) -> tuple[list[JsonDict], JsonDict]:
    """Build a balanced HumanEval split using official-check execution labels."""

    root_path = Path(root)
    manifest_path = root_path / HUMANEVAL_MANIFEST_REL_PATH
    rows: list[JsonDict] = []
    n_correct = 0
    n_errors = 0
    scanned = 0
    harness = executor or _default_humaneval_executor
    manifest_rows = exp3641._read_jsonl(manifest_path)
    if not manifest_rows:
        return [], {
            "selected_corpus_name": None,
            "selected_source_path": str(HUMANEVAL_MANIFEST_REL_PATH),
            "fallback_status": "missing_humaneval_manifest",
            "n_candidate_rows": 0,
            "n_manifest_rows_scanned": 0,
        }

    for manifest_row in manifest_rows:
        scanned += 1
        entry_point = str(manifest_row["entry_point"])
        stable_id = str(manifest_row["stable_id"])
        candidate = make_candidate_code(manifest_row)
        problem = {"test": manifest_row["tests"], "entry_point": entry_point}
        if n_correct < target_per_class:
            result = harness(candidate, problem, timeout_s)
            if bool(getattr(result, "passed", False)):
                rows.append(
                    corpus_row(
                        candidate_code=candidate,
                        label=True,
                        test_outcome="candidate_passed",
                        source="humaneval_manifest_canonical",
                        task_id=stable_id,
                        metadata={
                            "candidate_index": 0,
                            "corpus": "HumanEval",
                            "entry_point": entry_point,
                            "manifest_path": str(HUMANEVAL_MANIFEST_REL_PATH),
                            "mutation": "canonical",
                            "stable_id": stable_id,
                        },
                    )
                )
                n_correct += 1
        if n_errors < target_per_class:
            mutant = make_return_none_mutant(candidate, entry_point)
            result = harness(mutant, problem, timeout_s)
            if not bool(getattr(result, "passed", False)):
                rows.append(
                    corpus_row(
                        candidate_code=mutant,
                        label=False,
                        test_outcome=f"candidate_{getattr(result, 'error_type', 'failed')}",
                        source="humaneval_manifest_return_none_mutant",
                        task_id=stable_id,
                        metadata={
                            "candidate_index": 0,
                            "corpus": "HumanEval",
                            "entry_point": entry_point,
                            "error_message": str(getattr(result, "error_message", ""))[:240],
                            "error_type": str(getattr(result, "error_type", "failure")),
                            "manifest_path": str(HUMANEVAL_MANIFEST_REL_PATH),
                            "mutation": "return_none",
                            "stable_id": stable_id,
                        },
                    )
                )
                n_errors += 1
        if n_correct >= target_per_class and n_errors >= target_per_class:
            break

    return rows, {
        "selected_corpus_name": "HumanEval-split",
        "selected_source_path": str(HUMANEVAL_MANIFEST_REL_PATH),
        "fallback_status": "used_cached_humaneval_manifest_canonical_plus_return_none_mutants",
        "n_candidate_rows": len(rows),
        "n_manifest_rows_scanned": scanned,
        "target_per_class": target_per_class,
    }


def _default_humaneval_executor(code: str, problem: dict[str, Any], timeout: float) -> Any:
    from carnot.pipeline.humaneval_live_benchmark import execute_humaneval

    return execute_humaneval(code, problem, timeout=timeout)


def make_candidate_code(manifest_row: Mapping[str, Any]) -> str:
    """Join a HumanEval prompt and canonical solution body into full source."""

    return f"{manifest_row['prompt']}{manifest_row['canonical_solution']}\n"


def make_return_none_mutant(candidate_code: str, entry_point: str) -> str:
    """Insert a deterministic early wrong return into the target function."""

    tree = ast.parse(candidate_code)
    lines = candidate_code.splitlines()
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == entry_point:
            insert_at = node.body[0].lineno - 1 if node.body else node.lineno
            indent = " " * (node.col_offset + 4)
            mutated = lines[:insert_at] + [f"{indent}return None"] + lines[insert_at:]
            return "\n".join(mutated) + "\n"
    raise ValueError(f"entry point not found: {entry_point}")


def corpus_row(
    *,
    candidate_code: str,
    label: bool,
    test_outcome: str,
    source: str,
    task_id: str,
    metadata: Mapping[str, Any],
) -> JsonDict:
    """Normalize one candidate row into the Exp 3658 JSONL schema."""

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
    """Persist the accepted second corpus as deterministic JSONL."""

    output = root / CORPUS_REL_PATH
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "".join(json.dumps(dict(row), sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    return output


def class_balance(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Return class counts and the >=20% balance gate."""

    n_examples = len(rows)
    n_correct = sum(1 for row in rows if bool(row.get("label")))
    n_errors = n_examples - n_correct
    correct_fraction = n_correct / n_examples if n_examples else 0.0
    error_fraction = n_errors / n_examples if n_examples else 0.0
    min_fraction = min(correct_fraction, error_fraction) if n_examples else 0.0
    return {
        "n_correct": n_correct,
        "n_errors": n_errors,
        "correct_fraction": round(correct_fraction, 6),
        "error_fraction": round(error_fraction, 6),
        "min_class_fraction": round(min_fraction, 6),
        "balanced": bool(n_examples >= MIN_EXAMPLES and min_fraction >= MIN_CLASS_FRACTION),
        "minimum_required_fraction": MIN_CLASS_FRACTION,
    }


def acceptance_gate(
    n_examples: int, code_verifiers_fire: bool, balance: Mapping[str, Any]
) -> JsonDict:
    """Compute the user-specified replication acceptance gate."""

    passed = bool(n_examples >= MIN_EXAMPLES and code_verifiers_fire is True and balance)
    return {
        "condition": "n_examples >= 50 AND code_verifiers_fire == true AND class_balance present",
        "passed": passed,
        "principle": "A replication verdict requires the verifiers fired on a balanced second corpus -- a null without that is a wiring failure, not evidence.",
    }


def code_generalization_replicates(
    *,
    code_verifiers_fire: bool,
    balance: Mapping[str, Any],
    math_metrics: Mapping[str, Any],
    confidence_metrics: Mapping[str, Any],
    exp3641_artifact: Mapping[str, Any],
) -> bool:
    """Return the bare replication bool for the balanced second corpus."""

    math_point = math_metrics.get("point")
    confidence_point = confidence_metrics.get("point")
    prior_transfer = bool(
        exp3641_artifact.get("hypothesis_supported") == "transfer"
        or exp3641_artifact.get("transfer_delta_vs_literature", {}).get("meets_lower_anchor")
        is True
        or "transfers_to_code" in str(exp3641_artifact.get("honest_verdict") or "")
    )
    if not (code_verifiers_fire and balance.get("balanced") and prior_transfer):
        return False
    if math_point is None or confidence_point is None:
        return False
    observed_delta = float(math_point) - float(confidence_point)
    return bool(
        float(math_point) > 0.5
        and float(confidence_point) < 0.95
        and observed_delta >= TRANSFER_DELTA_THRESHOLD
    )


def terminal_verdict(
    *,
    n_examples: int,
    balance: Mapping[str, Any],
    code_verifiers_fire: bool,
    replicates: bool,
) -> str:
    """Return one of the required terminal verdict strings."""

    if n_examples < MIN_EXAMPLES or not balance.get("balanced"):
        return "complete: blocked_no_second_code_corpus"
    if not code_verifiers_fire:
        return "complete: code_verifiers_inert_on_second_corpus_diagnosed"
    if replicates:
        return "complete: code_generalization_replicates_on_balanced_second_corpus_claim_hardened"
    return "complete: code_generalization_does_not_replicate_single_corpus_was_artifact"


def cross_corpus_consistency(
    math_metrics: Mapping[str, Any],
    confidence_metrics: Mapping[str, Any],
    exp3641_artifact: Mapping[str, Any],
    replicates: bool,
) -> JsonDict:
    """Summarize consistency with Exp 3641 without reusing its labels."""

    math_point = math_metrics.get("point")
    confidence_point = confidence_metrics.get("point")
    second_delta = (
        None
        if math_point is None or confidence_point is None
        else round(float(math_point) - float(confidence_point), 6)
    )
    prior_delta = exp3641_artifact.get("transfer_delta_vs_literature", {}).get(
        "observed_delta_vs_confidence_baseline"
    )
    return {
        "exp3641_hypothesis_supported": exp3641_artifact.get("hypothesis_supported"),
        "exp3641_math_signal_code_auroc": exp3641_artifact.get("math_signal_code_auroc"),
        "exp3641_confidence_baseline_auroc": exp3641_artifact.get("code_confidence_baseline_auroc"),
        "exp3641_delta_vs_confidence_baseline": prior_delta,
        "second_corpus_delta_vs_confidence_baseline": second_delta,
        "consistent_with_exp3641_transfer_claim": bool(replicates),
    }


def reproducibility_checksum(
    rows: Sequence[Mapping[str, Any]],
    execution_scores: Sequence[float],
    math_scores: Sequence[float],
    confidence_scores: Sequence[float],
    exp3641_artifact: Mapping[str, Any],
) -> str:
    """Return a drift checksum over corpus identities, scores, and prior status."""

    payload = {
        "candidate_sha256": [row.get("candidate_sha256") for row in rows],
        "execution_scores": [round(float(score), 8) for score in execution_scores],
        "math_scores": [round(float(score), 8) for score in math_scores],
        "confidence_scores": [round(float(score), 8) for score in confidence_scores],
        "exp3641_hypothesis_supported": exp3641_artifact.get("hypothesis_supported"),
        "random_seed": RANDOM_SEED,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def source_artifacts(root: Path) -> list[str]:
    """List source files used by the offline workflow."""

    paths = [HUMANEVAL_MANIFEST_REL_PATH, EXP3641_REL_PATH]
    return [str(path) for path in paths if (root / path).exists()]


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Validate the Exp 3658 terminal artifact contract."""

    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    if missing:
        raise ValueError(f"missing required artifact fields: {missing}")
    if type(artifact.get("code_generalization_replicates")) is not bool:
        raise ValueError("code_generalization_replicates must be a bare top-level bool")
    if type(artifact.get("code_verifiers_fire")) is not bool:
        raise ValueError("code_verifiers_fire must be a bare top-level bool")
    if not str(artifact.get("honest_verdict") or "").startswith("complete:"):
        raise ValueError("honest_verdict must start with 'complete:'")
    if not isinstance(artifact.get("n_examples"), int):
        raise ValueError("n_examples must be an int")
    duration = artifact.get("duration_s")
    if not isinstance(duration, (int, float)) or float(duration) < 0.0:
        raise ValueError("duration_s must be a non-negative number")


def _read_exp3641_artifact(root: Path) -> JsonDict:
    return exp3641._read_json_object(root / EXP3641_REL_PATH)


def _repo_path(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path


__all__ = [
    "CORPUS_REL_PATH",
    "EXP3641_REL_PATH",
    "HUMANEVAL_MANIFEST_REL_PATH",
    "OUTPUT_REL_PATH",
    "REQUIRED_ARTIFACT_FIELDS",
    "acceptance_gate",
    "blocked_artifact",
    "build_artifact",
    "build_second_labeled_code_corpus",
    "class_balance",
    "code_generalization_replicates",
    "corpus_row",
    "make_candidate_code",
    "make_return_none_mutant",
    "terminal_verdict",
    "validate_artifact",
    "write_artifact",
]
