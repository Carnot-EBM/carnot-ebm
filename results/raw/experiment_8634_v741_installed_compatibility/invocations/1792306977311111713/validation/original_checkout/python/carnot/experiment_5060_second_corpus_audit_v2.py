#!/usr/bin/env python3
"""Exp 5060: audited D4 second-corpus confirmation v2.

Spec refs: REQ-VERIFY-5060, SCENARIO-VERIFY-5060.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Mapping, Sequence
import hashlib
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

from carnot import moat_benchmark_harness as harness  # noqa: E402
from carnot.moat_benchmark_harness import DEFAULT_RANDOM_SEED, evaluate_verifier  # noqa: E402


JsonDict = dict[str, Any]
JsonMap = Mapping[str, Any]
ScoreFn = Callable[[str, list[str]], list[float]]
Clock = Callable[[], float]

EXPERIMENT_ID = 5060
EXPERIMENT_NAME = "experiment_5060_second_corpus_audit_v2"
SCHEMA = "carnot.experiment_5060_second_corpus_audit_v2.v1"
RESULT_RELATIVE_PATH = "results/experiment_5060_second_corpus_audit_v2.json"
EXP5044_RESULT_RELATIVE_PATH = "results/experiment_5044_second_corpus_candidate_cache.json"
EXP5044_CACHE_RELATIVE_PATH = "results/experiment_5044_second_corpus_candidate_cache.jsonl"
EXP5059_RESULT_RELATIVE_PATH = "results/experiment_5059_d1_sota_refresh_audit.json"
SPEC_REFS = ["REQ-VERIFY-5060", "SCENARIO-VERIFY-5060"]
RANDOM_SEED = DEFAULT_RANDOM_SEED

MANDATED_MODEL_SPECS: dict[str, str] = {
    "flagship_moe": "unsloth/Qwen3.6-35B-A3B-GGUF",
    "flagship_dense": "unsloth/gemma-4-31B-it-GGUF",
    "middle_moe": "unsloth/gemma-4-26B-A4B-it-GGUF",
}

ORACLE_SELECTION_KEYS = frozenset(
    {
        "gold",
        "label",
        "label_correct",
        "candidate_label",
        "solver_verdict",
        "solver_score_used_for_selection",
        "answer_index",
        "answer_choice",
        "model_id",
        "generation_model",
        "scoring_model",
        "source_checkpoint_path",
        "oracle_answer",
    }
)

REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "model_specs",
    "second_corpus_name",
    "second_corpus_confirmed",
    "second_corpus_audit_clean",
    "delta_vs_tuned_sc_second",
    "paired_ci95_second",
    "mcnemar_p_second",
    "n_questions_second",
    "row_hash_manifest",
    "leak_audit_passed",
    "oracle_provenance_passed",
    "duplicate_audit_passed",
    "legacy_models_smoke_only",
)

FIELD_PRINCIPLES: dict[str, dict[str, str]] = {
    "honest_verdict": {
        "principle": (
            "terminal prefix and D4 class: clean confirmation, scoped clue, blocked, "
            "or retired after audit controls."
        )
    },
    "model_specs": {
        "principle": "mandated SOTA GGUF declarations plus Exp5044 and Exp5059 provenance."
    },
    "second_corpus_name": {"principle": "the audited second corpus name from Exp5044."},
    "second_corpus_confirmed": {
        "principle": "true only for a clean D4 confirmation after audits and Exp5059 proper-win gate."
    },
    "second_corpus_audit_clean": {
        "principle": "true iff overlap, duplicate, leak, and oracle-provenance audits all pass."
    },
    "delta_vs_tuned_sc_second": {
        "principle": "Exp5059 executable-arm accuracy minus genuine tuned-SC on Exp5044 rows."
    },
    "paired_ci95_second": {"principle": "paired bootstrap CI95 for verifier minus tuned-SC."},
    "mcnemar_p_second": {"principle": "McNemar exact p for verifier versus tuned-SC."},
    "n_questions_second": {"principle": "number of scored second-corpus questions."},
    "row_hash_manifest": {
        "principle": "stable SHA256 row hashes plus exact/source duplicate receipts."
    },
    "leak_audit_passed": {
        "principle": "true iff scorer-rendered text has no gold outside the candidate answer."
    },
    "oracle_provenance_passed": {
        "principle": "true iff oracle metadata is not available to the selection scorer."
    },
    "duplicate_audit_passed": {
        "principle": "true iff exact row hashes and source-instance ids are unique."
    },
    "legacy_models_smoke_only": {
        "principle": "true; legacy small models are smoke-only and not headline provenance."
    },
}


def json_dumps(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def write_json(path: Path, payload: JsonMap) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_json_object(path: Path) -> JsonDict | None:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return dict(payload) if isinstance(payload, Mapping) else None


def read_jsonl(path: Path) -> list[JsonDict]:
    try:
        lines = Path(path).read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    rows: list[JsonDict] = []
    for line in lines:
        if not line.strip():
            continue
        payload = json.loads(line)
        if isinstance(payload, Mapping):
            rows.append(dict(payload))
    return rows


def number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_") or "second_corpus"


def _delta_label(delta: float | None) -> str:
    if delta is None:
        return "unknown"
    prefix = "plus" if delta >= 0.0 else "minus"
    return f"{prefix}_{abs(delta):.3f}".replace(".", "p")


def _sha256_text(text: str) -> str:
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


def _stable_row_payload(row: JsonMap) -> JsonDict:
    candidates = []
    for candidate in row.get("candidates", []) or []:
        if isinstance(candidate, Mapping):
            candidates.append(
                {
                    "candidate_id": str(candidate.get("candidate_id") or ""),
                    "answer": str(candidate.get("answer") or ""),
                    "cache_index": candidate.get("cache_index"),
                    "temperature": candidate.get("temperature"),
                }
            )
    return {
        "row_id": str(row.get("row_id") or ""),
        "source_row_id": str(row.get("source_row_id") or ""),
        "corpus": str(row.get("corpus") or ""),
        "question": str(row.get("question") or ""),
        "context": str(row.get("context") or ""),
        "gold": str(row.get("gold") or ""),
        "candidates": candidates,
    }


def build_row_hash_manifest(rows: Sequence[JsonMap]) -> JsonDict:
    row_entries: list[JsonDict] = []
    source_counts: Counter[str] = Counter()
    hash_counts: Counter[str] = Counter()
    for index, row in enumerate(rows):
        row_id = str(row.get("row_id") or f"row-{index}")
        source_id = str(row.get("source_row_id") or row_id)
        row_hash = _sha256_text(json_dumps(_stable_row_payload(row)))
        source_counts[source_id] += 1
        hash_counts[row_hash] += 1
        row_entries.append(
            {
                "row_id": row_id,
                "source_row_id": source_id,
                "row_hash": row_hash,
            }
        )
    duplicate_hashes = [
        {"row_hash": row_hash, "count": count}
        for row_hash, count in sorted(hash_counts.items())
        if count > 1
    ]
    duplicate_sources = [
        {"source_row_id": source_id, "count": count}
        for source_id, count in sorted(source_counts.items())
        if count > 1
    ]
    duplicate_source_instances = sum(count - 1 for count in source_counts.values() if count > 1)
    return {
        "algorithm": "sha256",
        "canonicalization": "json_sort_keys_candidate_content_v1",
        "n_rows": len(row_entries),
        "n_unique_row_hashes": len(hash_counts),
        "n_duplicate_row_hashes": sum(count - 1 for count in hash_counts.values() if count > 1),
        "n_duplicate_source_instances": duplicate_source_instances,
        "duplicate_row_hashes": duplicate_hashes,
        "duplicate_source_row_ids": duplicate_sources,
        "rows": row_entries,
        "manifest_hash": _sha256_text(json_dumps(row_entries)),
    }


def duplicate_audit_passed(manifest: JsonMap) -> bool:
    return (
        int(manifest.get("n_duplicate_row_hashes") or 0) == 0
        and int(manifest.get("n_duplicate_source_instances") or 0) == 0
    )


def render_candidate_text(row: JsonMap, candidate: JsonMap) -> str:
    answer = str(candidate.get("answer") or "").strip()
    question = str(row.get("question") or "").strip()
    context = str(row.get("context") or "").strip()[:6000]
    text = f"Candidate answer: {answer}\nQuestion: {question}"
    if context:
        text += f"\nContext:\n{context}"
    return text


def sanitize_rows_for_scoring(rows: Sequence[JsonMap]) -> list[JsonDict]:
    sanitized: list[JsonDict] = []
    for row in rows:
        candidates = []
        for candidate in row.get("candidates", []) or []:
            if not isinstance(candidate, Mapping):
                continue
            clean = {
                str(key): value
                for key, value in dict(candidate).items()
                if str(key) not in ORACLE_SELECTION_KEYS
            }
            clean["answer"] = str(clean.get("answer") or "")
            clean["candidate_id"] = str(clean.get("candidate_id") or "")
            clean["text"] = render_candidate_text(row, clean)
            candidates.append(clean)
        if candidates:
            clean_row = {
                str(key): value
                for key, value in dict(row).items()
                if str(key) not in {"label"}
            }
            clean_row["candidates"] = candidates
            sanitized.append(clean_row)
    return sanitized


def audit_scorer_texts(rows: Sequence[JsonMap]) -> JsonDict:
    failures: list[JsonDict] = []
    for row in rows:
        gold = str(row.get("gold") or "")
        for candidate in row.get("candidates", []) or []:
            text = str(candidate.get("text") or "")
            answer = str(candidate.get("answer") or "")
            checked_text = text.replace(f"Candidate answer: {answer}", "", 1)
            if gold and gold in checked_text:
                failures.append(
                    {
                        "row_id": row.get("row_id"),
                        "candidate_id": candidate.get("candidate_id"),
                        "reason": "gold_outside_candidate_answer",
                    }
                )
    return {
        "passed": not failures,
        "n_failures": len(failures),
        "failures": failures[:20],
    }


def audit_oracle_provenance(raw_rows: Sequence[JsonMap], sanitized_rows: Sequence[JsonMap]) -> JsonDict:
    raw_metadata_fields: set[str] = set()
    solver_score_used = 0
    sanitized_forbidden_fields = 0
    for row in raw_rows:
        if row.get("label") is not None:
            raw_metadata_fields.add("label")
        for candidate in row.get("candidates", []) or []:
            if not isinstance(candidate, Mapping):
                continue
            raw_metadata_fields.update(str(key) for key in candidate if str(key) in ORACLE_SELECTION_KEYS)
            if candidate.get("solver_score_used_for_selection") is True:
                solver_score_used += 1
    for row in sanitized_rows:
        for candidate in row.get("candidates", []) or []:
            if isinstance(candidate, Mapping):
                sanitized_forbidden_fields += len(
                    [key for key in candidate if str(key) in ORACLE_SELECTION_KEYS]
                )
    passed = solver_score_used == 0 and sanitized_forbidden_fields == 0
    return {
        "passed": passed,
        "raw_oracle_metadata_fields_present": sorted(raw_metadata_fields),
        "raw_solver_score_used_for_selection_count": solver_score_used,
        "sanitized_forbidden_field_count": sanitized_forbidden_fields,
        "selection_inputs": "candidate_id_answer_cache_index_temperature_rendered_text_only",
    }


def audit_train_test_overlap(rows: Sequence[JsonMap], exp5059: JsonMap | None) -> JsonDict:
    musr_like = [
        str(row.get("row_id") or "")
        for row in rows
        if "musr" in str(row.get("corpus") or "").lower()
    ]
    return {
        "passed": not musr_like,
        "second_corpus_rows": len(rows),
        "upstream_exp5059_questions": int((exp5059 or {}).get("n_questions") or 0),
        "overlap_row_ids": musr_like[:20],
        "criterion": "second corpus row corpus must be distinct from MuSR training/eval corpus",
    }


def _resolve_cache_path(root: Path, artifact: JsonMap) -> Path:
    raw_path = str(artifact.get("candidate_cache_path") or EXP5044_CACHE_RELATIVE_PATH)
    path = Path(raw_path)
    return path if path.is_absolute() else root / path


def load_second_corpus(root: Path) -> tuple[JsonDict | None, list[JsonDict], Path | None, str | None]:
    artifact = read_json_object(root / EXP5044_RESULT_RELATIVE_PATH)
    if artifact is None:
        return None, [], None, "second_corpus_cache_unavailable"
    if artifact.get("verifier_is_oracle") is not False:
        return artifact, [], None, "second_corpus_oracle_tainted"
    if artifact.get("second_corpus_cache_built") is not True:
        return artifact, [], None, "second_corpus_cache_not_built"
    if artifact.get("headroom_present") is not True:
        return artifact, [], None, "second_corpus_not_headroom_present"
    cache_path = _resolve_cache_path(root, artifact)
    rows = read_jsonl(cache_path)
    if not rows:
        return artifact, [], cache_path, "second_corpus_cache_empty"
    return artifact, rows, cache_path, None


def _checkpoint_from_exp5059(exp5059: JsonMap) -> str:
    scorer = exp5059.get("model_specs", {}).get("powered_d1_scorer", {})
    return str(exp5059.get("checkpoint_path") or scorer.get("checkpoint_path") or "")


def load_exp5059_gate(root: Path) -> tuple[JsonDict | None, JsonDict]:
    exp5059 = read_json_object(root / EXP5059_RESULT_RELATIVE_PATH)
    if exp5059 is None:
        return None, {"available": False, "reason": "exp5059_artifact_unavailable"}
    checkpoint = _checkpoint_from_exp5059(exp5059)
    available = (
        exp5059.get("best_arm_available") is True
        and exp5059.get("verifier_is_oracle") is False
        and bool(checkpoint)
    )
    return exp5059, {
        "available": bool(available),
        "reason": "ok" if available else "exp5059_best_arm_unavailable",
        "checkpoint_path": checkpoint,
        "proper_musr_win": bool(exp5059.get("proper_musr_win")),
        "legacy_models_smoke_only": bool(exp5059.get("legacy_models_smoke_only", True)),
    }


def default_score_fn(checkpoint: str, texts: list[str]) -> list[float]:  # pragma: no cover - live
    from carnot import experiment_5031_lora_ebm_scorer_musr_v3 as d1

    config = d1.TrainingConfig(seed=RANDOM_SEED)
    return list(d1.default_score_fn(config)(checkpoint, texts))


def _energy_by_candidate_id(rows: Sequence[JsonMap], checkpoint: str, score_fn: ScoreFn) -> dict[str, float]:
    candidate_ids: list[str] = []
    texts: list[str] = []
    for row in rows:
        for candidate in row.get("candidates", []) or []:
            candidate_ids.append(str(candidate.get("candidate_id") or ""))
            texts.append(str(candidate.get("text") or ""))
    energies = list(score_fn(checkpoint, texts))
    if len(energies) != len(candidate_ids):
        raise RuntimeError(f"score_fn returned {len(energies)} energies for {len(candidate_ids)} candidates")
    return {candidate_id: float(energy) for candidate_id, energy in zip(candidate_ids, energies)}


def evaluate_second_corpus(
    rows: Sequence[JsonMap],
    *,
    checkpoint: str,
    score_fn: ScoreFn,
    seed: int = RANDOM_SEED,
    bootstrap_samples: int = 2000,
) -> JsonDict:
    energy_by_id = _energy_by_candidate_id(rows, checkpoint, score_fn)

    def scorer(candidate: JsonMap) -> float:
        return energy_by_id.get(str(candidate.get("candidate_id") or ""), math.inf)

    return evaluate_verifier(
        rows,
        scorer=scorer,
        seed=seed,
        bootstrap_samples=bootstrap_samples,
        headroom_threshold=harness.HEADROOM_THRESHOLD,
    )


def _ci_positive(ci95: Any) -> bool:
    return (
        isinstance(ci95, Sequence)
        and len(ci95) == 2
        and number(ci95[0]) is not None
        and number(ci95[1]) is not None
        and float(ci95[0]) > 0.0
        and float(ci95[1]) > 0.0
    )


def _stats_confirm(evaluation: JsonMap) -> bool:
    delta = number(evaluation.get("verifier_minus_tuned_sc_delta"))
    p_value = number(evaluation.get("mcnemar_p"))
    return (
        evaluation.get("headroom_present") is True
        and delta is not None
        and delta > 0.0
        and _ci_positive(evaluation.get("verifier_minus_tuned_sc_ci95"))
        and p_value is not None
        and p_value < 0.05
    )


def _model_specs(second_corpus: JsonMap | None, exp5059: JsonMap | None, gate: JsonMap) -> JsonDict:
    return {
        "mandated_sota": dict(MANDATED_MODEL_SPECS),
        "second_corpus": dict((second_corpus or {}).get("model_specs") or {}),
        "exp5059_best_executable_arm": {
            "source": EXP5059_RESULT_RELATIVE_PATH,
            "best_arm_available": bool(gate.get("available")),
            "checkpoint_path": gate.get("checkpoint_path"),
            "proper_musr_win": bool(gate.get("proper_musr_win")),
            "scorer_source": dict((exp5059 or {}).get("scorer_source") or {}),
        },
    }


def _empty_manifest() -> JsonDict:
    return build_row_hash_manifest([])


def _checksum(artifact: JsonMap) -> str:
    basis = {
        "experiment_id": artifact.get("experiment_id"),
        "honest_verdict": artifact.get("honest_verdict"),
        "second_corpus_name": artifact.get("second_corpus_name"),
        "d4_verdict_class": artifact.get("d4_verdict_class"),
        "delta_vs_tuned_sc_second": artifact.get("delta_vs_tuned_sc_second"),
        "paired_ci95_second": artifact.get("paired_ci95_second"),
        "row_hash_manifest": {
            "manifest_hash": (artifact.get("row_hash_manifest") or {}).get("manifest_hash"),
            "n_rows": (artifact.get("row_hash_manifest") or {}).get("n_rows"),
            "n_duplicate_source_instances": (artifact.get("row_hash_manifest") or {}).get(
                "n_duplicate_source_instances"
            ),
        },
    }
    return _sha256_text(json_dumps(basis))


def _base_artifact(
    *,
    root: Path,
    artifact_path: Path,
    honest_verdict: str,
    d4_verdict_class: str,
    second_corpus: JsonMap | None,
    exp5059: JsonMap | None,
    gate: JsonMap,
    cache_path: Path | None,
    row_hash_manifest: JsonMap | None = None,
    leak_receipt: JsonMap | None = None,
    oracle_receipt: JsonMap | None = None,
    duplicate_passed: bool = True,
    train_test_receipt: JsonMap | None = None,
    duration_s: float = 0.0,
    blocked_error: str | None = None,
) -> JsonDict:
    leak = dict(leak_receipt or {"passed": False, "n_failures": 0, "failures": []})
    oracle = dict(oracle_receipt or {"passed": False})
    train_test = dict(train_test_receipt or {"passed": False})
    audit_clean = bool(
        leak.get("passed")
        and oracle.get("passed")
        and duplicate_passed
        and train_test.get("passed")
    )
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT_NAME,
        "experiment_id": EXPERIMENT_ID,
        "spec_refs": list(SPEC_REFS),
        "result_path": artifact_path.as_posix(),
        "honest_verdict": honest_verdict,
        "d4_verdict_class": d4_verdict_class,
        "model_specs": _model_specs(second_corpus, exp5059, gate),
        "second_corpus_name": (second_corpus or {}).get("second_corpus_name"),
        "second_corpus_confirmed": False,
        "second_corpus_audit_clean": audit_clean,
        "genuine_sc_accuracy_second": None,
        "accuracy_breakdown_second": {},
        "delta_vs_tuned_sc_second": None,
        "paired_ci95_second": None,
        "mcnemar_p_second": None,
        "n_questions_second": 0,
        "headroom_present": bool((second_corpus or {}).get("headroom_present")),
        "oracle_at_k_second": None,
        "oracle_k_second": 0,
        "row_hash_manifest": dict(row_hash_manifest or _empty_manifest()),
        "leak_audit_passed": bool(leak.get("passed")),
        "oracle_provenance_passed": bool(oracle.get("passed")),
        "duplicate_audit_passed": bool(duplicate_passed),
        "train_test_overlap_passed": bool(train_test.get("passed")),
        "legacy_models_smoke_only": bool(gate.get("legacy_models_smoke_only", True)),
        "upstream_exp5059_proper_win": bool(gate.get("proper_musr_win")),
        "no_oracle_status": {
            "verifier_is_oracle": False,
            "selection_inputs": oracle.get("selection_inputs"),
            "guard": "sanitized_candidate_text_plus_shared_harness_guard",
        },
        "leak_audit_receipt": leak,
        "oracle_provenance_receipt": oracle,
        "train_test_overlap_receipt": train_test,
        "candidate_cache_path": cache_path.as_posix() if cache_path else None,
        "source_artifacts": {
            "second_corpus": EXP5044_RESULT_RELATIVE_PATH,
            "best_executable_arm": EXP5059_RESULT_RELATIVE_PATH,
        },
        "inference_substrate": "deterministic_verifier",
        "random_seed": RANDOM_SEED,
        "duration_s": round(max(0.0, float(duration_s)), 6),
        "field_principles": dict(FIELD_PRINCIPLES),
        "reproducibility_checksum": "",
        "repo_root": root.as_posix(),
    }
    if blocked_error:
        artifact["blocked_error"] = blocked_error[:1000]
    artifact["reproducibility_checksum"] = _checksum(artifact)
    return artifact


def _final_verdict_class(*, audit_clean: bool, stats_confirm: bool, upstream_proper_win: bool) -> str:
    if not audit_clean:
        return "retired"
    if stats_confirm and upstream_proper_win:
        return "clean_confirmation"
    return "scoped_clue"


def _honest_verdict(verdict_class: str, corpus: str | None, delta: float | None) -> str:
    corpus_slug = _slug(str(corpus or "second_corpus"))
    delta_label = _delta_label(delta)
    if verdict_class == "clean_confirmation":
        return f"success_d4_clean_confirmation_{corpus_slug}_{delta_label}"
    if verdict_class == "retired":
        return f"retired_d4_second_corpus_audit_failed_{corpus_slug}_{delta_label}"
    return f"complete_d4_scoped_clue_{corpus_slug}_{delta_label}"


def _complete_artifact(
    *,
    root: Path,
    artifact_path: Path,
    second_corpus: JsonMap,
    exp5059: JsonMap,
    gate: JsonMap,
    cache_path: Path,
    row_hash_manifest: JsonMap,
    leak_receipt: JsonMap,
    oracle_receipt: JsonMap,
    train_test_receipt: JsonMap,
    evaluation: JsonMap,
    duration_s: float,
) -> JsonDict:
    duplicate_passed = duplicate_audit_passed(row_hash_manifest)
    audit_clean = bool(
        leak_receipt.get("passed")
        and oracle_receipt.get("passed")
        and duplicate_passed
        and train_test_receipt.get("passed")
    )
    stats_confirm = _stats_confirm(evaluation)
    verdict_class = _final_verdict_class(
        audit_clean=audit_clean,
        stats_confirm=stats_confirm,
        upstream_proper_win=bool(gate.get("proper_musr_win")),
    )
    tuned = dict(evaluation.get("tuned_self_consistency") or {})
    verifier = dict(evaluation.get("verifier") or {})
    delta = float(evaluation.get("verifier_minus_tuned_sc_delta") or 0.0)
    artifact = _base_artifact(
        root=root,
        artifact_path=artifact_path,
        honest_verdict=_honest_verdict(
            verdict_class,
            str(second_corpus.get("second_corpus_name") or "second_corpus"),
            delta,
        ),
        d4_verdict_class=verdict_class,
        second_corpus=second_corpus,
        exp5059=exp5059,
        gate=gate,
        cache_path=cache_path,
        row_hash_manifest=row_hash_manifest,
        leak_receipt=leak_receipt,
        oracle_receipt=oracle_receipt,
        duplicate_passed=duplicate_passed,
        train_test_receipt=train_test_receipt,
        duration_s=duration_s,
    )
    artifact.update(
        {
            "second_corpus_confirmed": verdict_class == "clean_confirmation",
            "second_corpus_audit_clean": audit_clean,
            "genuine_sc_accuracy_second": float(tuned.get("accuracy") or 0.0),
            "accuracy_breakdown_second": {
                "genuine_tuned_sc": float(tuned.get("accuracy") or 0.0),
                "verifier": float(verifier.get("accuracy") or 0.0),
            },
            "delta_vs_tuned_sc_second": round(delta, 6),
            "paired_ci95_second": [
                float(value)
                for value in evaluation.get("verifier_minus_tuned_sc_ci95", [0.0, 0.0])
            ],
            "mcnemar_p_second": float(evaluation.get("mcnemar_p") or 0.0),
            "n_questions_second": int(evaluation.get("n_rows") or 0),
            "headroom_present": bool(evaluation.get("headroom_present")),
            "oracle_at_k_second": float(evaluation.get("oracle_at_k") or 0.0),
            "oracle_k_second": int(evaluation.get("oracle_k") or 0),
            "evaluation": evaluation,
        }
    )
    artifact["reproducibility_checksum"] = _checksum(artifact)
    return artifact


def run(
    *,
    root: Path = REPO_ROOT,
    artifact_path: Path | None = None,
    score_fn: ScoreFn = default_score_fn,
    bootstrap_samples: int = 2000,
    seed: int = RANDOM_SEED,
    now: Clock = time.perf_counter,
    write: bool = True,
) -> JsonDict:
    root = Path(root)
    artifact_path = Path(artifact_path) if artifact_path else root / RESULT_RELATIVE_PATH
    start = float(now())
    second_corpus, raw_rows, cache_path, cache_error = load_second_corpus(root)
    if cache_error is not None:
        artifact = _base_artifact(
            root=root,
            artifact_path=artifact_path,
            honest_verdict=f"blocked_{cache_error}",
            d4_verdict_class="blocked",
            second_corpus=second_corpus,
            exp5059=None,
            gate={"available": False, "legacy_models_smoke_only": True},
            cache_path=cache_path,
            duration_s=float(now()) - start,
            blocked_error=cache_error,
        )
        if write:
            write_json(artifact_path, artifact)
        return artifact

    assert second_corpus is not None
    assert cache_path is not None
    sanitized_rows = sanitize_rows_for_scoring(raw_rows)
    row_manifest = build_row_hash_manifest(raw_rows)
    leak_receipt = audit_scorer_texts(sanitized_rows)
    oracle_receipt = audit_oracle_provenance(raw_rows, sanitized_rows)
    exp5059, gate = load_exp5059_gate(root)
    train_test_receipt = audit_train_test_overlap(raw_rows, exp5059)
    if not gate.get("available"):
        artifact = _base_artifact(
            root=root,
            artifact_path=artifact_path,
            honest_verdict="blocked_exp5059_best_arm_unavailable",
            d4_verdict_class="blocked",
            second_corpus=second_corpus,
            exp5059=exp5059,
            gate=gate,
            cache_path=cache_path,
            row_hash_manifest=row_manifest,
            leak_receipt=leak_receipt,
            oracle_receipt=oracle_receipt,
            duplicate_passed=duplicate_audit_passed(row_manifest),
            train_test_receipt=train_test_receipt,
            duration_s=float(now()) - start,
            blocked_error=str(gate.get("reason") or "exp5059_best_arm_unavailable"),
        )
        if write:
            write_json(artifact_path, artifact)
        return artifact

    try:
        evaluation = evaluate_second_corpus(
            sanitized_rows,
            checkpoint=str(gate.get("checkpoint_path") or ""),
            score_fn=score_fn,
            seed=seed,
            bootstrap_samples=bootstrap_samples,
        )
        artifact = _complete_artifact(
            root=root,
            artifact_path=artifact_path,
            second_corpus=second_corpus,
            exp5059=exp5059 or {},
            gate=gate,
            cache_path=cache_path,
            row_hash_manifest=row_manifest,
            leak_receipt=leak_receipt,
            oracle_receipt=oracle_receipt,
            train_test_receipt=train_test_receipt,
            evaluation=evaluation,
            duration_s=float(now()) - start,
        )
    except Exception as exc:  # pragma: no cover - defensive local scoring guard
        artifact = _base_artifact(
            root=root,
            artifact_path=artifact_path,
            honest_verdict="blocked_second_corpus_scoring_unavailable",
            d4_verdict_class="blocked",
            second_corpus=second_corpus,
            exp5059=exp5059,
            gate=gate,
            cache_path=cache_path,
            row_hash_manifest=row_manifest,
            leak_receipt=leak_receipt,
            oracle_receipt=oracle_receipt,
            duplicate_passed=duplicate_audit_passed(row_manifest),
            train_test_receipt=train_test_receipt,
            duration_s=float(now()) - start,
            blocked_error=f"{type(exc).__name__}: {exc}",
        )
    if write:
        write_json(artifact_path, artifact)
    return artifact


def artifact_schema_errors(artifact: JsonMap) -> list[str]:
    errors: list[str] = []
    for field in REQUIRED_ARTIFACT_FIELDS:
        if field not in artifact:
            errors.append(field)
    if artifact.get("schema") != SCHEMA:
        errors.append("schema")
    if artifact.get("spec_refs") != SPEC_REFS:
        errors.append("spec_refs")
    if not isinstance(artifact.get("model_specs"), Mapping):
        errors.append("model_specs")
    if not isinstance(artifact.get("row_hash_manifest"), Mapping):
        errors.append("row_hash_manifest")
    if artifact.get("d4_verdict_class") not in {
        "clean_confirmation",
        "scoped_clue",
        "blocked",
        "retired",
    }:
        errors.append("d4_verdict_class")
    for field in (
        "second_corpus_confirmed",
        "second_corpus_audit_clean",
        "leak_audit_passed",
        "oracle_provenance_passed",
        "duplicate_audit_passed",
        "legacy_models_smoke_only",
    ):
        if not isinstance(artifact.get(field), bool):
            errors.append(field)
    if artifact.get("legacy_models_smoke_only") is not True:
        errors.append("legacy_models_smoke_only")
    if not isinstance(artifact.get("n_questions_second"), int) or int(
        artifact.get("n_questions_second", -1)
    ) < 0:
        errors.append("n_questions_second")
    delta = artifact.get("delta_vs_tuned_sc_second")
    if delta is not None and not isinstance(delta, (int, float)):
        errors.append("delta_vs_tuned_sc_second")
    ci95 = artifact.get("paired_ci95_second")
    if ci95 is not None and (
        not isinstance(ci95, list)
        or len(ci95) != 2
        or not all(isinstance(value, (int, float)) for value in ci95)
    ):
        errors.append("paired_ci95_second")
    p_value = artifact.get("mcnemar_p_second")
    if p_value is not None and not (
        isinstance(p_value, (int, float)) and 0.0 <= float(p_value) <= 1.0
    ):
        errors.append("mcnemar_p_second")
    if not str(artifact.get("honest_verdict") or "").startswith(
        ("blocked_", "complete_", "success_", "retired_")
    ):
        errors.append("honest_verdict")
    return sorted(set(errors))


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - CLI entrypoint
    _ = argv
    artifact = run()
    print(
        json.dumps(
            {
                "result_path": str(REPO_ROOT / RESULT_RELATIVE_PATH),
                "honest_verdict": artifact.get("honest_verdict"),
                "d4_verdict_class": artifact.get("d4_verdict_class"),
                "second_corpus_confirmed": artifact.get("second_corpus_confirmed"),
                "second_corpus_audit_clean": artifact.get("second_corpus_audit_clean"),
                "delta_vs_tuned_sc_second": artifact.get("delta_vs_tuned_sc_second"),
            },
            sort_keys=True,
        )
    )
    errors = artifact_schema_errors(artifact)
    if errors:
        print(f"schema_errors={errors}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main(sys.argv[1:]))
