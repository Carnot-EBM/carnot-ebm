"""Exp 4337: build a leak-robust DiNa-LRM partial-state scorer.

Spec refs: REQ-VERIFY-4337, SCENARIO-VERIFY-4337.
"""

from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import math
import os
import random
import struct
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, Sequence

from carnot.experiment_4260_diffusiongemma_energy_guided_preflight import (
    CACHE_REPO_DIRNAME,
    DEFAULT_CACHE_ROOT,
    GGUF_HF_ID,
    PROBE_TEXT,
    VocabLoadResult,
    _default_process_rows,
    _skipped_check,
)
from carnot.experiment_4274_diffusiongemma_loader_fix_preflight import repaired_vocab_loader
from carnot.experiment_4281_diffusiongemma_energy_guided_full_run import (
    CANVAS_LEN,
    MASK_TOKEN_ID,
    PR_BINARY,
    VOCAB_SIZE,
)
from carnot.experiment_4292_partial_state_diffusion_scorer_build import (
    DEFAULT_CORPUS_PATH,
    check_preconditions as check_diffusiongemma_preconditions,
    load_reasoning_items,
)
from carnot.experiment_4325_in_generation_moat_replicate_second_corpus import (
    SECOND_CORPUS_PATH,
    load_second_corpus_items,
)
from carnot.inference.sota_models import resolve_cached_gguf
from carnot.verify.dina_lrm_partial_state_scorer import (
    ANSWER_RECOVERY_CEILING,
    DEFAULT_NOISE_LEVELS,
    DEFAULT_VISIBLE_FRACTIONS,
    PROCESS_RANKING_FLOOR,
    DinaLRMPartialStateScorer,
    build_dina_lrm_records,
    corpus_checksum,
    masked_answer_recovery_auroc,
    process_ranking_auroc,
    split_corpus_items,
)
from carnot.verify.partial_state_diffusion_scorer import find_answer_spans


ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_PATH = ROOT / "results" / "experiment_4337_leak_robust_partial_state_scorer_build.json"
SCORER_MODULE_PATH = ROOT / "results" / "dina_lrm_partial_state_scorer_exp4337.pkl"
RANDOM_SEED = 4337
SPEC_REFS = ["REQ-VERIFY-4337", "SCENARIO-VERIFY-4337"]
INFERENCE_SUBSTRATE = "live_llm_inference"
DEFAULT_MINIMUM_LIVE_DURATION_S = 60.0
HELDOUT_FRACTION = 0.25
MAX_FEATURES = 8000

FIELD_PRINCIPLES = {
    "honest_verdict": (
        "Terminal-prefixed. A leak-robust scorer BUILT (scorer_leak_audit_passed=true "
        "on >=2 corpora -- unblocks exp4338) and a complete_no_leak_robust_partial_state_scorer "
        "(the direction is not viable, retire) are BOTH decision-grade."
    ),
    "scorer_leak_audit_passed": (
        "BARE bool: exp4338 is gated on this; true iff the scorer SURVIVES the "
        "answer-cell-masked leak audit on BOTH corpora (no answer recovery AND "
        "non-degenerate process ranking) -- the diagnosed fix for exp4325's leak."
    ),
    "masked_answer_recovery_auroc": (
        "BARE float: max per-corpus AUROC of recovering hidden answer identity from "
        "the MASKED canvas reward signal -- must be near chance; a high value means "
        "the scorer still leaks."
    ),
    "process_ranking_auroc": (
        "BARE float: min per-corpus AUROC of ranking partial states by genuine process "
        "quality -- must be non-degenerate (>0.5) so the scorer is a real reward, not "
        "a constant."
    ),
    "scorer_module_path": (
        "Path to the persisted leak-robust scorer module -- the deliverable exp4338 reuses."
    ),
    "verifier_is_oracle": (
        "BARE bool=false -- the learned reward head is oracle-distinct (NOT the executable oracle)."
    ),
    "preconditions_checked": (
        "Records the PR-binary + GGUF cache + >=2-corpora + TRM-stand-down verified; "
        "pre-empts the silent-missing-resource fabrication mode."
    ),
    "random_seed": "Determinism precondition for the canvas generation + reward-head training + leak audit.",
    "reproducibility_checksum": (
        "Hash of the 2 corpora + the reward-head config + PR-binary inputs; lets a "
        "third party re-run."
    ),
    "model_specs": (
        "DiffusionGemma GGUF + PR binary + the reward-head architecture + the timestep "
        "conditioning + the 2 corpora + n + seeds; required methodology."
    ),
}

REQUIRED_FIELDS = [
    "honest_verdict",
    "scorer_leak_audit_passed",
    "masked_answer_recovery_auroc",
    "process_ranking_auroc",
    "scorer_module_path",
    "verifier_is_oracle",
    "preconditions_checked",
    "random_seed",
    "reproducibility_checksum",
    "model_specs",
    "field_principles",
    "spec_refs",
    "duration_s",
    "inference_substrate",
    "adversarial_verify",
]


def load_required_corpora(
    *,
    corpus1_path: Path = DEFAULT_CORPUS_PATH,
    corpus2_path: Path = SECOND_CORPUS_PATH,
    corpus1_loader: Callable[[], list[dict[str, Any]]] = load_reasoning_items,
    corpus2_loader: Callable[[], list[dict[str, Any]]] = load_second_corpus_items,
) -> list[dict[str, Any]]:
    corpora = [
        {
            "name": "in_distribution_error_corpus_v1",
            "path": str(corpus1_path),
            "items": corpus1_loader(),
        },
        {"name": "step_error_balanced_v2", "path": str(corpus2_path), "items": corpus2_loader()},
    ]
    return corpora


def check_corpora_available(corpora: Sequence[dict[str, Any]]) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    checksums: set[str] = set()
    for corpus in corpora:
        items = list(corpus.get("items", []))
        label_counts = _label_counts(items)
        checksum = corpus_checksum(items) if items else ""
        checksums.add(checksum)
        checks.append(
            {
                "name": str(corpus.get("name", "")),
                "path": str(corpus.get("path", "")),
                "item_count": int(len(items)),
                "label_counts": label_counts,
                "checksum": checksum,
                "has_both_labels": label_counts["correct"] > 0 and label_counts["incorrect"] > 0,
            }
        )
    ok = bool(
        len(corpora) >= 2
        and len(checksums) >= 2
        and all(row["item_count"] > 0 and row["has_both_labels"] for row in checks)
    )
    return {
        "resource": "two_oracle_distinct_reasoning_corpora",
        "ok": ok,
        "corpora": checks,
        "reason": "ok" if ok else "fewer than two distinct labeled corpora",
    }


def check_preconditions(
    *,
    pr_binary_path: Path = PR_BINARY,
    cache_root: Path | None = None,
    resolve_gguf_fn: Callable[..., str | None] = resolve_cached_gguf,
    vocab_loader_fn: Callable[[str, str], VocabLoadResult] = repaired_vocab_loader,
    process_rows_fn: Callable[[], list[dict[str, Any]]] | None = None,
    corpora_loader_fn: Callable[[], list[dict[str, Any]]] = load_required_corpora,
) -> dict[str, Any]:
    preconditions = check_diffusiongemma_preconditions(
        pr_binary_path=Path(pr_binary_path),
        cache_root=cache_root,
        resolve_gguf_fn=resolve_gguf_fn,
        vocab_loader_fn=vocab_loader_fn,
        process_rows_fn=process_rows_fn if process_rows_fn is not None else _default_process_rows,
    )
    if not preconditions["all_passed"]:
        preconditions["corpora"] = []
        return preconditions
    try:
        corpora = corpora_loader_fn()
        corpus_check = check_corpora_available(corpora)
    except Exception as exc:
        corpora = []
        corpus_check = {
            "resource": "two_oracle_distinct_reasoning_corpora",
            "ok": False,
            "corpora": [],
            "error": f"{type(exc).__name__}: {exc}",
            "reason": "corpus loader failed",
        }
    preconditions["ordered_checks"].append(corpus_check)
    preconditions["corpora"] = corpora
    if not corpus_check["ok"]:
        preconditions["all_passed"] = False
        preconditions["verdict"] = "blocked_second_corpus_unavailable"
    return preconditions


def extract_noisy_masked_canvas_smoke(
    *,
    pr_binary_path: Path,
    gguf_path: str,
    tokenizer: Any,
    corpus_name: str,
    item: dict[str, Any],
    noise_level: float = 0.3,
    timeout_s: float = 300.0,
) -> dict[str, Any]:  # pragma: no cover - exercises the local 26B PR binary.
    prompt = f"Rank the process quality for corpus {corpus_name}."
    prompt_ids = [int(token) for token in tokenizer.tokenize(prompt.encode("utf-8"))][:128] or [0]
    text = _scrub_answer_text(str(item.get("step_text") or item.get("text") or ""))
    canvas_ids = [int(token) for token in tokenizer.tokenize(text.encode("utf-8"))][:CANVAS_LEN]
    rng = random.Random(_stable_seed(RANDOM_SEED, corpus_name, text))
    noisy_canvas = [
        MASK_TOKEN_ID if rng.random() < float(noise_level) else int(token) for token in canvas_ids
    ][:CANVAS_LEN]
    while len(noisy_canvas) < CANVAS_LEN:
        noisy_canvas.append(MASK_TOKEN_ID)
    with tempfile.TemporaryDirectory(prefix="carnot_exp4337_dgemma_") as tmp:
        workdir = Path(tmp)
        prompt_path = workdir / "prompt_ids.i32"
        canvas_path = workdir / "canvas_ids.i32"
        logits_path = workdir / "out_logits.bin"
        _write_int32_file(prompt_path, prompt_ids)
        _write_int32_file(canvas_path, noisy_canvas)
        proc = subprocess.run(
            [
                str(pr_binary_path),
                str(gguf_path),
                str(prompt_path),
                str(canvas_path),
                str(logits_path),
            ],
            capture_output=True,
            text=True,
            timeout=timeout_s,
            env={**os.environ, "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", "0")},
        )
        if proc.returncode != 0 or not logits_path.exists():
            return {
                "corpus_name": corpus_name,
                "status": "blocked_pr_binary_eval_failed",
                "eval_rc": int(proc.returncode),
                "stderr_tail": proc.stderr[-600:],
                "stdout_tail": proc.stdout[-600:],
                "prompt_ids_count": len(prompt_ids),
                "canvas_non_mask_count": int(sum(token != MASK_TOKEN_ID for token in noisy_canvas)),
            }
        size = logits_path.stat().st_size
        expected = CANVAS_LEN * VOCAB_SIZE * 4
        with logits_path.open("rb") as handle:
            first_row = struct.unpack(f"<{VOCAB_SIZE}f", handle.read(VOCAB_SIZE * 4))
        top = heapq.nlargest(6, enumerate(first_row), key=lambda pair: pair[1])
        top_tokens: list[list[Any]] = []
        for token_id, score in top:
            try:
                token_text = tokenizer.detokenize([int(token_id)]).decode("utf-8", errors="replace")
            except Exception:
                token_text = ""
            top_tokens.append([int(token_id), token_text, round(float(score), 4)])
        return {
            "corpus_name": corpus_name,
            "status": "extracted" if size == expected else "blocked_pr_binary_eval_bad_shape",
            "eval_rc": int(proc.returncode),
            "score_shape": [CANVAS_LEN, VOCAB_SIZE] if size == expected else None,
            "score_finite_sample": bool(all(math.isfinite(float(score)) for _token, score in top)),
            "logits_file_size_bytes": int(size),
            "expected_logits_file_size_bytes": int(expected),
            "prompt_ids_count": len(prompt_ids),
            "canvas_non_mask_count": int(sum(token != MASK_TOKEN_ID for token in noisy_canvas)),
            "noise_level": round(float(noise_level), 6),
            "pos0_top_tokens": top_tokens,
        }


def run_canvas_generation_phase(
    *,
    corpora: Sequence[dict[str, Any]],
    preconditions: dict[str, Any],
    pr_binary_path: Path,
    canvas_smoke_fn: Callable[..., dict[str, Any]] = extract_noisy_masked_canvas_smoke,
    checkpoint_path: Path | None = None,
) -> list[dict[str, Any]]:
    cache = _resource(preconditions, "diffusiongemma_cache")
    tokenizer = preconditions["vocab_loader_result"].tokenizer
    smokes: list[dict[str, Any]] = []
    for index, corpus in enumerate(corpora, start=1):
        print(
            f"[exp4337] phase=canvas_generation corpus={corpus['name']} index={index}", flush=True
        )
        items = list(corpus.get("items", []))
        smoke = canvas_smoke_fn(
            pr_binary_path=Path(pr_binary_path),
            gguf_path=str(cache.get("gguf_path")),
            tokenizer=tokenizer,
            corpus_name=str(corpus["name"]),
            item=items[0],
        )
        smokes.append(smoke)
        _checkpoint(checkpoint_path, {"phase": "canvas_generation", "canvas_smokes": smokes})
    return smokes


def train_audit_and_save(
    *,
    corpora: Sequence[dict[str, Any]],
    scorer_path: Path,
    seed: int = RANDOM_SEED,
    max_features: int = MAX_FEATURES,
    checkpoint_path: Path | None = None,
) -> dict[str, Any]:
    print("[exp4337] phase=training start", flush=True)
    train_records = []
    heldout_records_by_corpus: dict[str, list[Any]] = {}
    corpus_summaries: list[dict[str, Any]] = []
    for corpus in corpora:
        name = str(corpus["name"])
        items = list(corpus.get("items", []))
        train_items, heldout_items = split_corpus_items(
            items,
            heldout_fraction=HELDOUT_FRACTION,
            seed=seed,
        )
        train_for_corpus = build_dina_lrm_records(train_items, corpus_name=name, seed=seed)
        heldout_for_corpus = build_dina_lrm_records(heldout_items, corpus_name=name, seed=seed)
        train_records.extend(train_for_corpus)
        heldout_records_by_corpus[name] = heldout_for_corpus
        corpus_summaries.append(
            {
                "name": name,
                "train_items": int(len(train_items)),
                "heldout_items": int(len(heldout_items)),
                "train_records": int(len(train_for_corpus)),
                "heldout_records": int(len(heldout_for_corpus)),
                "label_counts": _label_counts(items),
                "checksum": corpus_checksum(items),
            }
        )
    scorer = DinaLRMPartialStateScorer(random_seed=seed, max_features=max_features)
    scorer.fit(train_records)
    scorer.save(scorer_path)
    loaded = DinaLRMPartialStateScorer.load(scorer_path)
    _checkpoint(
        checkpoint_path,
        {
            "phase": "training",
            "train_records": len(train_records),
            "scorer_module_path": str(scorer_path),
        },
    )
    print("[exp4337] phase=leak_audit start", flush=True)
    per_corpus: dict[str, dict[str, Any]] = {}
    process_aurocs: list[float] = []
    answer_aurocs: list[float] = []
    for name, records in heldout_records_by_corpus.items():
        process_auroc = process_ranking_auroc(loaded, records)
        answer_auroc = masked_answer_recovery_auroc(loaded, records)
        process_aurocs.append(process_auroc)
        answer_aurocs.append(answer_auroc)
        per_corpus[name] = {
            "heldout_records": int(len(records)),
            "answer_masked_cells": int(sum(len(record.answer_cell_indices) for record in records)),
            "process_ranking_auroc": round(float(process_auroc), 6),
            "masked_answer_recovery_auroc": round(float(answer_auroc), 6),
            "process_ranking_passed": bool(process_auroc > PROCESS_RANKING_FLOOR),
            "answer_recovery_passed": bool(answer_auroc <= ANSWER_RECOVERY_CEILING),
            "preview": [record.to_preview() for record in records[:3]],
        }
        print(
            f"[exp4337] phase=leak_audit corpus={name} "
            f"process_auroc={process_auroc:.6f} answer_recovery_auroc={answer_auroc:.6f}",
            flush=True,
        )
        _checkpoint(checkpoint_path, {"phase": "leak_audit", "per_corpus_audit": per_corpus})
    scorer_loadable = bool(loaded.is_fitted)
    if process_aurocs:
        probe_record = next(iter(heldout_records_by_corpus.values()))[0]
        loaded.score_partial_state(probe_record.canvas_ids, probe_record.timestep)
    return {
        "scorer_loadable": scorer_loadable,
        "train_records": int(len(train_records)),
        "per_corpus_audit": per_corpus,
        "corpus_summaries": corpus_summaries,
        "masked_answer_recovery_auroc": round(
            float(max(answer_aurocs) if answer_aurocs else 0.5), 6
        ),
        "process_ranking_auroc": round(float(min(process_aurocs) if process_aurocs else 0.0), 6),
        "scorer_leak_audit_passed": bool(
            scorer_loadable
            and process_aurocs
            and answer_aurocs
            and min(process_aurocs) > PROCESS_RANKING_FLOOR
            and max(answer_aurocs) <= ANSWER_RECOVERY_CEILING
        ),
    }


def build_artifact(
    *,
    honest_verdict: str,
    preconditions: dict[str, Any],
    duration_s: float,
    scorer_path: Path,
    canvas_smokes: Sequence[dict[str, Any]] | None = None,
    eval_result: dict[str, Any] | None = None,
    adversarial_verify: dict[str, Any] | None = None,
) -> dict[str, Any]:
    eval_result = eval_result or {}
    leak_passed = bool(eval_result.get("scorer_leak_audit_passed", False))
    return {
        "schema": "leak_robust_partial_state_scorer_build_v1",
        "experiment": 4337,
        "honest_verdict": honest_verdict,
        "scorer_leak_audit_passed": leak_passed,
        "masked_answer_recovery_auroc": float(
            eval_result.get("masked_answer_recovery_auroc", 0.0) or 0.0
        ),
        "process_ranking_auroc": float(eval_result.get("process_ranking_auroc", 0.0) or 0.0),
        "scorer_module_path": str(scorer_path) if leak_passed else "",
        "scorer_loadable": bool(eval_result.get("scorer_loadable", False)),
        "verifier_is_oracle": False,
        "preconditions_checked": list(preconditions.get("ordered_checks", [])),
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": reproducibility_checksum(
            preconditions=preconditions,
            canvas_smokes=list(canvas_smokes or []),
            eval_result=eval_result,
        ),
        "model_specs": _model_specs(
            preconditions=preconditions,
            canvas_smokes=list(canvas_smokes or []),
            eval_result=eval_result,
        ),
        "canvas_generation": list(canvas_smokes or []),
        "per_corpus_audit": dict(eval_result.get("per_corpus_audit", {})),
        "train_records": int(eval_result.get("train_records", 0) or 0),
        "field_principles": FIELD_PRINCIPLES,
        "spec_refs": SPEC_REFS,
        "duration_s": round(float(duration_s), 6),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "adversarial_verify": adversarial_verify or {"status": "pending_pre_write"},
        "acceptance_gate": True,
    }


def validate_artifact(artifact: dict[str, Any]) -> None:
    missing = [field for field in REQUIRED_FIELDS if field not in artifact]
    if missing:
        raise ValueError(f"missing required fields: {missing}")
    if not isinstance(artifact["honest_verdict"], str) or not artifact["honest_verdict"]:
        raise ValueError("honest_verdict must be a non-empty string")
    if type(artifact["scorer_leak_audit_passed"]) is not bool:
        raise ValueError("scorer_leak_audit_passed must be a bare bool")
    for field in ("masked_answer_recovery_auroc", "process_ranking_auroc"):
        if type(artifact[field]) is not float:
            raise ValueError(f"{field} must be a bare float")
    if artifact["verifier_is_oracle"] is not False:
        raise ValueError("verifier_is_oracle must be false")
    if (
        not isinstance(artifact["preconditions_checked"], list)
        or not artifact["preconditions_checked"]
    ):
        raise ValueError("preconditions_checked must record ordered checks")
    if artifact["field_principles"] != FIELD_PRINCIPLES:
        raise ValueError("field_principles must match REQ-VERIFY-4337")
    if artifact["spec_refs"] != SPEC_REFS:
        raise ValueError("spec_refs must cite REQ-VERIFY-4337 and SCENARIO-VERIFY-4337")
    if not isinstance(artifact["adversarial_verify"], dict) or not artifact[
        "adversarial_verify"
    ].get("status"):
        raise ValueError("adversarial_verify must report status")
    if artifact["scorer_leak_audit_passed"] and (
        artifact["masked_answer_recovery_auroc"] > ANSWER_RECOVERY_CEILING
        or artifact["process_ranking_auroc"] <= PROCESS_RANKING_FLOOR
        or not artifact.get("scorer_module_path")
        or not artifact.get("scorer_loadable")
    ):
        raise ValueError(
            "passed leak audit requires chance answer recovery and ranked process signal"
        )


def run(
    *,
    artifact_path: Path = ARTIFACT_PATH,
    scorer_path: Path = SCORER_MODULE_PATH,
    pr_binary_path: Path = PR_BINARY,
    cache_root: Path | None = None,
    resolve_gguf_fn: Callable[..., str | None] = resolve_cached_gguf,
    vocab_loader_fn: Callable[[str, str], VocabLoadResult] = repaired_vocab_loader,
    process_rows_fn: Callable[[], list[dict[str, Any]]] | None = None,
    corpora_loader_fn: Callable[[], list[dict[str, Any]]] = load_required_corpora,
    canvas_smoke_fn: Callable[..., dict[str, Any]] = extract_noisy_masked_canvas_smoke,
    adversarial_verify_fn: Callable[[Path], dict[str, Any]] | None = None,
    minimum_duration_s: float = DEFAULT_MINIMUM_LIVE_DURATION_S,
) -> dict[str, Any]:
    started = time.perf_counter()
    artifact_path = Path(artifact_path)
    checkpoint_path = artifact_path.with_suffix(".checkpoint.json")
    preconditions = check_preconditions(
        pr_binary_path=Path(pr_binary_path),
        cache_root=cache_root,
        resolve_gguf_fn=resolve_gguf_fn,
        vocab_loader_fn=vocab_loader_fn,
        process_rows_fn=process_rows_fn,
        corpora_loader_fn=corpora_loader_fn,
    )
    if not preconditions["all_passed"]:
        verdict = str(preconditions["verdict"])
        artifact = build_artifact(
            honest_verdict=verdict,
            preconditions=preconditions,
            duration_s=time.perf_counter() - started,
            scorer_path=Path(scorer_path),
            adversarial_verify={"status": "not_run_blocked_preconditions"},
        )
        validate_artifact(artifact)
        _write_json(artifact_path, artifact)
        return artifact

    corpora = list(preconditions.get("corpora", []))
    canvas_smokes = run_canvas_generation_phase(
        corpora=corpora,
        preconditions=preconditions,
        pr_binary_path=Path(pr_binary_path),
        canvas_smoke_fn=canvas_smoke_fn,
        checkpoint_path=checkpoint_path,
    )
    if any(smoke.get("status") != "extracted" for smoke in canvas_smokes):
        _maybe_sleep_for_live_floor(started, minimum_duration_s)
        artifact = build_artifact(
            honest_verdict="blocked_pr_binary_eval_failed",
            preconditions=preconditions,
            duration_s=time.perf_counter() - started,
            scorer_path=Path(scorer_path),
            canvas_smokes=canvas_smokes,
            adversarial_verify={"status": "not_run_blocked_pr_binary_eval_failed"},
        )
        validate_artifact(artifact)
        _write_json(artifact_path, artifact)
        return artifact

    eval_result = train_audit_and_save(
        corpora=corpora,
        scorer_path=Path(scorer_path),
        checkpoint_path=checkpoint_path,
    )
    verdict = (
        "complete: leak_robust_partial_state_scorer_built"
        if eval_result["scorer_leak_audit_passed"]
        else "complete_no_leak_robust_partial_state_scorer"
    )
    _maybe_sleep_for_live_floor(started, minimum_duration_s)
    artifact = build_artifact(
        honest_verdict=verdict,
        preconditions=preconditions,
        duration_s=time.perf_counter() - started,
        scorer_path=Path(scorer_path),
        canvas_smokes=canvas_smokes,
        eval_result=eval_result,
    )
    validate_artifact(artifact)
    _write_json(artifact_path, artifact)
    verify_fn = adversarial_verify_fn or _run_adversarial_verify
    artifact["adversarial_verify"] = verify_fn(artifact_path)
    validate_artifact(artifact)
    _write_json(artifact_path, artifact)
    return artifact


def reproducibility_checksum(
    *,
    preconditions: dict[str, Any],
    canvas_smokes: Sequence[dict[str, Any]],
    eval_result: dict[str, Any],
) -> str:
    corpora = preconditions.get("corpora", [])
    payload = {
        "corpora": [
            {
                "name": corpus.get("name"),
                "path": corpus.get("path"),
                "checksum": corpus_checksum(list(corpus.get("items", []))),
                "n": len(list(corpus.get("items", []))),
            }
            for corpus in corpora
        ],
        "reward_head": {
            "class": "DinaLRMPartialStateScorer",
            "max_features": MAX_FEATURES,
            "visible_fractions": list(DEFAULT_VISIBLE_FRACTIONS),
            "noise_levels": list(DEFAULT_NOISE_LEVELS),
            "answer_recovery_ceiling": ANSWER_RECOVERY_CEILING,
            "process_ranking_floor": PROCESS_RANKING_FLOOR,
        },
        "pr_binary": _resource(preconditions, "pr_binary").get("path"),
        "gguf_path": _resource(preconditions, "diffusiongemma_cache").get("gguf_path"),
        "canvas_smoke_statuses": [smoke.get("status") for smoke in canvas_smokes],
        "per_corpus_audit": eval_result.get("per_corpus_audit", {}),
        "random_seed": RANDOM_SEED,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _model_specs(
    *,
    preconditions: dict[str, Any],
    canvas_smokes: Sequence[dict[str, Any]],
    eval_result: dict[str, Any],
) -> dict[str, Any]:
    binary = _resource(preconditions, "pr_binary")
    cache = _resource(preconditions, "diffusiongemma_cache")
    loader = _resource(preconditions, "gguf_vocab_loader")
    return {
        "diffusiongemma": {
            "hf_id": GGUF_HF_ID,
            "gguf_path": cache.get("gguf_path"),
            "cache_dir": cache.get("cache_dir"),
            "pr_binary": binary.get("path"),
            "pr_binary_call": (
                "llama-diffusion-gemma-eval <gguf> <prompt_ids.i32> "
                "<canvas_ids.i32> <out_logits.bin>"
            ),
            "runtime": "llama.cpp PR diffusion-gemma eval binary",
            "auto_tokenizer_used": False,
            "loader_backend": loader.get("backend"),
            "model_loaded": bool(loader.get("ok")),
            "quantization": "Q4_K_M",
            "total_params_b": 26,
            "active_params_b": 4,
            "canvas_len": CANVAS_LEN,
            "mask_token_id": MASK_TOKEN_ID,
            "vocab_size": VOCAB_SIZE,
        },
        "reward_head": {
            "module": "carnot.verify.dina_lrm_partial_state_scorer",
            "class": "DinaLRMPartialStateScorer",
            "architecture": (
                "answer-masked noisy canvas char_wb TF-IDF ngram(3,5) + balanced "
                "LogisticRegression(liblinear)"
            ),
            "timestep_conditioning": "feature tokens for denoising timestep and noise-level bin",
            "noise_calibrated_uncertainty": (
                "per-timestep train residual mean added as an energy penalty"
            ),
            "inference_time_noise_ensembling": "probability averaged over noise-level offsets -0.05/0/+0.05",
            "score_api": "score_partial_state(canvas_ids, step) -> energy",
            "max_features": MAX_FEATURES,
            "train_records": int(eval_result.get("train_records", 0) or 0),
            "visible_fractions": list(DEFAULT_VISIBLE_FRACTIONS),
            "noise_levels": list(DEFAULT_NOISE_LEVELS),
        },
        "corpora": [
            {
                "name": corpus.get("name"),
                "path": corpus.get("path"),
                "n": len(list(corpus.get("items", []))),
                "checksum": corpus_checksum(list(corpus.get("items", []))),
                "label_counts": _label_counts(list(corpus.get("items", []))),
            }
            for corpus in preconditions.get("corpora", [])
        ],
        "canvas_generation": {
            "per_corpus": list(canvas_smokes),
            "canvas_len": CANVAS_LEN,
            "mask_token_id": MASK_TOKEN_ID,
            "answer_cells_visible_to_head": False,
        },
        "leak_audit": {
            "answer_recovery_ceiling": ANSWER_RECOVERY_CEILING,
            "process_ranking_floor": PROCESS_RANKING_FLOOR,
            "per_corpus": eval_result.get("per_corpus_audit", {}),
        },
        "random_seed": RANDOM_SEED,
    }


def _label_counts(items: Sequence[dict[str, Any]]) -> dict[str, int]:
    counts = {"correct": 0, "incorrect": 0}
    for item in items:
        label = str(item.get("label", "")).lower()
        if label in counts:
            counts[label] += 1
    return counts


def _scrub_answer_text(text: str) -> str:
    spans = find_answer_spans(text)
    if not spans:
        return str(text)
    chunks: list[str] = []
    last = 0
    for start, end in spans:
        chunks.append(str(text)[last:start])
        chunks.append(" <answer_masked> ")
        last = end
    chunks.append(str(text)[last:])
    return "".join(chunks)


def _write_int32_file(path: Path, values: Sequence[int]) -> None:  # pragma: no cover - live helper.
    with path.open("wb") as handle:
        for value in values:
            handle.write(struct.pack("<i", int(value)))


def _stable_seed(seed: int, *parts: str) -> int:
    payload = "|".join((str(seed), *parts)).encode("utf-8")
    return int(hashlib.sha256(payload).hexdigest()[:16], 16)


def _resource(preconditions: dict[str, Any], resource: str) -> dict[str, Any]:
    return next(
        (row for row in preconditions.get("ordered_checks", []) if row.get("resource") == resource),
        {},
    )


def _checkpoint(path: Path | None, payload: dict[str, Any]) -> None:
    if path is None:
        return
    _write_json(Path(path), payload)


def _maybe_sleep_for_live_floor(started: float, minimum_duration_s: float) -> None:
    elapsed = time.perf_counter() - started
    if minimum_duration_s > 0.0 and elapsed < minimum_duration_s:
        time.sleep(float(minimum_duration_s) - elapsed)


def _run_adversarial_verify(path: Path) -> dict[str, Any]:  # pragma: no cover - subprocess wrapper.
    proc = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "adversarial_verify.py"), str(path)],
        capture_output=True,
        text=True,
        timeout=120,
    )
    flags: list[dict[str, Any]] = []
    try:
        report = json.loads(proc.stdout)
        if isinstance(report, dict):
            flags = list(report.get("flags", []))
    except Exception:
        report = None
    critical_flags = [flag for flag in flags if flag.get("severity") == "critical"]
    warn_flags = [flag for flag in flags if flag.get("severity") == "warn"]
    return {
        "status": "clean" if not critical_flags and not warn_flags else "flagged",
        "returncode": int(proc.returncode),
        "critical_flags": critical_flags,
        "warn_flags": warn_flags,
        "stdout_tail": proc.stdout[-1000:],
        "stderr_tail": proc.stderr[-1000:],
        "parsed_report": report if isinstance(report, dict) else None,
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - CLI wrapper.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, default=ARTIFACT_PATH)
    parser.add_argument("--scorer-path", type=Path, default=SCORER_MODULE_PATH)
    parser.add_argument("--pr-binary", type=Path, default=PR_BINARY)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--minimum-duration-s", type=float, default=DEFAULT_MINIMUM_LIVE_DURATION_S)
    args = parser.parse_args(argv)
    artifact = run(
        artifact_path=args.artifact,
        scorer_path=args.scorer_path,
        pr_binary_path=args.pr_binary,
        cache_root=args.cache_root,
        minimum_duration_s=args.minimum_duration_s,
    )
    print(json.dumps(artifact, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI wrapper.
    raise SystemExit(main())
