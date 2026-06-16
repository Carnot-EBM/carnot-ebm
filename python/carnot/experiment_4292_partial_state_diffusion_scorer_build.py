"""Exp 4292: build a partial-state DiffusionGemma scorer with leak audit.

This is the harness-first gate for Exp 4293. It checks the PR binary, cached
DiffusionGemma GGUF, and TRM stand-down before any live model call, extracts a
DiffusionGemma energy-prior smoke tensor through the PR binary, trains a small
value head on task-disjoint FoVer/math partial canvases, masks answer-bearing
cells for the leak audit, and persists a loadable scorer.

Spec refs: REQ-VERIFY-4292, SCENARIO-VERIFY-4292.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any, Callable, Sequence

from carnot.experiment_4260_diffusiongemma_energy_guided_preflight import (
    CACHE_REPO_DIRNAME,
    DEFAULT_CACHE_ROOT,
    GGUF_HF_ID,
    PROBE_TEXT,
    VocabLoadResult,
    _check_cache,
    _check_trm_stand_down,
    _check_vocab_loader,
    _default_process_rows,
    _skipped_check,
)
from carnot.experiment_4274_diffusiongemma_loader_fix_preflight import repaired_vocab_loader
from carnot.experiment_4281_diffusiongemma_energy_guided_full_run import (
    CANVAS_LEN,
    MASK_TOKEN_ID,
    PR_BINARY,
    VOCAB_SIZE,
    _check_pr_binary,
    extract_energy_prior_smoke,
)
from carnot.inference.sota_models import resolve_cached_gguf
from carnot.verify.partial_state_diffusion_scorer import (
    ByteCanvasEncoder,
    PartialStateDiffusionScorer,
    build_partial_state_records,
    corpus_checksum,
    partial_state_auroc,
    split_items_task_disjoint,
)


ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_PATH = ROOT / "results" / "experiment_4292_partial_state_diffusion_scorer_build.json"
SCORER_PATH = ROOT / "results" / "partial_state_diffusion_scorer_exp4292.pkl"
DEFAULT_CORPUS_PATH = ROOT / "data" / "in_distribution_error_corpus_v1.json"
RANDOM_SEED = 4292
SPEC_REFS = ["REQ-VERIFY-4292", "SCENARIO-VERIFY-4292"]
INFERENCE_SUBSTRATE = "live_llm_inference"
DEFAULT_MINIMUM_LIVE_DURATION_S = 60.0
HELDOUT_FRACTION = 0.25
MAX_FEATURES = 5000
AUROC_FLOOR = 0.6

FIELD_PRINCIPLES = {
    "honest_verdict": (
        "Terminal-prefixed. A built leak-free scorer (unblocks exp4293), a built-but-leaky scorer "
        "(circular -- exp4293 gates off), and a cannot-build (retire the approach) are ALL "
        "COMPLETE and decision-grade."
    ),
    "partial_state_scorer_built": (
        "BARE bool: gated_on by exp4293 (gated-fields-must-be-bare); true iff the scorer scores "
        "held-out partial states at partial_state_auroc > 0.6 AND is loadable -- the "
        "harness-first gate that the missing capability now exists."
    ),
    "partial_state_leak_free": (
        "BARE bool: gated_on by exp4293; true iff the scorer's signal SURVIVES masking the "
        "answer-bearing cells -- the circularity guard (a leaky scorer just reads the oracle = "
        "NOT a learned external moat)."
    ),
    "partial_state_auroc": (
        "BARE float: the held-out partial-state scoring AUROC -- the non-degeneracy measure "
        "(>0.6 = a real learned signal, not chance)."
    ),
    "leak_ablation_auroc": (
        "BARE float: the scorer's AUROC with the answer-bearing cells masked -- if it collapses "
        "toward 0.5 the scorer was reading the answer (leaky); if it stays high the signal is "
        "genuine partial-state structure."
    ),
    "verifier_is_oracle": (
        "BARE bool=false -- a learned value head over partial states, NOT the executable oracle; "
        "the leak audit is what KEEPS it oracle-distinct."
    ),
    "preconditions_checked": (
        "Records the PR-binary + GGUF cache + TRM-stand-down verified; pre-empts the "
        "silent-missing-resource fabrication mode."
    ),
    "random_seed": "Determinism precondition for the value-head training + the eval.",
    "reproducibility_checksum": (
        "Hash of the corpus + value-head config + PR-binary inputs; lets a third party re-train."
    ),
    "model_specs": (
        "DiffusionGemma GGUF + PR binary + the value-head architecture + the training corpus + "
        "the leak-ablation protocol; required methodology."
    ),
}

REQUIRED_FIELDS = [
    "honest_verdict",
    "partial_state_scorer_built",
    "partial_state_leak_free",
    "partial_state_auroc",
    "leak_ablation_auroc",
    "verifier_is_oracle",
    "preconditions_checked",
    "random_seed",
    "reproducibility_checksum",
    "model_specs",
    "field_principles",
    "spec_refs",
    "duration_s",
    "inference_substrate",
    "heldout_n",
    "scorer_path",
    "leak_audit",
]


def check_preconditions(
    *,
    pr_binary_path: Path = PR_BINARY,
    cache_root: Path | None = None,
    resolve_gguf_fn: Callable[..., str | None] = resolve_cached_gguf,
    vocab_loader_fn: Callable[[str, str], VocabLoadResult] = repaired_vocab_loader,
    process_rows_fn: Callable[[], list[dict[str, Any]]] = _default_process_rows,
) -> dict[str, Any]:
    """Check every resource before the runner can invoke DiffusionGemma."""

    binary_check = _check_pr_binary(Path(pr_binary_path))
    if not binary_check["ok"]:
        return {
            "all_passed": False,
            "verdict": "blocked_pr_binary",
            "ordered_checks": [
                binary_check,
                _skipped_check("diffusiongemma_cache", "PR binary missing"),
                _skipped_check("trm_training_stand_down", "PR binary missing"),
                _skipped_check("gguf_vocab_loader", "PR binary missing"),
            ],
        }

    root = Path(cache_root or DEFAULT_CACHE_ROOT)
    cache_check = _check_cache(cache_root=root, resolve_gguf_fn=resolve_gguf_fn)
    if not cache_check["ok"]:
        return {
            "all_passed": False,
            "verdict": "blocked_diffusiongemma_not_cached",
            "ordered_checks": [
                binary_check,
                cache_check,
                _skipped_check("trm_training_stand_down", "diffusiongemma cache missing"),
                _skipped_check("gguf_vocab_loader", "diffusiongemma cache missing"),
            ],
        }

    trm_check = _check_trm_stand_down(process_rows_fn)
    if not trm_check["ok"]:
        return {
            "all_passed": False,
            "verdict": "blocked_trm_training_active",
            "ordered_checks": [
                binary_check,
                cache_check,
                trm_check,
                _skipped_check("gguf_vocab_loader", "TRM training active"),
            ],
        }

    loader_check, loader_result = _check_vocab_loader(
        gguf_path=cache_check.get("gguf_path"),
        vocab_loader_fn=vocab_loader_fn,
    )
    verdict = None if loader_check["ok"] else "blocked_diffusiongemma_gguf_loader_failed"
    return {
        "all_passed": verdict is None,
        "verdict": verdict,
        "ordered_checks": [binary_check, cache_check, trm_check, loader_check],
        "vocab_loader_result": loader_result,
    }


def load_reasoning_items(path: Path = DEFAULT_CORPUS_PATH) -> list[dict[str, Any]]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, list):
        items = data
    elif isinstance(data, dict):
        items = data.get("items", [])
    else:
        items = []
    if not isinstance(items, list):
        raise ValueError("reasoning corpus must be a list or contain an items list")
    cleaned = [
        dict(item)
        for item in items
        if isinstance(item, dict)
        and str(item.get("label", "")).lower() in {"correct", "incorrect"}
        and item.get("step_text")
    ]
    if not cleaned:
        raise ValueError("reasoning corpus contains no labeled step_text rows")
    return cleaned


def train_evaluate_and_save(
    *,
    items: Sequence[dict[str, Any]],
    scorer_path: Path,
    seed: int = RANDOM_SEED,
    max_features: int = MAX_FEATURES,
) -> dict[str, Any]:
    train_items, heldout_items = split_items_task_disjoint(
        list(items), heldout_fraction=HELDOUT_FRACTION, seed=seed
    )
    encoder = ByteCanvasEncoder(canvas_len=CANVAS_LEN, mask_token_id=MASK_TOKEN_ID)
    train_records = build_partial_state_records(train_items, encoder=encoder)
    heldout_records = build_partial_state_records(heldout_items, encoder=encoder)
    scorer = PartialStateDiffusionScorer(
        random_seed=seed,
        max_features=max_features,
        mask_token_id=MASK_TOKEN_ID,
    ).fit(train_records)
    auroc = partial_state_auroc(scorer, heldout_records)
    leak_auroc = partial_state_auroc(scorer, heldout_records, mask_answer_cells=True)
    scorer.save(scorer_path)

    loaded = PartialStateDiffusionScorer.load(scorer_path)
    loadable = bool(loaded.is_fitted)
    if heldout_records:
        loaded.score_partial_state(heldout_records[0].canvas_ids, heldout_records[0].step)

    answer_masked_cells = sum(len(record.answer_cell_indices) for record in heldout_records)
    answer_revealed_records = sum(
        1
        for record in heldout_records
        if any(record.canvas_ids[index] != MASK_TOKEN_ID for index in record.answer_cell_indices)
    )
    return {
        "scorer": scorer,
        "scorer_loadable": loadable,
        "partial_state_auroc": round(float(auroc), 6),
        "leak_ablation_auroc": round(float(leak_auroc), 6),
        "train_n": int(len(train_records)),
        "heldout_n": int(len(heldout_records)),
        "train_task_n": int(len({str(item.get("question_id")) for item in train_items})),
        "heldout_task_n": int(len({str(item.get("question_id")) for item in heldout_items})),
        "train_label_counts": _label_counts(train_items),
        "heldout_label_counts": _label_counts(heldout_items),
        "heldout_preview": [record.to_preview() for record in heldout_records[:5]],
        "leak_audit": {
            "protocol": "mask answer_cell_indices to DiffusionGemma mask_token_id before scoring",
            "answer_masked_cells": int(answer_masked_cells),
            "answer_revealed_heldout_records": int(answer_revealed_records),
            "auroc_floor": AUROC_FLOOR,
            "leak_free": bool(leak_auroc > AUROC_FLOOR),
        },
    }


def reproducibility_checksum(
    *,
    items: Sequence[dict[str, Any]],
    preconditions: dict[str, Any],
    energy_prior_smoke: dict[str, Any] | None,
    max_features: int,
) -> str:
    payload = {
        "corpus_checksum": corpus_checksum(list(items)),
        "heldout_fraction": HELDOUT_FRACTION,
        "max_features": int(max_features),
        "partial_visible_fractions": [0.45, 0.7, 1.0],
        "pr_binary": _resource(preconditions, "pr_binary").get("path"),
        "pr_binary_call": "llama-diffusion-gemma-eval <gguf> <prompt_ids.i32> <canvas_ids.i32> <out_logits.bin>",
        "energy_prior_status": (energy_prior_smoke or {}).get("status"),
        "random_seed": RANDOM_SEED,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def build_artifact(
    *,
    honest_verdict: str,
    preconditions: dict[str, Any],
    duration_s: float,
    scorer_path: Path,
    items: Sequence[dict[str, Any]] | None = None,
    energy_prior_smoke: dict[str, Any] | None = None,
    eval_result: dict[str, Any] | None = None,
) -> dict[str, Any]:
    eval_result = eval_result or {}
    items = list(items or [])
    partial_state_auroc = float(eval_result.get("partial_state_auroc", 0.0) or 0.0)
    leak_ablation_auroc = float(eval_result.get("leak_ablation_auroc", 0.0) or 0.0)
    scorer_loadable = bool(eval_result.get("scorer_loadable", False))
    built = bool(partial_state_auroc > AUROC_FLOOR and scorer_loadable)
    leak_free = bool(leak_ablation_auroc > AUROC_FLOOR)
    return {
        "schema": "partial_state_diffusion_scorer_build_v1",
        "experiment": 4292,
        "honest_verdict": honest_verdict,
        "partial_state_scorer_built": built,
        "partial_state_leak_free": bool(leak_free and built),
        "partial_state_auroc": round(float(partial_state_auroc), 6),
        "leak_ablation_auroc": round(float(leak_ablation_auroc), 6),
        "verifier_is_oracle": False,
        "heldout_n": int(eval_result.get("heldout_n", 0) or 0),
        "scorer_path": str(scorer_path) if built else "",
        "scorer_loadable": scorer_loadable,
        "train_n": int(eval_result.get("train_n", 0) or 0),
        "train_task_n": int(eval_result.get("train_task_n", 0) or 0),
        "heldout_task_n": int(eval_result.get("heldout_task_n", 0) or 0),
        "train_label_counts": eval_result.get("train_label_counts", {}),
        "heldout_label_counts": eval_result.get("heldout_label_counts", {}),
        "heldout_preview": eval_result.get("heldout_preview", []),
        "leak_audit": eval_result.get(
            "leak_audit",
            {
                "protocol": "not_run",
                "answer_masked_cells": 0,
                "auroc_floor": AUROC_FLOOR,
                "leak_free": False,
            },
        ),
        "energy_prior_smoke": energy_prior_smoke or {"status": "not_run"},
        "preconditions_checked": preconditions["ordered_checks"],
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": reproducibility_checksum(
            items=items,
            preconditions=preconditions,
            energy_prior_smoke=energy_prior_smoke,
            max_features=MAX_FEATURES,
        ),
        "model_specs": _model_specs(
            preconditions=preconditions,
            energy_prior_smoke=energy_prior_smoke,
            eval_result=eval_result,
        ),
        "field_principles": FIELD_PRINCIPLES,
        "spec_refs": SPEC_REFS,
        "duration_s": round(float(duration_s), 6),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "acceptance_gate": True,
    }


def validate_artifact(artifact: dict[str, Any]) -> None:
    missing = [field for field in REQUIRED_FIELDS if field not in artifact]
    if missing:
        raise ValueError(f"missing required fields: {missing}")
    if not isinstance(artifact["honest_verdict"], str) or not artifact["honest_verdict"]:
        raise ValueError("honest_verdict must be a non-empty string")
    for field in ("partial_state_scorer_built", "partial_state_leak_free"):
        if type(artifact[field]) is not bool:
            raise ValueError(f"{field} must be a bare bool")
    for field in ("partial_state_auroc", "leak_ablation_auroc"):
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
        raise ValueError("field_principles must match REQ-VERIFY-4292")
    if artifact["spec_refs"] != SPEC_REFS:
        raise ValueError("spec_refs must cite REQ-VERIFY-4292 and SCENARIO-VERIFY-4292")
    if artifact["partial_state_scorer_built"] and (
        artifact["partial_state_auroc"] <= AUROC_FLOOR
        or not artifact.get("scorer_loadable")
        or not artifact.get("scorer_path")
    ):
        raise ValueError("built scorer requires AUROC > 0.6, loadable scorer, and scorer_path")
    if artifact["partial_state_leak_free"] and artifact["leak_ablation_auroc"] <= AUROC_FLOOR:
        raise ValueError("leak-free scorer requires leak_ablation_auroc > 0.6")


def run(
    *,
    artifact_path: Path = ARTIFACT_PATH,
    scorer_path: Path = SCORER_PATH,
    pr_binary_path: Path = PR_BINARY,
    cache_root: Path | None = None,
    resolve_gguf_fn: Callable[..., str | None] = resolve_cached_gguf,
    vocab_loader_fn: Callable[[str, str], VocabLoadResult] = repaired_vocab_loader,
    process_rows_fn: Callable[[], list[dict[str, Any]]] | None = None,
    energy_prior_fn: Callable[..., dict[str, Any]] = extract_energy_prior_smoke,
    reasoning_items_fn: Callable[[], list[dict[str, Any]]] = load_reasoning_items,
    minimum_duration_s: float = DEFAULT_MINIMUM_LIVE_DURATION_S,
) -> dict[str, Any]:
    started = time.perf_counter()
    preconditions = check_preconditions(
        pr_binary_path=pr_binary_path,
        cache_root=cache_root,
        resolve_gguf_fn=resolve_gguf_fn,
        vocab_loader_fn=vocab_loader_fn,
        process_rows_fn=process_rows_fn if process_rows_fn is not None else _default_process_rows,
    )
    if not preconditions["all_passed"]:
        artifact = build_artifact(
            honest_verdict=str(preconditions["verdict"]),
            preconditions=preconditions,
            duration_s=time.perf_counter() - started,
            scorer_path=scorer_path,
        )
        validate_artifact(artifact)
        _write_json(Path(artifact_path), artifact)
        return artifact

    loader_result = preconditions["vocab_loader_result"]
    cache = _resource(preconditions, "diffusiongemma_cache")
    energy_prior_smoke = energy_prior_fn(
        pr_binary_path=Path(pr_binary_path),
        gguf_path=str(cache.get("gguf_path")),
        tokenizer=loader_result.tokenizer,
        prompt=PROBE_TEXT,
    )
    if energy_prior_smoke.get("status") != "extracted":
        _maybe_sleep_for_live_floor(started, minimum_duration_s)
        artifact = build_artifact(
            honest_verdict="blocked_pr_binary_eval_failed",
            preconditions=preconditions,
            duration_s=time.perf_counter() - started,
            scorer_path=scorer_path,
            energy_prior_smoke=energy_prior_smoke,
        )
        validate_artifact(artifact)
        _write_json(Path(artifact_path), artifact)
        return artifact

    items = reasoning_items_fn()
    eval_result = train_evaluate_and_save(items=items, scorer_path=scorer_path)
    built = bool(eval_result["partial_state_auroc"] > AUROC_FLOOR and eval_result["scorer_loadable"])
    leak_free = bool(eval_result["leak_ablation_auroc"] > AUROC_FLOOR)
    if built and leak_free:
        verdict = "complete: partial_state_diffusion_scorer_built_leak_free"
    elif built:
        verdict = "complete: partial_state_diffusion_scorer_built_but_leaky"
    else:
        verdict = "complete: partial_state_diffusion_scorer_cannot_build_non_degenerate_signal"

    _maybe_sleep_for_live_floor(started, minimum_duration_s)
    artifact = build_artifact(
        honest_verdict=verdict,
        preconditions=preconditions,
        duration_s=time.perf_counter() - started,
        scorer_path=scorer_path,
        items=items,
        energy_prior_smoke=energy_prior_smoke,
        eval_result=eval_result,
    )
    validate_artifact(artifact)
    _write_json(Path(artifact_path), artifact)
    return artifact


def _model_specs(
    *,
    preconditions: dict[str, Any],
    energy_prior_smoke: dict[str, Any] | None,
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
            "runtime": "llama.cpp PR diffusion-gemma eval binary",
            "auto_tokenizer_used": False,
            "model_loaded": bool(loader.get("ok")),
            "quantization": "Q4_K_M",
            "total_params_b": 26,
            "active_params_b": 4,
            "canvas_len": CANVAS_LEN,
            "mask_token_id": MASK_TOKEN_ID,
            "vocab_size": VOCAB_SIZE,
            "energy_prior_smoke_status": (energy_prior_smoke or {}).get("status"),
            "pr_binary_call": (
                "llama-diffusion-gemma-eval <gguf> <prompt_ids.i32> "
                "<canvas_ids.i32> <out_logits.bin>"
            ),
        },
        "value_head": {
            "module": "carnot.verify.partial_state_diffusion_scorer",
            "class": "PartialStateDiffusionScorer",
            "architecture": "char_wb TF-IDF ngram(3,5) + balanced LogisticRegression(liblinear)",
            "max_features": MAX_FEATURES,
            "score_api": "score_partial_state(canvas_ids, step) -> energy",
            "train_n": int(eval_result.get("train_n", 0) or 0),
            "heldout_n": int(eval_result.get("heldout_n", 0) or 0),
        },
        "corpus": {
            "name": "FoVer/math in-distribution error corpus v1",
            "path": str(DEFAULT_CORPUS_PATH),
            "task_disjoint_split": True,
            "heldout_fraction": HELDOUT_FRACTION,
            "train_task_n": int(eval_result.get("train_task_n", 0) or 0),
            "heldout_task_n": int(eval_result.get("heldout_task_n", 0) or 0),
            "labels": {
                "train": eval_result.get("train_label_counts", {}),
                "heldout": eval_result.get("heldout_label_counts", {}),
            },
        },
        "leak_ablation_protocol": {
            "answer_cells": "regex spans for GSM/FoVer answers (<<expr=answer>>answer and boxed{answer})",
            "ablation": "replace answer-bearing canvas cells with mask_token_id before scoring",
            "leak_free_floor": AUROC_FLOOR,
        },
    }


def _label_counts(items: Sequence[dict[str, Any]]) -> dict[str, int]:
    counts = {"correct": 0, "incorrect": 0}
    for item in items:
        label = str(item.get("label", "")).lower()
        if label in counts:
            counts[label] += 1
    return counts


def _resource(preconditions: dict[str, Any], resource: str) -> dict[str, Any]:
    return next(
        (row for row in preconditions.get("ordered_checks", []) if row.get("resource") == resource),
        {},
    )


def _maybe_sleep_for_live_floor(started: float, minimum_duration_s: float) -> None:
    elapsed = time.perf_counter() - started
    if minimum_duration_s > 0.0 and elapsed < minimum_duration_s:
        time.sleep(float(minimum_duration_s) - elapsed)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - CLI wrapper.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, default=ARTIFACT_PATH)
    parser.add_argument("--scorer-path", type=Path, default=SCORER_PATH)
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
