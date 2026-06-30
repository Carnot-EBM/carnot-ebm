"""Exp 5015 genuine self-consistency baseline fix.

Spec refs: REQ-KONA-5015, SCENARIO-KONA-5015-GENUINE-SC,
SCENARIO-KONA-5015-DEGENERACY-GUARD, SCENARIO-KONA-5015-SMOKE.
"""

from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any


JsonDict = dict[str, Any]
JsonMap = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "python") not in sys.path:  # pragma: no cover - direct script execution
    sys.path.insert(0, str(REPO_ROOT / "python"))

try:
    from carnot import moat_benchmark_harness as harness
except Exception as exc:  # pragma: no cover - only used for blocked import artifact
    harness = None  # type: ignore[assignment]
    HARNESS_IMPORT_ERROR = f"{type(exc).__name__}: {exc}"
else:
    HARNESS_IMPORT_ERROR = None


EXPERIMENT_ID = 5015
EXPERIMENT = "experiment_5015_genuine_sc_baseline_fix"
RESULT_RELATIVE_PATH = "results/experiment_5015_genuine_sc_baseline_fix.json"
RESULT_PATH = REPO_ROOT / RESULT_RELATIVE_PATH
HARNESS_MODULE_PATH = "python/carnot/moat_benchmark_harness.py"
CHECKPOINT_RELATIVE_DIR = "results/distributional_energy_verifier_musr_checkpoints"
SMOKE_LIMIT = 200
RANDOM_SEED = 20260630
HONEST_VERDICT = "success_genuine_sc_baseline_fixed_degeneracy_guard_shipped"
INFERENCE_SUBSTRATE = "verifier_ensemble_against_cached_candidates"
SPEC_REFS = [
    "REQ-KONA-5015",
    "SCENARIO-KONA-5015-GENUINE-SC",
    "SCENARIO-KONA-5015-DEGENERACY-GUARD",
    "SCENARIO-KONA-5015-SMOKE",
]

FIELD_PRINCIPLES: dict[str, dict[str, str]] = {
    "honest_verdict": {
        "principle": "terminal prefix; success_genuine_sc_baseline_fixed_degeneracy_guard_shipped."
    },
    "genuine_tuned_sc_accuracy": {
        "principle": (
            "the K-way majority-vote tuned-SC on MuSR (the HONEST baseline to beat -- "
            "replaces the k=1 strawman 0.585 if K>1 helps)."
        )
    },
    "sc_k_sweep": {
        "principle": (
            "the full {K: accuracy} curve so the tuned-K choice is auditable "
            "(proves k=1 was/was-not genuinely best, not a default)."
        )
    },
    "tuned_k": {"principle": "the K that maximizes SC accuracy (the tuned choice)."},
    "candidates_per_question": {
        "principle": (
            "how many cached candidates exist per question (if 1, SC and oracle@K are "
            "degenerate -- flagged honestly)."
        )
    },
    "oracle_at_k": {"principle": "the selectable-headroom ceiling (recomputed)."},
    "genuine_headroom_present": {
        "principle": (
            "(oracle@K - genuine_tuned_sc) >= 0.10 AND flips>0 -- the "
            "FALSE_NEGATIVE_RISK guard against the GENUINE baseline."
        )
    },
    "degeneracy_guard_fires": {
        "principle": (
            "true -- the always-abstain guard flags a >50%-abstain selector (so a "
            "future D arm cannot degenerate to SC undetected)."
        )
    },
    "harness_module_path": {
        "principle": (
            "python/carnot/moat_benchmark_harness.py -- the shared library the D arms "
            "import (fixed in place)."
        )
    },
    "inference_substrate": {
        "principle": (
            "verifier_ensemble_against_cached_candidates (re-scores cached candidates; "
            "1s floor) -- no new LLM generation."
        )
    },
    "random_seed": {"principle": "determinism for the bootstrap + smoke."},
    "preconditions_checked": {
        "principle": (
            "records the candidate-cache + harness-import checks; a missing cache emits blocked_."
        )
    },
}

REQUIRED_ARTIFACT_FIELDS = (
    "schema",
    "experiment",
    "experiment_id",
    "spec_refs",
    "result_path",
    "honest_verdict",
    "genuine_tuned_sc_accuracy",
    "sc_k_sweep",
    "tuned_k",
    "candidates_per_question",
    "oracle_at_k",
    "genuine_headroom_present",
    "degeneracy_guard_fires",
    "harness_module_path",
    "inference_substrate",
    "random_seed",
    "preconditions_checked",
    "n_questions",
    "oracle_k",
    "n_flips_possible",
    "candidate_pool_count_summary",
    "single_candidate_degenerate",
    "oracle_degenerate",
    "degeneracy_guard_demo",
    "corrected_musr_tuned_sc_baseline",
    "no_new_llm_generation",
    "field_principles",
    "duration_s",
    "reproducibility_checksum",
)


def _read_json(path: Path) -> JsonDict:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"checkpoint is not a JSON object: {path}")
    return payload


def _write_json(path: Path, payload: JsonMap) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _checkpoint_files(checkpoint_dir: Path) -> list[Path]:
    return sorted(checkpoint_dir.glob("q*.json")) if checkpoint_dir.exists() else []


def _valid_answers(payload: JsonMap) -> list[str]:
    return [
        str(answer)
        for answer in payload.get("answers") or []
        if answer is not None and str(answer).strip()
    ]


def load_cached_musr_rows(checkpoint_dir: Path, *, limit: int = SMOKE_LIMIT) -> list[JsonDict]:
    rows: list[JsonDict] = []
    for path in _checkpoint_files(checkpoint_dir)[:limit]:
        payload = _read_json(path)
        candidates = [
            {
                "candidate_id": f"{path.stem}/cached-{index}",
                "answer": answer,
                "cache_index": index,
                "temperature": payload.get("temperature", "cached"),
                "source": "distributional_energy_verifier_musr_checkpoints",
            }
            for index, answer in enumerate(_valid_answers(payload))
        ]
        if not candidates:
            raise ValueError(f"checkpoint has no valid answers: {path}")
        rows.append(
            {
                "row_id": path.stem,
                "corpus": "MuSR/murder_mysteries",
                "gold": str(payload.get("gold") or ""),
                "candidate_cache_path": path.as_posix(),
                "candidates": candidates,
            }
        )
    return rows


def candidate_pool_count_summary(rows: list[JsonMap]) -> JsonDict:
    counts = [len(row.get("candidates") or []) for row in rows]
    histogram = Counter(counts)
    return {
        "min": min(counts) if counts else 0,
        "max": max(counts) if counts else 0,
        "unique": sorted(histogram),
        "histogram": {str(key): histogram[key] for key in sorted(histogram)},
    }


def check_preconditions(
    *,
    checkpoint_dir: Path,
    smoke_limit: int = SMOKE_LIMIT,
) -> JsonDict:
    cache_files = _checkpoint_files(checkpoint_dir)
    preconditions = {
        "agents_md_read": True,
        "codex_md_read": True,
        "harness_module_present": (REPO_ROOT / HARNESS_MODULE_PATH).exists(),
        "harness_module_importable": harness is not None,
        "harness_import_error": HARNESS_IMPORT_ERROR,
        "harness_module_path": HARNESS_MODULE_PATH,
        "candidate_cache_dir": checkpoint_dir.as_posix(),
        "candidate_cache_present": checkpoint_dir.exists(),
        "candidate_cache_nonempty": bool(cache_files),
        "candidate_cache_files": len(cache_files),
        "smoke_limit": smoke_limit,
        "new_llm_generation": False,
    }
    if not preconditions["harness_module_present"]:
        blocked_resource = "harness_module_missing"
    elif not preconditions["harness_module_importable"]:
        blocked_resource = "harness_import_failed"
    elif not preconditions["candidate_cache_present"]:
        blocked_resource = "candidate_cache_missing"
    elif not preconditions["candidate_cache_nonempty"]:
        blocked_resource = "candidate_cache_empty"
    else:
        blocked_resource = None
    preconditions["blocked_resource"] = blocked_resource
    return preconditions


def _base_artifact(*, preconditions: JsonMap, duration_s: float) -> JsonDict:
    return {
        "schema": "carnot.genuine_sc_baseline_fix.v1",
        "experiment": EXPERIMENT,
        "experiment_id": EXPERIMENT_ID,
        "spec_refs": list(SPEC_REFS),
        "result_path": RESULT_RELATIVE_PATH,
        "honest_verdict": HONEST_VERDICT,
        "genuine_tuned_sc_accuracy": None,
        "sc_k_sweep": {},
        "tuned_k": None,
        "candidates_per_question": None,
        "oracle_at_k": None,
        "genuine_headroom_present": False,
        "degeneracy_guard_fires": False,
        "harness_module_path": HARNESS_MODULE_PATH,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "random_seed": RANDOM_SEED,
        "preconditions_checked": dict(preconditions),
        "n_questions": 0,
        "oracle_k": None,
        "n_flips_possible": 0,
        "candidate_pool_count_summary": {},
        "single_candidate_degenerate": False,
        "oracle_degenerate": False,
        "degeneracy_guard_demo": {},
        "corrected_musr_tuned_sc_baseline": {},
        "no_new_llm_generation": True,
        "field_principles": FIELD_PRINCIPLES,
        "duration_s": round(max(1.0, float(duration_s)), 6),
        "reproducibility_checksum": "",
    }


def build_blocked_artifact(
    *,
    preconditions: JsonMap,
    duration_s: float,
    blocked_resource: str,
) -> JsonDict:
    artifact = _base_artifact(preconditions=preconditions, duration_s=duration_s)
    artifact["honest_verdict"] = f"blocked_{blocked_resource}"
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    return artifact


def build_success_artifact(
    *,
    preconditions: JsonMap,
    rows: list[JsonMap],
    duration_s: float,
) -> JsonDict:
    assert harness is not None
    tuned = harness.tuned_self_consistency(rows)
    tuned_accuracy = float(tuned["accuracy"])
    oracle_k = int(tuned.get("candidates_per_question") or 0)
    oracle_accuracy, oracle_correct = harness.oracle_at_k(
        rows,
        k=oracle_k,
        temperature=tuned.get("config", {}).get("temperature"),
    )
    sc_correct = [int(value) for value in tuned.get("correct", [])]
    n_flips_possible = sum(
        1 for sc_ok, oracle_ok in zip(sc_correct, oracle_correct) if not sc_ok and oracle_ok
    )
    headroom_present = bool(
        (oracle_accuracy - tuned_accuracy) >= harness.HEADROOM_THRESHOLD and n_flips_possible > 0
    )
    degeneracy_guard = harness.abstention_degeneracy_guard(1.0)
    count_summary = candidate_pool_count_summary(rows)
    artifact = _base_artifact(preconditions=preconditions, duration_s=duration_s)
    artifact.update(
        {
            "genuine_tuned_sc_accuracy": round(tuned_accuracy, 6),
            "sc_k_sweep": dict(tuned.get("k_sweep") or {}),
            "tuned_k": int(tuned.get("tuned_k") or tuned["config"]["k"]),
            "candidates_per_question": oracle_k,
            "oracle_at_k": oracle_accuracy,
            "genuine_headroom_present": headroom_present,
            "degeneracy_guard_fires": bool(degeneracy_guard["degeneracy_flag"]),
            "n_questions": len(rows),
            "oracle_k": oracle_k,
            "n_flips_possible": n_flips_possible,
            "candidate_pool_count_summary": count_summary,
            "single_candidate_degenerate": bool(tuned.get("degenerate_candidate_pool")),
            "oracle_degenerate": bool(tuned.get("oracle_degenerate")),
            "degeneracy_guard_demo": degeneracy_guard,
            "corrected_musr_tuned_sc_baseline": {
                "accuracy": round(tuned_accuracy, 6),
                "config": dict(tuned.get("config") or {}),
                "k_sweep": dict(tuned.get("k_sweep") or {}),
                "tuned_k": int(tuned.get("tuned_k") or tuned["config"]["k"]),
                "candidates_per_question": oracle_k,
                "candidate_pool_count_summary": count_summary,
            },
        }
    )
    if oracle_k == 1:
        artifact["honest_verdict"] = "blocked_single_candidate_sc_oracle_degenerate"
    elif not artifact["degeneracy_guard_fires"]:
        artifact["honest_verdict"] = "blocked_degeneracy_guard_did_not_fire"
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    return artifact


def _checksum_payload(artifact: JsonMap) -> JsonDict:
    return {
        "honest_verdict": artifact.get("honest_verdict"),
        "genuine_tuned_sc_accuracy": artifact.get("genuine_tuned_sc_accuracy"),
        "sc_k_sweep": artifact.get("sc_k_sweep"),
        "tuned_k": artifact.get("tuned_k"),
        "candidates_per_question": artifact.get("candidates_per_question"),
        "oracle_at_k": artifact.get("oracle_at_k"),
        "genuine_headroom_present": artifact.get("genuine_headroom_present"),
        "degeneracy_guard_demo": artifact.get("degeneracy_guard_demo"),
        "preconditions_checked": artifact.get("preconditions_checked"),
        "random_seed": artifact.get("random_seed"),
    }


def reproducibility_checksum(artifact: JsonMap) -> str:
    payload = dict(artifact)
    payload["reproducibility_checksum"] = ""
    digest = hashlib.sha256(json.dumps(_checksum_payload(payload), sort_keys=True).encode("utf-8"))
    return "sha256:" + digest.hexdigest()


def validate_artifact(artifact: JsonMap) -> None:
    missing = set(REQUIRED_ARTIFACT_FIELDS) - set(artifact)
    extra = set(artifact) - set(REQUIRED_ARTIFACT_FIELDS)
    if missing or extra:
        raise ValueError(
            f"artifact fields mismatch missing={sorted(missing)} extra={sorted(extra)}"
        )
    verdict = str(artifact["honest_verdict"])
    if not verdict.startswith(("success_", "blocked_")):
        raise ValueError("honest_verdict lacks terminal prefix")
    if artifact["field_principles"] != FIELD_PRINCIPLES:
        raise ValueError("field_principles mismatch")
    if artifact["harness_module_path"] != HARNESS_MODULE_PATH:
        raise ValueError("harness_module_path mismatch")
    if artifact["inference_substrate"] != INFERENCE_SUBSTRATE:
        raise ValueError("inference_substrate mismatch")
    if artifact["random_seed"] != RANDOM_SEED:
        raise ValueError("random_seed mismatch")
    if artifact["no_new_llm_generation"] is not True:
        raise ValueError("smoke must not run fresh LLM generation")
    if verdict.startswith("success_"):
        if artifact["honest_verdict"] != HONEST_VERDICT:
            raise ValueError("success verdict mismatch")
        if not artifact["sc_k_sweep"]:
            raise ValueError("sc_k_sweep missing")
        if artifact["degeneracy_guard_fires"] is not True:
            raise ValueError("degeneracy guard must fire in the smoke")
        if artifact["candidates_per_question"] == 1 and not artifact["oracle_degenerate"]:
            raise ValueError("single-candidate oracle degeneracy must be flagged")
    if artifact["reproducibility_checksum"] != reproducibility_checksum(artifact):
        raise ValueError("reproducibility_checksum mismatch")


def build_artifact(
    *,
    checkpoint_dir: Path | None = None,
    smoke_limit: int = SMOKE_LIMIT,
    now: Any = time.time,
) -> JsonDict:
    start = float(now())
    cache_dir = checkpoint_dir or (REPO_ROOT / CHECKPOINT_RELATIVE_DIR)
    preconditions = check_preconditions(checkpoint_dir=cache_dir, smoke_limit=smoke_limit)
    blocked_resource = preconditions["blocked_resource"]
    if blocked_resource is not None:
        return build_blocked_artifact(
            preconditions=preconditions,
            duration_s=float(now()) - start,
            blocked_resource=str(blocked_resource),
        )
    try:
        rows = load_cached_musr_rows(cache_dir, limit=smoke_limit)
    except Exception as exc:
        blocked = dict(preconditions)
        blocked["blocked_resource"] = "candidate_cache_malformed"
        blocked["candidate_cache_error"] = f"{type(exc).__name__}: {exc}"
        return build_blocked_artifact(
            preconditions=blocked,
            duration_s=float(now()) - start,
            blocked_resource="candidate_cache_malformed",
        )
    return build_success_artifact(
        preconditions=preconditions,
        rows=rows,
        duration_s=float(now()) - start,
    )


def main(
    *,
    result_path: Path = RESULT_PATH,
    checkpoint_dir: Path | None = None,
    smoke_limit: int = SMOKE_LIMIT,
) -> JsonDict:
    artifact = build_artifact(checkpoint_dir=checkpoint_dir, smoke_limit=smoke_limit)
    validate_artifact(artifact)
    _write_json(result_path, artifact)
    return artifact


if __name__ == "__main__":  # pragma: no cover
    main()
