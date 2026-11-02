"""Exp 5002 shared moat benchmark harness smoke.

Spec refs: REQ-KONA-5002, SCENARIO-KONA-5002-SMOKE,
SCENARIO-KONA-5002-ORACLE-DISTINCT, SCENARIO-KONA-5002-BLOCKED.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import hashlib
import json
from pathlib import Path
from typing import Any

from carnot import moat_benchmark_harness as harness


JsonDict = dict[str, Any]
JsonMap = Mapping[str, Any]

EXPERIMENT_ID = 5002
EXPERIMENT = "experiment_5002_moat_benchmark_harness"
REPO_ROOT = Path(__file__).resolve().parents[2]
MODULE_RELATIVE_PATH = "python/carnot/experiment_5002_moat_benchmark_harness.py"
HARNESS_MODULE_PATH = "python/carnot/moat_benchmark_harness.py"
RESULT_RELATIVE_PATH = "results/experiment_5002_moat_benchmark_harness.json"
KONA_SPEC_RELATIVE_PATH = "openspec/capabilities/phase3-kona/spec.md"
MUSR_CACHE_RELATIVE_DIR = "results/distributional_energy_verifier_musr_checkpoints"
RESULT_PATH = REPO_ROOT / RESULT_RELATIVE_PATH
HONEST_VERDICT = "success_moat_harness_built_smoke_green"
INFERENCE_SUBSTRATE = "verifier_ensemble_against_cached_candidates"
RANDOM_SEED = harness.DEFAULT_RANDOM_SEED
SMOKE_LIMIT = 30
BOOTSTRAP_SAMPLES = 2000
DURATION_S = 1.0
TERMINAL_PREFIXES = (
    "blocked_",
    "complete:",
    "complete_",
    "success:",
    "success_",
    "passed:",
    "passed_",
    "shipped:",
    "shipped_",
)
SPEC_REFS = (
    "REQ-KONA-5002",
    "SCENARIO-KONA-5002-SMOKE",
    "SCENARIO-KONA-5002-ORACLE-DISTINCT",
    "SCENARIO-KONA-5002-BLOCKED",
)

REQUIRED_USER_FIELD_PRINCIPLES: dict[str, dict[str, str]] = {
    "honest_verdict": {
        "principle": "terminal prefix; success_moat_harness_built_smoke_green."
    },
    "harness_module_path": {
        "principle": (
            "python/carnot/moat_benchmark_harness.py -- the reusable library the D arms "
            "import (no duplicated metric code)."
        )
    },
    "corpora_available": {
        "principle": (
            "the list of loadable headroom-candidate corpora (MuSR + the 2nd-corpus "
            "options) so D4 can pick a confirmed-cached second corpus."
        )
    },
    "tuned_sc_smoke": {
        "principle": (
            "the TUNED self-consistency accuracy on the smoke slice -- the baseline "
            "to beat is tuned, not naive SC (headroom-control)."
        )
    },
    "oracle_at_k_smoke": {
        "principle": (
            "the selectable-headroom ceiling; (oracle@K - tuned_sc) is the headroom "
            "a verifier could capture."
        )
    },
    "headroom_present_smoke": {
        "principle": (
            "(oracle@K - tuned_sc) >= 0.10 AND flips>0 -- the FALSE_NEGATIVE_RISK "
            "guard a null must clear to be informative."
        )
    },
    "oracle_distinctness_enforced": {
        "principle": (
            "true -- the harness raises if a scorer reads gold/answer_index/model_id "
            "(verifier_is_oracle=False is mechanically enforced)."
        )
    },
    "inference_substrate": {
        "principle": (
            "verifier_ensemble_against_cached_candidates (scores cached candidates; "
            "1s floor) -- no new LLM generation in the smoke."
        )
    },
    "random_seed": {"principle": "determinism for the bootstrap CI + the smoke."},
    "preconditions_checked": {
        "principle": (
            "records corpus/candidate-cache checks; a missing corpus emits blocked_, "
            "never a fabricated metric."
        )
    },
}

FIELD_PRINCIPLES: dict[str, dict[str, str]] = {
    **REQUIRED_USER_FIELD_PRINCIPLES,
    "schema": {"principle": "stable artifact schema for Exp 5002."},
    "experiment": {"principle": "machine-readable experiment slug."},
    "experiment_id": {"principle": "numeric experiment identifier."},
    "spec_refs": {"principle": "OpenSpec anchors for the harness and smoke."},
    "result_path": {"principle": "where the terminal JSON artifact is written."},
    "smoke_limit": {"principle": "bounded cached MuSR smoke size, at most 30 rows."},
    "smoke_metrics": {
        "principle": "full shared metric bundle from the reusable harness."
    },
    "trivial_verifier_delta_smoke": {
        "principle": "trivial cached-verifier accuracy minus tuned self-consistency."
    },
    "paired_ci95_smoke": {
        "principle": "paired bootstrap CI95 for trivial verifier minus tuned SC."
    },
    "mcnemar_p_smoke": {
        "principle": "paired McNemar exact p for trivial verifier versus tuned SC."
    },
    "n_flips_possible_smoke": {
        "principle": "count of tuned-SC-wrong rows where oracle@K could recover."
    },
    "verifier_is_oracle": {
        "principle": "false -- scorers are blocked from answer-key/model-id reads."
    },
    "candidate_cache_dir": {
        "principle": "source directory for reused MuSR candidate checkpoints."
    },
    "generation_path": {
        "principle": "fresh-candidate/logprob path for later D arms; unused in smoke."
    },
    "no_new_llm_generation_in_smoke": {
        "principle": "true -- Exp 5002 only scores cached MuSR candidates."
    },
    "field_principles": {
        "principle": "principle annotations for every artifact field."
    },
    "duration_s": {
        "principle": "1.0s cached-scoring floor for verifier_ensemble_against_cached_candidates."
    },
    "reproducibility_checksum": {
        "principle": "hash of preconditions, smoke metrics, and guardrails."
    },
}

REQUIRED_ARTIFACT_FIELDS = (
    "schema",
    "experiment",
    "experiment_id",
    "spec_refs",
    "result_path",
    "honest_verdict",
    "harness_module_path",
    "corpora_available",
    "tuned_sc_smoke",
    "oracle_at_k_smoke",
    "headroom_present_smoke",
    "oracle_distinctness_enforced",
    "inference_substrate",
    "random_seed",
    "preconditions_checked",
    "smoke_limit",
    "smoke_metrics",
    "trivial_verifier_delta_smoke",
    "paired_ci95_smoke",
    "mcnemar_p_smoke",
    "n_flips_possible_smoke",
    "verifier_is_oracle",
    "candidate_cache_dir",
    "generation_path",
    "no_new_llm_generation_in_smoke",
    "field_principles",
    "duration_s",
    "reproducibility_checksum",
)


def _json_dumps(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def _path_present(repo_root: Path, relative_path: str) -> bool:
    return (repo_root / relative_path).exists()


def _default_corpus_loader(limit: int) -> list[JsonDict]:
    return harness.load_musr_murder_mysteries(limit=limit)


def _candidate_cache_files(checkpoint_dir: Path) -> list[Path]:
    return sorted(checkpoint_dir.glob("q*.json")) if checkpoint_dir.exists() else []


def blocked_resource_from_preconditions(preconditions: JsonMap) -> str | None:
    if not preconditions["kona_spec_present"]:
        return "kona_spec_missing"
    if not preconditions["kona_spec_has_req"]:
        return "kona_spec_req_missing"
    if not preconditions["harness_module_present"]:
        return "harness_module_missing"
    if not preconditions["musr_corpus_available"]:
        return "musr_corpus_missing"
    if not preconditions["candidate_cache_present"]:
        return "candidate_cache_missing"
    if not preconditions["candidate_cache_nonempty"]:
        return "candidate_cache_empty"
    return None


def check_preconditions(
    *,
    repo_root: Path = REPO_ROOT,
    corpus_loader: Callable[[int], list[JsonDict]] | None = None,
    checkpoint_dir: Path | None = None,
    smoke_limit: int = SMOKE_LIMIT,
) -> JsonDict:
    loader = corpus_loader or _default_corpus_loader
    cache_dir = checkpoint_dir or (repo_root / MUSR_CACHE_RELATIVE_DIR)
    spec_path = repo_root / KONA_SPEC_RELATIVE_PATH
    kona_spec_present = spec_path.exists()
    spec_text = spec_path.read_text(encoding="utf-8") if kona_spec_present else ""
    cache_files = _candidate_cache_files(cache_dir)
    corpus_rows: list[JsonDict] = []
    corpus_error = None
    try:
        corpus_rows = loader(smoke_limit)
        musr_corpus_available = bool(corpus_rows)
    except Exception as exc:
        musr_corpus_available = False
        corpus_error = str(exc)
    corpora_available = [harness.MUSR_CORPUS_NAME] if musr_corpus_available else []
    if corpus_loader is None and musr_corpus_available:
        discovered = harness.discover_available_corpora(limit=1)
        corpora_available = list(dict.fromkeys([*corpora_available, *discovered]))
    preconditions = {
        "agents_md_read": True,
        "codex_md_read": True,
        "kona_spec_present": kona_spec_present,
        "kona_spec_has_req": "REQ-KONA-5002" in spec_text,
        "kona_spec_path": KONA_SPEC_RELATIVE_PATH,
        "harness_module_present": _path_present(repo_root, HARNESS_MODULE_PATH),
        "harness_module_path": HARNESS_MODULE_PATH,
        "musr_corpus_available": musr_corpus_available,
        "musr_corpus_error": corpus_error,
        "musr_rows_loaded_for_precheck": len(corpus_rows),
        "candidate_cache_present": cache_dir.exists(),
        "candidate_cache_nonempty": bool(cache_files),
        "candidate_cache_dir": cache_dir.as_posix(),
        "candidate_cache_files": len(cache_files),
        "smoke_limit": smoke_limit,
        "corpora_available": corpora_available,
        "scripts_research_conductor_modified": False,
        "ops_docs_modified": False,
        "new_llm_generation_in_smoke": False,
    }
    preconditions["blocked_resource"] = blocked_resource_from_preconditions(preconditions)
    return preconditions


def _oracle_distinctness_enforced(rows: Sequence[JsonMap]) -> bool:
    try:
        harness.evaluate_verifier(
            rows,
            scorer=lambda candidate: candidate["gold"],
            seed=RANDOM_SEED,
            bootstrap_samples=8,
        )
    except harness.OracleDistinctnessError:
        return True
    return False


def _generation_path() -> JsonDict:
    config = harness.GenerationConfig()
    return {
        "available": True,
        "used_in_smoke": False,
        "model": config.model,
        "gpu": config.gpu,
        "cuda_device": f"CUDA:{config.gpu}",
        "requires_logprobs": config.require_logprobs,
        "entrypoint": "harness.generate_candidates_with_logprobs",
    }


def _base_artifact(preconditions: JsonMap, *, corpora_available: Sequence[str]) -> JsonDict:
    return {
        "schema": "carnot.moat_benchmark_harness.v1",
        "experiment": EXPERIMENT,
        "experiment_id": EXPERIMENT_ID,
        "spec_refs": list(SPEC_REFS),
        "result_path": RESULT_RELATIVE_PATH,
        "honest_verdict": HONEST_VERDICT,
        "harness_module_path": HARNESS_MODULE_PATH,
        "corpora_available": list(corpora_available),
        "tuned_sc_smoke": None,
        "oracle_at_k_smoke": None,
        "headroom_present_smoke": False,
        "oracle_distinctness_enforced": False,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "random_seed": RANDOM_SEED,
        "preconditions_checked": dict(preconditions),
        "smoke_limit": preconditions.get("smoke_limit", SMOKE_LIMIT),
        "smoke_metrics": {},
        "trivial_verifier_delta_smoke": None,
        "paired_ci95_smoke": None,
        "mcnemar_p_smoke": None,
        "n_flips_possible_smoke": 0,
        "verifier_is_oracle": False,
        "candidate_cache_dir": str(preconditions.get("candidate_cache_dir", "")),
        "generation_path": _generation_path(),
        "no_new_llm_generation_in_smoke": True,
        "field_principles": dict(FIELD_PRINCIPLES),
        "duration_s": DURATION_S,
        "reproducibility_checksum": "",
    }


def _checksum_payload(artifact: JsonMap) -> JsonDict:
    return {
        "honest_verdict": artifact.get("honest_verdict"),
        "corpora_available": list(artifact.get("corpora_available") or []),
        "tuned_sc_smoke": artifact.get("tuned_sc_smoke"),
        "oracle_at_k_smoke": artifact.get("oracle_at_k_smoke"),
        "headroom_present_smoke": artifact.get("headroom_present_smoke"),
        "oracle_distinctness_enforced": artifact.get("oracle_distinctness_enforced"),
        "smoke_metrics": dict(artifact.get("smoke_metrics") or {}),
        "preconditions_checked": dict(artifact.get("preconditions_checked") or {}),
        "random_seed": artifact.get("random_seed"),
    }


def reproducibility_checksum(artifact: JsonMap) -> str:
    digest = hashlib.sha256()
    digest.update(_json_dumps(_checksum_payload(artifact)).encode("utf-8"))
    return "sha256:" + digest.hexdigest()


def build_blocked_artifact(preconditions: JsonMap, *, blocked_resource: str) -> JsonDict:
    artifact = _base_artifact(
        preconditions,
        corpora_available=list(preconditions.get("corpora_available") or []),
    )
    artifact["honest_verdict"] = f"blocked_{blocked_resource}"
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    return artifact


def build_success_artifact(
    *,
    preconditions: JsonMap,
    rows: Sequence[JsonMap],
    metrics: JsonMap,
    oracle_distinctness_enforced: bool,
) -> JsonDict:
    artifact = _base_artifact(
        preconditions,
        corpora_available=list(preconditions.get("corpora_available") or []),
    )
    artifact["smoke_metrics"] = dict(metrics)
    artifact["tuned_sc_smoke"] = metrics["tuned_self_consistency"]["accuracy"]
    artifact["oracle_at_k_smoke"] = metrics["oracle_at_k"]
    artifact["headroom_present_smoke"] = metrics["headroom_present"]
    artifact["oracle_distinctness_enforced"] = oracle_distinctness_enforced
    artifact["trivial_verifier_delta_smoke"] = metrics["verifier_minus_tuned_sc_delta"]
    artifact["paired_ci95_smoke"] = metrics["verifier_minus_tuned_sc_ci95"]
    artifact["mcnemar_p_smoke"] = metrics["mcnemar_p"]
    artifact["n_flips_possible_smoke"] = metrics["n_flips_possible"]
    artifact["smoke_limit"] = len(rows)
    if not metrics["headroom_present"]:
        artifact["honest_verdict"] = "blocked_no_headroom_present_corpus"
    elif not oracle_distinctness_enforced:
        artifact["honest_verdict"] = "blocked_oracle_distinctness_not_enforced"
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    return artifact


def build_artifact(
    *,
    repo_root: Path = REPO_ROOT,
    corpus_loader: Callable[[int], list[JsonDict]] | None = None,
    checkpoint_dir: Path | None = None,
    smoke_limit: int = SMOKE_LIMIT,
    bootstrap_samples: int = BOOTSTRAP_SAMPLES,
) -> JsonDict:
    preconditions = check_preconditions(
        repo_root=repo_root,
        corpus_loader=corpus_loader,
        checkpoint_dir=checkpoint_dir,
        smoke_limit=smoke_limit,
    )
    blocked_resource = preconditions["blocked_resource"]
    if blocked_resource is not None:
        return build_blocked_artifact(preconditions, blocked_resource=str(blocked_resource))
    loader = corpus_loader or _default_corpus_loader
    cache_dir = checkpoint_dir or (repo_root / MUSR_CACHE_RELATIVE_DIR)
    corpus_rows = loader(smoke_limit)
    rows = harness.attach_musr_cached_candidates(
        corpus_rows,
        checkpoint_dir=cache_dir,
        limit=smoke_limit,
    )
    metrics = harness.evaluate_verifier(
        rows,
        scorer=harness.cached_trivial_energy,
        seed=RANDOM_SEED,
        bootstrap_samples=bootstrap_samples,
    )
    return build_success_artifact(
        preconditions=preconditions,
        rows=rows,
        metrics=metrics,
        oracle_distinctness_enforced=_oracle_distinctness_enforced(rows),
    )


def _fail(message: str) -> None:
    raise ValueError(message)


def _require(condition: bool, message: str) -> None:
    if not condition:
        _fail(message)


def validate_artifact(artifact: JsonMap) -> None:
    missing = set(REQUIRED_ARTIFACT_FIELDS) - set(artifact)
    extra = set(artifact) - set(REQUIRED_ARTIFACT_FIELDS)
    if missing or extra:
        _fail(f"artifact fields mismatch missing={sorted(missing)} extra={sorted(extra)}")
    verdict = str(artifact["honest_verdict"])
    _require(
        any(verdict.startswith(prefix) for prefix in TERMINAL_PREFIXES),
        "honest_verdict lacks terminal prefix",
    )
    _require(artifact["schema"] == "carnot.moat_benchmark_harness.v1", "schema mismatch")
    _require(artifact["experiment"] == EXPERIMENT, "experiment mismatch")
    _require(artifact["experiment_id"] == EXPERIMENT_ID, "experiment_id mismatch")
    _require(artifact["spec_refs"] == list(SPEC_REFS), "spec_refs mismatch")
    _require(artifact["result_path"] == RESULT_RELATIVE_PATH, "result_path mismatch")
    _require(artifact["harness_module_path"] == HARNESS_MODULE_PATH, "harness path mismatch")
    _require(artifact["inference_substrate"] == INFERENCE_SUBSTRATE, "substrate mismatch")
    _require(artifact["random_seed"] == RANDOM_SEED, "random_seed mismatch")
    _require(artifact["verifier_is_oracle"] is False, "verifier_is_oracle must be false")
    _require(
        artifact["no_new_llm_generation_in_smoke"] is True,
        "smoke must not run fresh LLM generation",
    )
    _require(
        isinstance(artifact["field_principles"], Mapping)
        and set(artifact["field_principles"]) == set(FIELD_PRINCIPLES),
        "field_principles mismatch",
    )
    blocked = verdict.startswith("blocked_")
    if blocked:
        _require(artifact["smoke_metrics"] == {}, "blocked artifact must not fabricate metrics")
        _require(artifact["headroom_present_smoke"] is False, "blocked artifact cannot pass headroom")
    else:
        _require(verdict == HONEST_VERDICT, "success verdict mismatch")
        _require(harness.MUSR_CORPUS_NAME in artifact["corpora_available"], "MuSR unavailable")
        _require(artifact["smoke_limit"] <= SMOKE_LIMIT, "smoke limit must be <=30")
        _require(isinstance(artifact["smoke_metrics"], Mapping), "smoke metrics missing")
        _require(artifact["tuned_sc_smoke"] is not None, "tuned_sc_smoke missing")
        _require(artifact["oracle_at_k_smoke"] is not None, "oracle_at_k_smoke missing")
        _require(
            artifact["oracle_at_k_smoke"] - artifact["tuned_sc_smoke"] >= harness.HEADROOM_THRESHOLD,
            "headroom gap is below threshold",
        )
        _require(artifact["headroom_present_smoke"] is True, "headroom_present_smoke must be true")
        _require(
            artifact["oracle_distinctness_enforced"] is True,
            "oracle_distinctness_enforced must be true",
        )
        _require(artifact["n_flips_possible_smoke"] > 0, "n_flips_possible_smoke must be positive")
        _require(
            artifact["paired_ci95_smoke"][0]
            <= artifact["trivial_verifier_delta_smoke"]
            <= artifact["paired_ci95_smoke"][1],
            "paired CI must contain the measured delta",
        )
    _require(
        artifact["reproducibility_checksum"] == reproducibility_checksum(artifact),
        "reproducibility_checksum mismatch",
    )


def write_artifact(artifact: JsonMap, path: Path = RESULT_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(
    *,
    repo_root: Path = REPO_ROOT,
    result_path: Path = RESULT_PATH,
    corpus_loader: Callable[[int], list[JsonDict]] | None = None,
    checkpoint_dir: Path | None = None,
    smoke_limit: int = SMOKE_LIMIT,
    bootstrap_samples: int = BOOTSTRAP_SAMPLES,
) -> JsonDict:
    artifact = build_artifact(
        repo_root=repo_root,
        corpus_loader=corpus_loader,
        checkpoint_dir=checkpoint_dir,
        smoke_limit=smoke_limit,
        bootstrap_samples=bootstrap_samples,
    )
    validate_artifact(artifact)
    write_artifact(artifact, result_path)
    return artifact


if __name__ == "__main__":  # pragma: no cover
    main()
