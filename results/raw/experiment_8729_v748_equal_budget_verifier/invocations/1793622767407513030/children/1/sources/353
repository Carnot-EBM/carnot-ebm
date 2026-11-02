"""Exp 3070 first-token confidence and abstention panel.

Spec refs: REQ-VERIFY-3070,
           SCENARIO-VERIFY-3070,
           SCENARIO-VERIFY-3070-BLOCKED.

This module keeps the claim deliberately narrow: a mandated local GGUF scores a
tiny set of exact-labeled SAT/SMT candidate solutions, the first generated
token's confidence is calibrated on one split, and an abstention rule is
evaluated on the held-out split. The exact solver remains the authority; the
LLM contributes only a scored signal whose limits are recorded in the artifact.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any

from carnot.eval.local_sota_solution_verifier_gain_panel_v1 import (
    build_sat_smt_fixtures,
    compute_exact_ground_truth,
    evaluate_candidate,
)
from carnot.experiment_3043_verified_speculation_transcript_fingerprint import (
    _cuda_probe,
    _extract_text,
    _file_evidence,
    _gpu_inventory,
    _normalize_output,
    _python_environment,
    _repo_commit,
)
from carnot.inference.sota_models import SOTA_GGUF_MODELS, resolve_cached_gguf


JsonDict = dict[str, Any]
ResolveGgufFn = Callable[[str, str], str | None]
LlamaFactory = Callable[..., Any]
ClockFn = Callable[[], float]
RepoCommitFn = Callable[[Path], str]

ARTIFACT = "experiment_3070_first_token_abstention_sota_panel_v1"
ARTIFACT_FILENAME = f"{ARTIFACT}.json"
SCRIPT_FILENAME = f"{ARTIFACT}.py"
SCHEMA = "carnot.first_token_abstention_sota_panel.v1"
RUN_DATE = "20260525"
REPO_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_REL_PATH = Path("results") / ARTIFACT_FILENAME
PANEL_ROWS_REL_PATH = Path("results") / "first_token_abstention_sota_panel_3070" / "rows.jsonl"
EXP3057_REL_PATH = Path("results/experiment_3057_local_sota_solution_verifier_gain_panel_v1.json")
DEFAULT_SEED = 307000
DEFAULT_LOGPROBS = 5
DEFAULT_DECODE_CONFIG: JsonDict = {
    "max_tokens": 4,
    "temperature": 0.0,
    "top_p": 1.0,
    "top_k": 40,
    "repeat_penalty": 1.0,
    "stop": ["\n"],
}
VALIDITY_GRAMMAR = 'root ::= "VALID" | "INVALID"'
DEFAULT_LOAD_CONFIG: JsonDict = {
    "n_ctx": 1024,
    "n_batch": 64,
    "n_ubatch": 64,
    "n_gpu_layers": -1,
    "main_gpu": 0,
    "logits_all": True,
    "verbose": False,
}
MANDATED_MODEL_IDS = tuple(model["hf_id"] for model in SOTA_GGUF_MODELS)
REQUIRED_ARTIFACT_FIELDS = (
    "first_token_panel_ready",
    "confidence_signal",
    "first_token_auc",
    "abstention_precision",
    "rejection_recall",
    "abstention_coverage",
    "verifier_gain_delta_with_abstention",
    "false_positive_rate",
    "false_negative_rate",
    "exact_ground_truth_count",
    "models_used",
    "model_specs",
    "legacy_smoke_only_used",
    "prompt_hashes",
    "inference_substrate",
    "honest_verdict",
)
SUCCESS_PREFIXES = (
    "complete:",
    "complete_",
    "success:",
    "success_",
    "passed:",
    "passed_",
    "shipped:",
    "shipped_",
)


@dataclass(frozen=True)
class ExperimentConfig:
    """Runtime knobs for the Exp 3070 confidence panel."""

    repo_root: Path = REPO_ROOT
    output_path: Path | None = None
    rows_path: Path | None = None
    seed: int = DEFAULT_SEED
    preferred_quant: str = "Q4_K_M"
    logprobs: int = DEFAULT_LOGPROBS
    decode_config: Mapping[str, Any] | None = None
    load_config: Mapping[str, Any] | None = None
    tests_run: Sequence[str] = ()

    def artifact_path(self) -> Path:
        return self.output_path or self.repo_root / OUTPUT_REL_PATH

    def panel_rows_path(self) -> Path:
        return self.rows_path or self.repo_root / PANEL_ROWS_REL_PATH

    def effective_decode_config(self) -> JsonDict:
        config = dict(DEFAULT_DECODE_CONFIG)
        if self.decode_config:
            config.update(dict(self.decode_config))
        return config

    def effective_load_config(self, gpu: int = 0) -> JsonDict:
        config = dict(DEFAULT_LOAD_CONFIG)
        if self.load_config:
            config.update(dict(self.load_config))
        config["main_gpu"] = int(gpu)
        return config


def build_scoring_rows() -> list[JsonDict]:
    """Build exact-labeled candidate rows from the Exp 3057 fixtures."""

    truth_rows = compute_exact_ground_truth(build_sat_smt_fixtures())
    split_cut = len(truth_rows) // 2
    rows: list[JsonDict] = []
    for fixture_index, truth_row in enumerate(truth_rows):
        split = "calibration" if fixture_index < split_cut else "heldout"
        for candidate_id, candidate in (
            ("candidate_good", truth_row["ground_truth_candidate"]),
            ("candidate_bad", _distractor(truth_row)),
        ):
            exact_correct = evaluate_candidate(truth_row, candidate)
            rows.append(
                {
                    "fixture_id": truth_row["fixture_id"],
                    "fixture_index": fixture_index,
                    "split": split,
                    "candidate_id": candidate_id,
                    "candidate": dict(candidate),
                    "exact_correct": bool(exact_correct),
                    "exact_label": "VALID" if exact_correct else "INVALID",
                    "exact_authority": "z3_solver",
                    "exact_checked": True,
                    "variables": list(truth_row["variables"]),
                    "constraints": [dict(row) for row in truth_row["constraints"]],
                }
            )
    return rows


def run_experiment(
    config: ExperimentConfig | None = None,
    *,
    resolve_gguf_func: ResolveGgufFn = resolve_cached_gguf,
    llama_factory: LlamaFactory | None = None,
    monotonic: ClockFn = time.monotonic,
    repo_commit_func: RepoCommitFn = _repo_commit,
) -> JsonDict:
    """Run the first-token panel and write the terminal artifact."""

    active = config or ExperimentConfig()
    started = monotonic()
    cache_resolution = _resolve_cache(resolve_gguf_func, active)
    selected_model = _select_model(cache_resolution)
    if selected_model is None:
        artifact = _blocked_artifact(
            config=active,
            cache_resolution=cache_resolution,
            selected_models=[],
            duration_s=round(monotonic() - started, 6),
            runtime_blocker="no_mandated_gguf_resolved",
            repo_commit_func=repo_commit_func,
        )
        _write_json(active.artifact_path(), artifact)
        return artifact

    try:
        rows = _run_live_scoring(
            config=active,
            selected_model=selected_model,
            llama_factory=llama_factory or _default_llama_factory,
        )
        if rows and all(row.get("confidence_available") for row in rows):
            rows = _annotate_abstention(rows)
        runtime_blocker = None
    except Exception as exc:  # pragma: no cover - live runtime failure path.
        rows = []
        runtime_blocker = f"{type(exc).__name__}: {exc}"

    duration_s = round(monotonic() - started, 6)
    if rows:
        _write_jsonl(active.panel_rows_path(), rows)
    artifact = _build_artifact(
        config=active,
        rows=rows,
        selected_models=[selected_model],
        cache_resolution=cache_resolution,
        duration_s=duration_s,
        runtime_blocker=runtime_blocker,
        repo_commit_func=repo_commit_func,
    )
    validate_artifact(artifact)
    _write_json(active.artifact_path(), artifact)
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Raise when an Exp 3070 artifact violates the confidence-panel contract."""

    missing = sorted(set(REQUIRED_ARTIFACT_FIELDS) - set(artifact))
    if missing:
        raise ValueError(f"missing required fields: {missing}")
    if artifact.get("legacy_smoke_only_used") is not False:
        raise ValueError("legacy smoke evidence cannot satisfy REQ-VERIFY-3070")
    verdict = str(artifact.get("honest_verdict", ""))
    if artifact.get("first_token_panel_ready") is not True:
        if not verdict.startswith("blocked_sota_confidence_unavailable"):
            raise ValueError("honest_verdict must disclose blocked_sota_confidence_unavailable")
        return
    if not artifact.get("model_specs"):
        raise ValueError("model_specs must be present when the panel is ready")
    if int(artifact.get("exact_ground_truth_count") or 0) < 6:
        raise ValueError("exact_ground_truth_count must be at least 6 when ready")
    if not artifact.get("prompt_hashes"):
        raise ValueError("prompt_hashes must be non-empty when ready")
    if artifact.get("confidence_signal") == "unavailable":
        raise ValueError("confidence_signal must name the measured signal when ready")
    if int(artifact.get("heldout_split_count") or 0) <= 0:
        raise ValueError("abstention metrics require a held-out split")
    if not verdict.startswith(SUCCESS_PREFIXES):
        raise ValueError("honest_verdict must start with a terminal success prefix")


def load_jsonl(path: Path) -> list[JsonDict]:
    """Load JSONL panel rows written by this module."""

    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _run_live_scoring(
    *,
    config: ExperimentConfig,
    selected_model: Mapping[str, Any],
    llama_factory: LlamaFactory,
) -> list[JsonDict]:
    load_config = config.effective_load_config(int(selected_model.get("gpu", 0)))
    llm = llama_factory(model_path=str(selected_model["model_path"]), **load_config)
    rows: list[JsonDict] = []
    try:
        for row in build_scoring_rows():
            prompt = _scoring_prompt(row)
            call_config = config.effective_decode_config()
            grammar = _validity_grammar()
            if grammar is not None:
                call_config["grammar"] = grammar
            raw = llm(
                prompt,
                **call_config,
                seed=config.seed,
                logprobs=config.logprobs,
            )
            text = _normalize_output(_extract_text(raw))
            confidence = _confidence_from_output(raw)
            predicted_valid = _parse_validity_decision(text)
            rows.append(
                {
                    **row,
                    "model_id": selected_model["hf_id"],
                    "model_name": selected_model["name"],
                    "prompt_hash": _sha256_text(prompt),
                    "raw_output_hash": _sha256_text(text),
                    "decision_text": text[:80],
                    "predicted_valid": predicted_valid,
                    "model_exact_agreement": predicted_valid == row["exact_correct"],
                    **confidence,
                }
            )
    finally:
        close = getattr(llm, "close", None)
        if callable(close):
            close()
    return rows


def _build_artifact(
    *,
    config: ExperimentConfig,
    rows: Sequence[Mapping[str, Any]],
    selected_models: Sequence[Mapping[str, Any]],
    cache_resolution: Mapping[str, str | None],
    duration_s: float,
    runtime_blocker: str | None,
    repo_commit_func: RepoCommitFn,
) -> JsonDict:
    confidence_available = bool(rows) and all(row.get("confidence_available") for row in rows)
    if rows and not confidence_available and runtime_blocker is None:
        runtime_blocker = "confidence_signal_unavailable"
    metrics = _metrics(rows) if confidence_available else _empty_metrics()
    model_specs = [_model_spec(row) for row in selected_models] if selected_models else []
    models_used = [str(row["hf_id"]) for row in selected_models]
    ready = (
        runtime_blocker is None
        and confidence_available
        and metrics["exact_ground_truth_count"] >= 6
        and metrics["heldout_split_count"] > 0
        and metrics["non_vacuous_abstention_metrics"]
        and bool(model_specs)
    )
    artifact: JsonDict = {
        "artifact": ARTIFACT,
        "schema": SCHEMA,
        "run_date": RUN_DATE,
        "first_token_panel_ready": ready,
        "confidence_signal": _selected_signal(rows) if confidence_available else "unavailable",
        "first_token_auc": metrics["first_token_auc"],
        "abstention_precision": metrics["abstention_precision"],
        "rejection_recall": metrics["rejection_recall"],
        "abstention_coverage": metrics["abstention_coverage"],
        "verifier_gain_delta_with_abstention": metrics[
            "verifier_gain_delta_with_abstention"
        ],
        "false_positive_rate": metrics["false_positive_rate"],
        "false_negative_rate": metrics["false_negative_rate"],
        "exact_ground_truth_count": metrics["exact_ground_truth_count"] if ready else 0,
        "models_used": models_used,
        "model_specs": model_specs if runtime_blocker != "no_mandated_gguf_resolved" else [],
        "legacy_smoke_only_used": False,
        "prompt_hashes": [str(row["prompt_hash"]) for row in rows]
        if confidence_available
        else [],
        "inference_substrate": _substrate(
            config=config,
            cache_resolution=cache_resolution,
            selected_models=selected_models,
            rows=rows,
            duration_s=duration_s,
            repo_commit_func=repo_commit_func,
        ),
        "honest_verdict": _honest_verdict(ready, metrics, runtime_blocker),
        "calibration_threshold": metrics["calibration_threshold"],
        "calibration_split_count": metrics["calibration_split_count"],
        "heldout_split_count": metrics["heldout_split_count"],
        "accepted_count": metrics["accepted_count"],
        "rejected_count": metrics["rejected_count"],
        "abstained_count": metrics["abstained_count"],
        "prior_exp3057_verifier_selected_accuracy": metrics[
            "prior_exp3057_verifier_selected_accuracy"
        ],
        "panel_rows_path": str(_relative_to(config.repo_root, config.panel_rows_path())),
        "panel_row_count": len(rows),
        "panel_rows_sha256": _sha256_file(config.panel_rows_path()) if rows else "",
        "tests_or_checks_run": list(config.tests_run),
        "decode_config": config.effective_decode_config(),
        "seed": config.seed,
        "duration_s": duration_s,
        "runtime_blocker": runtime_blocker,
        "reproducibility_checksum": "",
    }
    artifact["reproducibility_checksum"] = _sha256_json(
        {
            "models_used": artifact["models_used"],
            "prompt_hashes": artifact["prompt_hashes"],
            "metrics": {
                name: artifact[name]
                for name in (
                    "first_token_auc",
                    "abstention_precision",
                    "rejection_recall",
                    "abstention_coverage",
                    "false_positive_rate",
                    "false_negative_rate",
                )
            },
        }
    )
    return artifact


def _blocked_artifact(
    *,
    config: ExperimentConfig,
    cache_resolution: Mapping[str, str | None],
    selected_models: Sequence[Mapping[str, Any]],
    duration_s: float,
    runtime_blocker: str,
    repo_commit_func: RepoCommitFn,
) -> JsonDict:
    artifact = _build_artifact(
        config=config,
        rows=[],
        selected_models=selected_models,
        cache_resolution=cache_resolution,
        duration_s=duration_s,
        runtime_blocker=runtime_blocker,
        repo_commit_func=repo_commit_func,
    )
    validate_artifact(artifact)
    return artifact


def _metrics(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    calibration = [row for row in rows if row["split"] == "calibration"]
    heldout = [row for row in rows if row["split"] == "heldout"]
    threshold = _derive_threshold(calibration)
    heldout_scored = [
        dict(row)
        if row.get("abstention_decision")
        else _with_abstention(row, threshold)
        for row in heldout
    ]
    good = [row for row in heldout_scored if row["exact_correct"]]
    bad = [row for row in heldout_scored if not row["exact_correct"]]
    accepted = [row for row in heldout_scored if row["abstention_decision"] == "accept"]
    rejected = [row for row in heldout_scored if row["abstention_decision"] == "reject"]
    abstained = [row for row in heldout_scored if row["abstention_decision"] == "abstain"]
    accepted_good = [row for row in accepted if row["exact_correct"]]
    rejected_bad = [row for row in rejected if not row["exact_correct"]]
    abstained_bad = [row for row in abstained if not row["exact_correct"]]
    accepted_bad = [row for row in accepted if not row["exact_correct"]]
    rejected_good = [row for row in rejected if row["exact_correct"]]
    safety_accuracy = (
        (len(accepted_good) + len(rejected_bad) + len(abstained_bad)) / len(heldout_scored)
        if heldout_scored
        else 0.0
    )
    prior_accuracy = _prior_exp3057_verifier_selected_accuracy(REPO_ROOT)
    return {
        "exact_ground_truth_count": len({row["fixture_id"] for row in rows}),
        "calibration_split_count": len(calibration),
        "heldout_split_count": len(heldout_scored),
        "calibration_threshold": round(threshold, 6),
        "first_token_auc": round(
            _auc(
                [bool(row["model_exact_agreement"]) for row in heldout_scored],
                [_float(row.get("confidence_score")) for row in heldout_scored],
            ),
            6,
        ),
        "abstention_precision": round(len(accepted_good) / len(accepted), 6)
        if accepted
        else 0.0,
        "rejection_recall": round(len(rejected_bad) / len(bad), 6) if bad else 0.0,
        "abstention_coverage": round(len(abstained) / len(heldout_scored), 6)
        if heldout_scored
        else 0.0,
        "false_positive_rate": round(len(accepted_bad) / len(bad), 6) if bad else 0.0,
        "false_negative_rate": round(len(rejected_good) / len(good), 6) if good else 0.0,
        "verifier_gain_delta_with_abstention": round(safety_accuracy - prior_accuracy, 6),
        "accepted_count": len(accepted),
        "rejected_count": len(rejected),
        "abstained_count": len(abstained),
        "prior_exp3057_verifier_selected_accuracy": round(prior_accuracy, 6),
        "non_vacuous_abstention_metrics": bool(heldout_scored and accepted and (rejected or abstained)),
    }


def _empty_metrics() -> JsonDict:
    return {
        "exact_ground_truth_count": 0,
        "calibration_split_count": 0,
        "heldout_split_count": 0,
        "calibration_threshold": 0.0,
        "first_token_auc": 0.0,
        "abstention_precision": 0.0,
        "rejection_recall": 0.0,
        "abstention_coverage": 0.0,
        "false_positive_rate": 0.0,
        "false_negative_rate": 0.0,
        "verifier_gain_delta_with_abstention": 0.0,
        "accepted_count": 0,
        "rejected_count": 0,
        "abstained_count": 0,
        "prior_exp3057_verifier_selected_accuracy": 0.0,
        "non_vacuous_abstention_metrics": False,
    }


def _derive_threshold(calibration_rows: Sequence[Mapping[str, Any]]) -> float:
    correct_confidences = [
        _float(row.get("confidence_score"))
        for row in calibration_rows
        if row.get("model_exact_agreement") is True
    ]
    if correct_confidences:
        return min(correct_confidences)
    confidences = [_float(row.get("confidence_score")) for row in calibration_rows]
    return min(confidences) if confidences else 1.0


def _annotate_abstention(rows: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    calibration = [row for row in rows if row["split"] == "calibration"]
    threshold = _derive_threshold(calibration)
    return [
        _with_abstention(row, threshold)
        if row["split"] == "heldout"
        else dict(row) | {"abstention_decision": "calibration", "calibration_threshold": threshold}
        for row in rows
    ]


def _with_abstention(row: Mapping[str, Any], threshold: float) -> JsonDict:
    confidence = _float(row.get("confidence_score"))
    predicted_valid = row.get("predicted_valid")
    if confidence < threshold or predicted_valid is None:
        decision = "abstain"
    elif predicted_valid is True:
        decision = "accept"
    else:
        decision = "reject"
    return dict(row) | {"abstention_decision": decision, "calibration_threshold": threshold}


def _confidence_from_output(output: Mapping[str, Any]) -> JsonDict:
    choice = _first_choice(output)
    logprobs = choice.get("logprobs") if isinstance(choice, Mapping) else None
    if not isinstance(logprobs, Mapping):
        return _missing_confidence()
    token_logprobs = _float_list(logprobs.get("token_logprobs"))
    top_logprobs = logprobs.get("top_logprobs")
    tokens = [str(token) for token in logprobs.get("tokens", [])] if isinstance(logprobs, Mapping) else []
    index = _first_content_index(tokens, token_logprobs)
    topk = top_logprobs[index] if isinstance(top_logprobs, list) and index < len(top_logprobs) else None
    topk_confidence = _topk_entropy_confidence(topk if isinstance(topk, Mapping) else {})
    token = tokens[index].strip() if tokens and index < len(tokens) else ""
    if topk_confidence["confidence_available"]:
        return topk_confidence | {
            "first_token": token or topk_confidence["first_token"],
            "first_token_logprob": token_logprobs[index] if index < len(token_logprobs) else None,
        }
    if index < len(token_logprobs):
        probability = max(0.0, min(1.0, math.exp(token_logprobs[index])))
        return {
            "confidence_available": True,
            "confidence_signal": "first_token_logprob_proxy",
            "confidence_score": round(probability, 6),
            "first_token_entropy": None,
            "first_token": token,
            "first_token_logprob": token_logprobs[index],
            "first_token_top_logprobs": {},
            "confidence_limitation": "top_logprobs_unavailable; using chosen-token probability proxy",
        }
    return _missing_confidence()


def _topk_entropy_confidence(top_logprobs: Mapping[str, Any]) -> JsonDict:
    values: list[tuple[str, float]] = []
    for token, raw in top_logprobs.items():
        try:
            values.append((str(token), float(raw)))
        except (TypeError, ValueError):
            continue
    if not values:
        return _missing_confidence()
    max_logprob = max(value for _token, value in values)
    weights = [math.exp(value - max_logprob) for _token, value in values]
    total = sum(weights)
    if total <= 0.0:  # pragma: no cover - finite exp weights cannot sum to zero.
        return _missing_confidence()
    probs = [weight / total for weight in weights]
    entropy = -sum(prob * math.log(prob) for prob in probs if prob > 0.0)
    normalizer = math.log(len(probs)) if len(probs) > 1 else 1.0
    normalized_entropy = entropy / normalizer if normalizer > 0.0 else 0.0
    confidence = max(0.0, min(1.0, 1.0 - normalized_entropy))
    first_token = max(values, key=lambda item: item[1])[0].strip()
    return {
        "confidence_available": True,
        "confidence_signal": "first_token_topk_entropy",
        "confidence_score": round(confidence, 6),
        "first_token_entropy": round(normalized_entropy, 6),
        "first_token": first_token,
        "first_token_top_logprobs": {token: value for token, value in values},
        "confidence_limitation": None,
    }


def _missing_confidence() -> JsonDict:
    return {
        "confidence_available": False,
        "confidence_signal": "unavailable",
        "confidence_score": 0.0,
        "first_token_entropy": None,
        "first_token": "",
        "first_token_logprob": None,
        "first_token_top_logprobs": {},
        "confidence_limitation": "no_first_token_logprob_or_topk_logprobs",
    }


def _first_choice(output: Mapping[str, Any]) -> Mapping[str, Any]:
    choices = output.get("choices")
    if isinstance(choices, list) and choices and isinstance(choices[0], Mapping):
        return choices[0]
    return {}


def _first_content_index(tokens: Sequence[str], token_logprobs: Sequence[float]) -> int:
    for index, token in enumerate(tokens):
        if token.strip():
            return index
    return 0 if token_logprobs or tokens else 0


def _parse_validity_decision(text: str) -> bool | None:
    upper = text.strip().upper()
    if upper.startswith("INVALID"):
        return False
    if upper.startswith("VALID"):
        return True
    return None


def _auc(labels: Sequence[bool], scores: Sequence[float]) -> float:
    positives = [score for label, score in zip(labels, scores, strict=False) if label]
    negatives = [score for label, score in zip(labels, scores, strict=False) if not label]
    if not positives or not negatives:
        return 0.5
    wins = 0.0
    for positive in positives:
        for negative in negatives:
            if positive > negative:
                wins += 1.0
            elif positive == negative:
                wins += 0.5
    return wins / (len(positives) * len(negatives))


def _select_model(cache_resolution: Mapping[str, str | None]) -> JsonDict | None:
    for index, model in enumerate(SOTA_GGUF_MODELS):
        path = cache_resolution.get(model["hf_id"])
        if path:
            return {
                "name": model["name"],
                "hf_id": model["hf_id"],
                "model_path": path,
                "gpu": min(index, 1),
                "role": model["role"],
                "quantization": model["quantization"],
                "family": _model_family(model["hf_id"]),
            }
    return None


def _resolve_cache(
    resolve_gguf_func: ResolveGgufFn, config: ExperimentConfig
) -> dict[str, str | None]:
    return {hf_id: resolve_gguf_func(hf_id, config.preferred_quant) for hf_id in MANDATED_MODEL_IDS}


def _model_spec(model: Mapping[str, Any]) -> JsonDict:
    evidence = _file_evidence(str(model["model_path"]), full_limit_bytes=512 * 1024 * 1024)
    return {
        "name": model["name"],
        "hf_id": model["hf_id"],
        "model_path": model["model_path"],
        "gpu": model["gpu"],
        "role": model["role"],
        "family": model["family"],
        "quantization": model["quantization"],
        "model_hash_or_cache_path": evidence.get("model_hash_or_cache_path"),
        "checksum_feasibility": {
            "method": evidence.get("method"),
            "full_sha256_feasible": bool(evidence.get("full_sha256_feasible")),
            "size_bytes": evidence.get("size_bytes"),
        },
    }


def _substrate(
    *,
    config: ExperimentConfig,
    cache_resolution: Mapping[str, str | None],
    selected_models: Sequence[Mapping[str, Any]],
    rows: Sequence[Mapping[str, Any]],
    duration_s: float,
    repo_commit_func: RepoCommitFn,
) -> JsonDict:
    return {
        "cuda_probe": _cuda_probe(),
        "gpu_inventory": _gpu_inventory(),
        "python_environment": _python_environment(),
        "repo_commit": repo_commit_func(config.repo_root),
        "gguf_cache_resolution": dict(cache_resolution),
        "model_ids": list(MANDATED_MODEL_IDS),
        "selected_model_paths": [str(model["model_path"]) for model in selected_models],
        "quantization": config.preferred_quant,
        "seed": config.seed,
        "decode_config": config.effective_decode_config(),
        "load_config": config.effective_load_config(),
        "logprobs_requested": config.logprobs,
        "grammar": "VALID_OR_INVALID_GBNF",
        "logprob_support": {
            "any_confidence": any(row.get("confidence_available") for row in rows),
            "top_logprobs": any(row.get("confidence_signal") == "first_token_topk_entropy" for row in rows),
            "chosen_token_logprob": any(row.get("first_token_logprob") is not None for row in rows),
        },
        "wall_clock_duration_s": duration_s,
        "runtime": "llama_cpp",
        "exact_solver": "z3",
    }


def _honest_verdict(
    ready: bool,
    metrics: Mapping[str, Any],
    runtime_blocker: str | None,
) -> str:
    if ready:
        return (
            "complete: first_token_panel_ready=true; "
            f"first_token_auc={metrics['first_token_auc']}; "
            f"abstention_precision={metrics['abstention_precision']}; "
            f"verifier_gain_delta_with_abstention="
            f"{metrics['verifier_gain_delta_with_abstention']}"
        )
    reason = runtime_blocker or "abstention_metrics_vacuous"
    return f"blocked_sota_confidence_unavailable: {reason}"


def _scoring_prompt(row: Mapping[str, Any]) -> str:
    return (
        "Role: verifier\n"
        "Score whether this candidate satisfies the exact SAT/SMT fixture. "
        "Return one uppercase word only: VALID or INVALID.\n"
        f"Fixture ID: {row['fixture_id']}\n"
        f"Candidate ID: {row['candidate_id']}\n"
        f"Variables: {', '.join(row['variables'])}\n"
        f"Constraints: {_constraints_text(row['constraints'])}\n"
        f"Candidate: {json.dumps(row['candidate'], sort_keys=True)}\n"
    )


def _constraints_text(constraints: Sequence[Mapping[str, Any]]) -> str:
    return "; ".join(
        f"{constraint['terms']} {constraint['op']} {constraint['rhs']}"
        for constraint in constraints
    )


def _distractor(truth_row: Mapping[str, Any]) -> JsonDict:
    if truth_row["solver_status"] == "unsat":
        return {"status": "sat", "assignment": {name: 0 for name in truth_row["variables"]}}
    assignment = dict(truth_row["ground_truth_assignment"])
    first = truth_row["variables"][0]
    assignment[first] = int(assignment[first]) + 1
    return {"status": "sat", "assignment": assignment}


def _selected_signal(rows: Sequence[Mapping[str, Any]]) -> str:
    signals = {str(row.get("confidence_signal")) for row in rows if row.get("confidence_signal")}
    signals.discard("unavailable")
    if not signals:
        return "unavailable"
    if len(signals) == 1:
        return next(iter(signals))
    return "mixed_first_token_topk_entropy_and_logprob_proxy"


def _prior_exp3057_verifier_selected_accuracy(root: Path) -> float:
    path = root / EXP3057_REL_PATH
    if not path.is_file():
        path = REPO_ROOT / EXP3057_REL_PATH
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return 0.0
    return _float(payload.get("verifier_selected_accuracy"))


def _default_llama_factory(**kwargs: Any) -> Any:  # pragma: no cover - live hardware path.
    from llama_cpp import Llama  # noqa: PLC0415

    return Llama(**kwargs)


def _validity_grammar() -> Any | None:
    try:
        from llama_cpp import LlamaGrammar  # noqa: PLC0415
    except Exception:  # pragma: no cover - llama_cpp is present in the tested venv.
        return None
    return LlamaGrammar.from_string(VALIDITY_GRAMMAR)


def _model_family(hf_id: str) -> str:
    lowered = hf_id.lower()
    if "qwen" in lowered:
        return "qwen"
    if "gemma" in lowered:
        return "gemma"
    return hf_id.split("/", 1)[0].lower()


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _float_list(value: Any) -> list[float]:
    if not isinstance(value, list):
        return []
    output: list[float] = []
    for raw in value:
        try:
            output.append(float(raw))
        except (TypeError, ValueError):
            continue
    return output


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _sha256_json(value: Any) -> str:
    return _sha256_text(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True))


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(dict(row), sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _relative_to(root: Path, path: Path) -> Path:
    try:
        return path.relative_to(root)
    except ValueError:
        return path
