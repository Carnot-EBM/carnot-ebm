"""Exp 3099 local SOTA confidence and abstention panel v3.

Spec refs: REQ-VERIFY-3099, SCENARIO-VERIFY-3099,
SCENARIO-VERIFY-3099-BLOCKED.

This runner consumes the checked-in Exp 3097 exact-fixture protocol and the
Exp 3098 MaxSAT routing policy before attempting local GGUF inference. The
policy decision is intentionally computed after generation against exact
labels: the model proposes an answer and confidence, while the policy decides
whether that row is accepted, rejected, or abstained under the same safety
constraints downstream calibration tasks must use.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
import sys
import time
from typing import Any

from carnot.eval import maxsat_abstention_routing_policy_v1 as maxsat_policy
from carnot.inference.sota_models import SOTA_GGUF_MODELS, cached_sota_pair, resolve_cached_gguf


JsonDict = dict[str, Any]
ClockFn = Callable[[], float]
ResolveGgufFn = Callable[[str, str], str | None]
CachedPairFn = Callable[..., list[dict[str, Any]] | None]
LlamaFactory = Callable[..., Any]

REPO_ROOT = Path(__file__).resolve().parents[3]
RUN_DATE = "20260526"
ARTIFACT = "experiment_3099_local_sota_confidence_abstention_panel_v3"
SCHEMA = "carnot.local_sota_confidence_abstention_panel.v3"
OUTPUT_REL_PATH = Path("results/experiment_3099_local_sota_confidence_abstention_panel_v3.json")
PANEL_ROWS_REL_PATH = Path("results/local_sota_confidence_abstention_panel_3099/rows.jsonl")
EXP3097_REL_PATH = Path("results/experiment_3097_exact_fixture_eval_protocol_audit_v1.json")
EXP3098_REL_PATH = Path("results/experiment_3098_maxsat_abstention_routing_policy_v1.json")
POLICY_REL_PATH = Path("results/maxsat_abstention_routing_policy_3098/policy.json")
STRATIFIED_MANIFEST_REL_PATH = Path(
    "results/exact_fixture_eval_protocol_3097/stratified_eval_manifest.jsonl"
)
DEFAULT_SEED = 3099
DEFAULT_LOGPROBS: int | None = None
SUCCESS_PREFIXES = (
    "complete:",
    "complete_",
    "success:",
    "success_",
    "passed:",
    "passed_",
    "shipped_",
)
MANDATED_MODEL_IDS = tuple(model["hf_id"] for model in SOTA_GGUF_MODELS)
SOURCE_REL_PATHS: tuple[tuple[str, Path, str], ...] = (
    ("codex", Path("CODEX.md"), "repo spec-first workflow"),
    ("claude", Path("CLAUDE.md"), "artifact-authenticity and tiny-panel discipline"),
    ("research_references", Path("research-references.md"), "diagnostic telemetry context"),
    ("experiment_template", Path("scripts/experiment_template.py"), "SOTA cache helper context"),
    ("exp3097_protocol", EXP3097_REL_PATH, ".289 exact-fixture protocol authority"),
    ("exp3098_policy_artifact", EXP3098_REL_PATH, ".289 MaxSAT policy artifact"),
    ("exp3098_policy_json", POLICY_REL_PATH, "machine-readable MaxSAT routing policy"),
)
REQUIRED_ARTIFACT_FIELDS = (
    "abstention_panel_v3_ready",
    "model_specs",
    "exact_ground_truth_count",
    "abstention_precision",
    "rejection_recall",
    "abstention_coverage",
    "false_accept_rate",
    "false_reject_rate",
    "solve_accuracy",
    "verification_accuracy",
    "maxsat_policy_used",
    "thermodynamic_decode_telemetry",
    "prompt_hashes",
    "source_artifacts",
    "inference_substrate",
    "honest_verdict",
)
ANSWER_TOKENS = ("SAT", "UNSAT", "VALID", "INVALID", "REPAIRABLE", "UNREPAIRABLE", "UNKNOWN")
ACTION_TOKENS = ("ACCEPT", "REJECT", "ABSTAIN")


@dataclass(frozen=True)
class PanelConfig:
    """Runtime knobs for Exp 3099.

    Tests pass temporary paths and fake probes through this object. The default
    values point at the real repository artifact paths so `python -m` writes the
    deliverable requested by the conductor.
    """

    repo_root: Path = REPO_ROOT
    output_path: Path | None = None
    rows_path: Path | None = None
    minimum_live_eval_count: int | None = None
    preferred_quant: str = "Q4_K_M"
    decode_config: Mapping[str, Any] = field(
        default_factory=lambda: {
            "max_tokens": 16,
            "temperature": 0.0,
            "seed": DEFAULT_SEED,
            "logprobs": DEFAULT_LOGPROBS,
            "stop": ["\n\n", "</s>"],
        }
    )
    load_config: Mapping[str, Any] = field(
        default_factory=lambda: {
            "n_ctx": 2048,
            "n_gpu_layers": -1,
            "logits_all": False,
            "verbose": False,
        }
    )
    started_s: float | None = None
    clock: ClockFn = time.perf_counter
    tests_run: Sequence[str] = ()

    def artifact_path(self) -> Path:
        return self.output_path or self.repo_root / OUTPUT_REL_PATH

    def panel_rows_path(self) -> Path:
        return self.rows_path or self.repo_root / PANEL_ROWS_REL_PATH

    def start_time(self) -> float:
        return self.clock() if self.started_s is None else float(self.started_s)

    def effective_decode_config(self) -> JsonDict:
        return dict(self.decode_config)

    def effective_load_config(self, gpu: int) -> JsonDict:
        load = dict(self.load_config)
        load.setdefault("main_gpu", gpu)
        return load


def run_experiment(
    config: PanelConfig | None = None,
    *,
    resolve_gguf_func: ResolveGgufFn = resolve_cached_gguf,
    cached_pair_func: CachedPairFn = cached_sota_pair,
    llama_factory: LlamaFactory | None = None,
    cuda_probe_func: Callable[[], Mapping[str, Any]] | None = None,
    gpu_inventory_func: Callable[[], Mapping[str, Any]] | None = None,
    repo_commit_func: Callable[[Path], str] | None = None,
    python_environment_func: Callable[[], Mapping[str, Any]] | None = None,
) -> JsonDict:
    """Run the panel or write a blocked terminal artifact when preconditions fail."""

    active = config or PanelConfig()
    cuda_probe = cuda_probe_func or default_cuda_probe
    gpu_inventory_probe = gpu_inventory_func or default_gpu_inventory
    commit_probe = repo_commit_func or repo_commit
    env_probe = python_environment_func or python_environment
    started_s = active.start_time()
    exp3097 = safe_load_json(active.repo_root / EXP3097_REL_PATH)
    exp3098 = safe_load_json(active.repo_root / EXP3098_REL_PATH)
    policy = load_policy(active.repo_root, exp3098)
    maxsat_policy_used = bool(policy)
    manifest_path = active.repo_root / str(
        exp3097.get("stratified_eval_manifest_path") or STRATIFIED_MANIFEST_REL_PATH
    )
    manifest_rows = load_jsonl(manifest_path)
    minimum_count = int(
        active.minimum_live_eval_count
        or exp3097.get("minimum_live_eval_count")
        or 48
    )
    selected_rows = select_eval_rows(manifest_rows, minimum_count)
    cuda_status = dict(cuda_probe())
    gpu_inventory = dict(gpu_inventory_probe())
    cache_rows = model_cache_status(resolve_gguf_func, active.preferred_quant)
    cached_pair_status = exercise_cached_sota_pair(cached_pair_func)
    selected_model = select_model(cache_rows)
    source_rows = source_artifacts(active.repo_root)
    runtime_blocker = first_precondition_failure(
        exp3097=exp3097,
        exp3098=exp3098,
        policy=policy,
        manifest_rows=manifest_rows,
        selected_rows=selected_rows,
        minimum_count=minimum_count,
        cuda_status=cuda_status,
        selected_model=selected_model,
    )
    common = common_artifact_fields(
        active=active,
        exp3097=exp3097,
        exp3098=exp3098,
        source_rows=source_rows,
        policy_used=maxsat_policy_used,
        manifest_rows=manifest_rows,
        minimum_count=minimum_count,
        cache_rows=cache_rows,
        cached_pair_status=cached_pair_status,
        selected_model=selected_model,
        cuda_status=cuda_status,
        gpu_inventory=gpu_inventory,
        repo_commit_value=commit_probe(active.repo_root),
        python_env=env_probe(),
    )
    if runtime_blocker:
        artifact = blocked_artifact(
            common=common,
            active=active,
            started_s=started_s,
            runtime_blocker=runtime_blocker,
        )
        write_json(active.artifact_path(), artifact)
        validate_artifact(artifact)
        return artifact

    row_results, load_failure = run_live_rows(
        selected_rows,
        selected_model,
        policy,
        active,
        llama_factory=llama_factory,
    )
    if load_failure:
        artifact = blocked_artifact(
            common=common,
            active=active,
            started_s=started_s,
            runtime_blocker=load_failure,
        )
        write_json(active.artifact_path(), artifact)
        validate_artifact(artifact)
        return artifact

    write_jsonl(active.panel_rows_path(), row_results)
    metrics = metrics_from_rows(row_results)
    ready = (
        len(row_results) >= minimum_count
        and bool(selected_model)
        and common["maxsat_policy_used"] is True
    )
    artifact: JsonDict = {
        **common,
        **metrics,
        "abstention_panel_v3_ready": ready,
        "exact_ground_truth_count": len(row_results),
        "evaluated_fixture_count": len(row_results),
        "models_used": [selected_model["hf_id"]],
        "selected_model_ids": [selected_model["hf_id"]],
        "prompt_hashes": [row["prompt_hash"] for row in row_results],
        "prompt_hash_count": len(row_results),
        "panel_rows_path": relative_path(active.repo_root, active.panel_rows_path()),
        "panel_rows_sha256": sha256_file(active.panel_rows_path()),
        "thermodynamic_decode_telemetry": thermodynamic_decode_telemetry(row_results),
        "blocked_outcomes": [],
        "skipped_outcomes": [],
        "negative_outcomes": negative_outcomes(row_results),
        "runtime_blocker": None,
        "duration_s": active.clock() - started_s,
    }
    artifact["honest_verdict"] = honest_verdict(artifact)
    write_json(active.artifact_path(), artifact)
    validate_artifact(artifact)
    return artifact


def run_live_rows(
    rows: Sequence[Mapping[str, Any]],
    selected_model: Mapping[str, Any] | None,
    policy: Mapping[str, Any],
    config: PanelConfig,
    *,
    llama_factory: LlamaFactory | None,
) -> tuple[list[JsonDict], str | None]:
    """Generate model answers and route every row through the loaded policy."""

    if selected_model is None:
        return [], "no_mandated_gguf_resolved"
    try:
        llama = (llama_factory or default_llama_factory)(
            model_path=selected_model["model_path"],
            **config.effective_load_config(int(selected_model.get("gpu", 0))),
        )
    except Exception as exc:
        return [], f"model_load_failed:{type(exc).__name__}:{exc}"

    results: list[JsonDict] = []
    try:
        for index, row in enumerate(rows):
            prompt = build_prompt(row)
            prompt_hash = sha256_text(prompt)
            try:
                output = llama(prompt, **config.effective_decode_config())
            except Exception as exc:
                return results, f"generation_failed:{type(exc).__name__}:{exc}"
            choice = first_choice(output)
            text = str(choice.get("text") or "")
            parsed = parse_response(text)
            confidence = confidence_from_output(output)
            effective_confidence = parsed["verbal_confidence"]
            if effective_confidence is None and confidence["confidence_available"]:
                effective_confidence = confidence["confidence_score"]
            exact_match = answer_matches(parsed["answer"], str(row.get("expected_answer", "")))
            route_case = route_case_from_row(row, exact_match, effective_confidence, len(rows))
            route = maxsat_policy.evaluate_route(route_case, policy=policy)
            results.append(
                {
                    "row_index": index,
                    "source_fixture_id": row.get("source_fixture_id"),
                    "task_family": row.get("task_family"),
                    "perturbation_type": row.get("perturbation_type"),
                    "expected_answer": row.get("expected_answer"),
                    "expected_action": row.get("verifier_target", {}).get("expected_action"),
                    "prompt_hash": prompt_hash,
                    "raw_output_hash": sha256_text(text),
                    "raw_action": parsed["raw_action"],
                    "parsed_answer": parsed["answer"],
                    "verbal_confidence": parsed["verbal_confidence"],
                    "confidence": effective_confidence,
                    "confidence_signal": confidence["confidence_signal"],
                    "confidence_available": confidence["confidence_available"],
                    "first_token_entropy": confidence["first_token_entropy"],
                    "first_token_negative_logprob": confidence["first_token_negative_logprob"],
                    "exact_answer_match": exact_match,
                    "route_decision": route["decision"],
                    "route_scores": route["scores"],
                    "route_hard_feasible_actions": route["hard_feasible_actions"],
                    "maxsat_policy_used": True,
                }
            )
    finally:
        close = getattr(llama, "close", None)
        if callable(close):
            close()
    return results, None


def route_case_from_row(
    row: Mapping[str, Any],
    exact_match: bool,
    confidence: float | None,
    exact_count: int,
) -> JsonDict:
    """Build the policy-evaluation record for one exact fixture row."""

    repair_target = (
        row.get("repair_target") if isinstance(row.get("repair_target"), Mapping) else {}
    )
    return {
        "expected_action": row.get("verifier_target", {}).get("expected_action"),
        "exact_label_match": exact_match,
        "model_cache_available": True,
        "headline_claim": True,
        "exact_ground_truth_count": exact_count,
        "minimum_live_eval_count": exact_count,
        "syntax_valid": True,
        "schema_valid": True,
        "repair_candidate": bool(repair_target.get("applicable", False)),
        "repair_intent_preserved": not bool(repair_target.get("applicable", False)),
        "repair_promotion": False,
        "formal_feedback_delta": 0.0,
        "confidence": confidence if confidence is not None else 0.0,
    }


def common_artifact_fields(
    *,
    active: PanelConfig,
    exp3097: Mapping[str, Any],
    exp3098: Mapping[str, Any],
    source_rows: list[JsonDict],
    policy_used: bool,
    manifest_rows: Sequence[Mapping[str, Any]],
    minimum_count: int,
    cache_rows: list[JsonDict],
    cached_pair_status: JsonDict,
    selected_model: Mapping[str, Any] | None,
    cuda_status: JsonDict,
    gpu_inventory: JsonDict,
    repo_commit_value: str,
    python_env: Mapping[str, Any],
) -> JsonDict:
    """Return fields shared by ready and blocked terminal artifacts."""

    return {
        "artifact": ARTIFACT,
        "schema": SCHEMA,
        "run_date": RUN_DATE,
        "source_artifacts": source_rows,
        "maxsat_policy_used": policy_used,
        "maxsat_policy_path": str(exp3098.get("routing_policy_path") or POLICY_REL_PATH),
        "exact_protocol_ready": exp3097.get("eval_protocol_ready") is True,
        "maxsat_policy_ready": exp3098.get("maxsat_policy_ready") is True,
        "available_exact_fixture_count": len(manifest_rows),
        "minimum_live_eval_count": minimum_count,
        "model_specs": cache_rows,
        "cached_sota_pair": cached_pair_status,
        "cache_missing_outcomes": [
            row["hf_id"] for row in cache_rows if row["cache_status"] == "cache_missing"
        ],
        "legacy_smoke_only_used": False,
        "decode_config": active.effective_decode_config(),
        "inference_substrate": {
            "kind": "local_sota_gguf_llama_cpp_or_blocked_preflight",
            "cuda": cuda_status,
            "gpu_inventory": gpu_inventory,
            "python": dict(python_env),
            "repo_commit": repo_commit_value,
            "executes_models": selected_model is not None,
            "live_llm_calls_planned": minimum_count if selected_model is not None else 0,
            "legacy_tiny_models_promoted": False,
        },
        "tests_or_checks_run": list(active.tests_run),
    }


def blocked_artifact(
    *,
    common: JsonDict,
    active: PanelConfig,
    started_s: float,
    runtime_blocker: str,
) -> JsonDict:
    """Return a complete blocked artifact with the required metric fields."""

    artifact: JsonDict = {
        **common,
        **blocked_metrics(),
        "abstention_panel_v3_ready": False,
        "exact_ground_truth_count": 0,
        "evaluated_fixture_count": 0,
        "models_used": [],
        "selected_model_ids": [],
        "prompt_hashes": [],
        "prompt_hash_count": 0,
        "panel_rows_path": relative_path(active.repo_root, active.panel_rows_path()),
        "panel_rows_sha256": None,
        "thermodynamic_decode_telemetry": thermodynamic_decode_telemetry([]),
        "blocked_outcomes": [runtime_blocker],
        "skipped_outcomes": ["live_exact_fixture_eval"],
        "negative_outcomes": [],
        "runtime_blocker": runtime_blocker,
        "duration_s": active.clock() - started_s,
    }
    artifact["honest_verdict"] = honest_verdict(artifact)
    return artifact


def first_precondition_failure(
    *,
    exp3097: Mapping[str, Any],
    exp3098: Mapping[str, Any],
    policy: Mapping[str, Any],
    manifest_rows: Sequence[Mapping[str, Any]],
    selected_rows: Sequence[Mapping[str, Any]],
    minimum_count: int,
    cuda_status: Mapping[str, Any],
    selected_model: Mapping[str, Any] | None,
) -> str | None:
    """Return the first failed precondition in fail-closed order."""

    if exp3097.get("eval_protocol_ready") is not True:
        return "exact_eval_protocol_unavailable"
    if exp3098.get("maxsat_policy_ready") is not True or not policy:
        return "maxsat_policy_unavailable"
    if len(manifest_rows) < minimum_count or len(selected_rows) < minimum_count:
        return "minimum_live_eval_count_unavailable"
    if cuda_status.get("cuda_available") is not True:
        return "cuda_unavailable"
    if selected_model is None:
        return "no_mandated_gguf_resolved"
    return None


def load_policy(repo_root: Path, exp3098: Mapping[str, Any]) -> JsonDict:
    """Load the exact policy file named by Exp 3098, returning empty on failure."""

    rel_path = Path(str(exp3098.get("routing_policy_path") or POLICY_REL_PATH))
    policy = safe_load_json(repo_root / rel_path)
    if policy.get("schema") != maxsat_policy.POLICY_SCHEMA:
        return {}
    return policy


def load_jsonl(path: Path) -> list[JsonDict]:
    """Load newline-delimited JSON rows, returning an empty list for absent files."""

    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    rows: list[JsonDict] = []
    for line in lines:
        if line.strip():
            rows.append(json.loads(line))
    return rows


def select_eval_rows(rows: Sequence[Mapping[str, Any]], minimum_count: int) -> list[JsonDict]:
    """Select a deterministic balanced prefix over perturbation strata."""

    grouped: dict[str, list[JsonDict]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("perturbation_type") or row.get("stratum_key") or "unknown")].append(
            dict(row)
        )
    for group_rows in grouped.values():
        group_rows.sort(key=lambda item: str(item.get("source_fixture_id", "")))
    selected: list[JsonDict] = []
    keys = sorted(grouped)
    cursor = 0
    while len(selected) < minimum_count and any(grouped.values()):
        key = keys[cursor % len(keys)]
        if grouped[key]:
            selected.append(grouped[key].pop(0))
        cursor += 1
    return selected


def model_cache_status(resolve_func: ResolveGgufFn, preferred_quant: str) -> list[JsonDict]:
    """Resolve every mandated GGUF and record exact cache status."""

    statuses: list[JsonDict] = []
    for model in SOTA_GGUF_MODELS:
        path = resolve_func(model["hf_id"], preferred_quant)
        exists = bool(path and Path(path).is_file())
        statuses.append(
            {
                "name": model["name"],
                "hf_id": model["hf_id"],
                "role": model["role"],
                "quantization": model["quantization"],
                "preferred_quant": preferred_quant,
                "cache_status": "cached" if exists else "cache_missing",
                "cache_present": exists,
                "model_path": str(path) if path else None,
                "selected": False,
                "file_size_bytes": Path(path).stat().st_size if exists else None,
            }
        )
    return statuses


def exercise_cached_sota_pair(cached_pair_func: CachedPairFn) -> JsonDict:
    """Call cached_sota_pair or its injected equivalent and summarize readiness."""

    try:
        pair = cached_pair_func(gpu_indices=(0, 1))
    except Exception as exc:
        return {
            "called": True,
            "ready": False,
            "error": f"{type(exc).__name__}:{exc}",
            "result": None,
        }
    return {
        "called": True,
        "ready": bool(pair),
        "error": None,
        "result": pair or None,
        "model_ids": [row.get("hf_id") for row in pair] if pair else [],
    }


def select_model(cache_rows: list[JsonDict]) -> JsonDict | None:
    """Pick the first cached mandated model for the bounded panel."""

    for index, row in enumerate(cache_rows):
        if row["cache_status"] == "cached":
            cache_rows[index] = dict(row) | {"selected": True, "gpu": 0}
            return cache_rows[index]
    return None


def build_prompt(row: Mapping[str, Any]) -> str:
    """Build a leakage-safe prompt from the manifest payload only."""

    payload = json.dumps(
        row.get("leakage_safe_prompt_payload", {}),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    case_hash = str(row.get("source_prompt_payload_sha256") or "")
    return "\n".join(
        [
            "You are checking an exact fixture without access to hidden labels.",
            f"Case Hash: {case_hash}",
            f"Task Family: {row.get('task_family', 'unknown')}",
            f"Prompt Payload JSON: {payload}",
            "Return one line: ACTION | answer=ANSWER | confidence=0.00",
            "ACTION must be ACCEPT, REJECT, or ABSTAIN.",
            "ANSWER must be SAT, UNSAT, VALID, INVALID, REPAIRABLE, UNREPAIRABLE, or UNKNOWN.",
        ]
    )


def parse_response(text: str) -> JsonDict:
    """Extract action, answer, and verbal confidence from a model response."""

    upper = text.upper()
    action = next(
        (token.lower() for token in ACTION_TOKENS if re.search(rf"\b{token}\b", upper)),
        None,
    )
    answer = next(
        (
            token
            for token in sorted(ANSWER_TOKENS, key=len, reverse=True)
            if re.search(rf"\b{token}\b", upper)
        ),
        None,
    )
    confidence = verbal_confidence(text)
    return {"raw_action": action, "answer": answer, "verbal_confidence": confidence}


def verbal_confidence(text: str) -> float | None:
    """Parse numeric or coarse natural-language confidence into [0, 1]."""

    match = re.search(r"confidence\s*[:=]\s*(-?\d+(?:\.\d+)?)", text, flags=re.I)
    if match:
        return clamp01(float(match.group(1)))
    lowered = text.lower()
    if "high confidence" in lowered:
        return 0.9
    if "medium confidence" in lowered:
        return 0.5
    if "low confidence" in lowered:
        return 0.2
    return None


def confidence_from_output(output: Mapping[str, Any]) -> JsonDict:
    """Return first-token confidence and entropy diagnostics when available."""

    choice = first_choice(output)
    logprobs = choice.get("logprobs")
    if not isinstance(logprobs, Mapping):
        return {
            "confidence_available": False,
            "confidence_signal": "unavailable",
            "confidence_score": None,
            "first_token_entropy": None,
            "first_token_negative_logprob": None,
        }
    token_logprobs = list(logprobs.get("token_logprobs") or [])
    top_logprobs = list(logprobs.get("top_logprobs") or [])
    first_logprob = float_or_none(token_logprobs[0]) if token_logprobs else None
    entropy = topk_entropy(top_logprobs[0]) if top_logprobs else None
    if first_logprob is None:
        return {
            "confidence_available": False,
            "confidence_signal": "unavailable",
            "confidence_score": None,
            "first_token_entropy": entropy,
            "first_token_negative_logprob": None,
        }
    return {
        "confidence_available": True,
        "confidence_signal": "first_token_logprob_proxy",
        "confidence_score": clamp01(math.exp(first_logprob)),
        "first_token_entropy": entropy,
        "first_token_negative_logprob": -first_logprob,
    }


def topk_entropy(top_logprobs: Any) -> float | None:
    """Compute normalized entropy from llama.cpp top-logprob telemetry."""

    if not isinstance(top_logprobs, Mapping) or not top_logprobs:
        return None
    probs = [math.exp(float(value)) for value in top_logprobs.values()]
    total = sum(probs)
    if total <= 0.0:
        return None
    normalized = [prob / total for prob in probs]
    return -sum(prob * math.log(prob) for prob in normalized if prob > 0.0)


def first_choice(output: Mapping[str, Any]) -> JsonDict:
    """Return the first llama.cpp choice object or an empty mapping."""

    choices = output.get("choices")
    if isinstance(choices, list) and choices and isinstance(choices[0], Mapping):
        return dict(choices[0])
    return {}


def metrics_from_rows(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Compute direct confidence/abstention safety metrics."""

    if not rows:
        return blocked_metrics()
    total = len(rows)
    counts = Counter(str(row.get("route_decision")) for row in rows)
    bad_targets = [
        row
        for row in rows
        if row.get("expected_action") == "reject" or row.get("exact_answer_match") is not True
    ]
    safe_accept_targets = [
        row
        for row in rows
        if row.get("expected_action") == "accept" and row.get("exact_answer_match") is True
    ]
    abstained = [row for row in rows if row.get("route_decision") == "abstain"]
    return {
        "solve_accuracy": ratio(
            sum(row.get("exact_answer_match") is True for row in rows),
            total,
        ),
        "verification_accuracy": ratio(
            sum(row.get("route_decision") == row.get("expected_action") for row in rows),
            total,
        ),
        "abstention_precision": ratio(
            sum(row in bad_targets for row in abstained),
            len(abstained),
        ),
        "rejection_recall": ratio(
            sum(row.get("route_decision") == "reject" for row in bad_targets),
            len(bad_targets),
        ),
        "abstention_coverage": ratio(len(abstained), total),
        "false_accept_rate": ratio(
            sum(row.get("route_decision") == "accept" for row in bad_targets),
            len(bad_targets),
        ),
        "false_reject_rate": ratio(
            sum(row.get("route_decision") == "reject" for row in safe_accept_targets),
            len(safe_accept_targets),
        ),
        "route_decision_counts": {
            "accept": counts.get("accept", 0),
            "reject": counts.get("reject", 0),
            "abstain": counts.get("abstain", 0),
        },
    }


def blocked_metrics() -> JsonDict:
    """Return zero-valued metric fields for fail-closed artifacts."""

    return {
        "solve_accuracy": 0.0,
        "verification_accuracy": 0.0,
        "abstention_precision": 0.0,
        "rejection_recall": 0.0,
        "abstention_coverage": 0.0,
        "false_accept_rate": 0.0,
        "false_reject_rate": 0.0,
        "route_decision_counts": {"accept": 0, "reject": 0, "abstain": 0},
    }


def thermodynamic_decode_telemetry(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Summarize optional entropy/free-energy-style diagnostics only."""

    entropies = [
        float(row["first_token_entropy"])
        for row in rows
        if row.get("first_token_entropy") is not None
    ]
    negative_logprobs = [
        float(row["first_token_negative_logprob"])
        for row in rows
        if row.get("first_token_negative_logprob") is not None
    ]
    return {
        "available": bool(entropies or negative_logprobs),
        "diagnostic_only": True,
        "reference": "arXiv:2604.07867",
        "row_count": len(rows),
        "mean_first_token_entropy": ratio(sum(entropies), len(entropies)),
        "mean_first_token_negative_logprob": ratio(sum(negative_logprobs), len(negative_logprobs)),
        "free_energy_proxy": ratio(sum(negative_logprobs) + sum(entropies), len(rows)),
        "gate_used": False,
    }


def negative_outcomes(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    """List row-level negative outcome categories without hiding zeros."""

    metrics = metrics_from_rows(rows)
    outcomes: list[str] = []
    if metrics["false_accept_rate"] > 0.0:
        outcomes.append("false_accepts_observed")
    if metrics["false_reject_rate"] > 0.0:
        outcomes.append("false_rejects_observed")
    if metrics["solve_accuracy"] < 1.0:
        outcomes.append("solver_answer_errors_observed")
    if metrics["abstention_precision"] == 0.0:
        outcomes.append("no_correct_abstentions_observed")
    return outcomes


def answer_matches(answer: str | None, expected: str) -> bool:
    """Return whether a parsed answer matches the exact manifest answer."""

    if answer is None:
        return False
    normalized = re.sub(r"[^A-Z]", "", answer.upper())
    return normalized == expected.upper()


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Raise when the terminal artifact overstates readiness."""

    missing = sorted(set(REQUIRED_ARTIFACT_FIELDS) - set(artifact))
    if missing:
        raise ValueError(f"missing required fields: {missing}")
    if not artifact.get("model_specs"):
        raise ValueError("model_specs must record mandated cache status")
    verdict = str(artifact.get("honest_verdict") or "")
    if artifact.get("abstention_panel_v3_ready") is True:
        if not verdict.startswith(SUCCESS_PREFIXES):
            raise ValueError("ready artifact honest_verdict must start with a success prefix")
        if artifact.get("maxsat_policy_used") is not True:
            raise ValueError("ready artifact requires maxsat_policy_used=true")
        if not artifact.get("prompt_hashes"):
            raise ValueError("ready artifact requires prompt_hashes")
        if int(artifact.get("exact_ground_truth_count") or 0) < int(
            artifact.get("minimum_live_eval_count") or 0
        ):
            raise ValueError("ready artifact must meet minimum_live_eval_count")
    elif not verdict.startswith("blocked_sota_or_panel_precondition_failed"):
        raise ValueError("blocked artifact must use blocked_sota_or_panel_precondition_failed")


def honest_verdict(artifact: Mapping[str, Any]) -> str:
    """Map readiness and blocker state to the conductor verdict vocabulary."""

    if artifact.get("abstention_panel_v3_ready") is True:
        return (
            "complete: abstention_panel_v3_ready=true; "
            f"exact_ground_truth_count={artifact.get('exact_ground_truth_count')}; "
            f"abstention_precision={artifact.get('abstention_precision')}; "
            f"rejection_recall={artifact.get('rejection_recall')}; "
            f"false_accept_rate={artifact.get('false_accept_rate')}"
        )
    return (
        "blocked_sota_or_panel_precondition_failed: "
        f"{artifact.get('runtime_blocker') or 'unknown_precondition'}"
    )


def source_artifacts(repo_root: Path) -> list[JsonDict]:
    """Return local source artifact checksums for reproducibility."""

    rows: list[JsonDict] = []
    for source_id, rel_path, role in SOURCE_REL_PATHS:
        path = repo_root / rel_path
        exists = path.is_file()
        rows.append(
            {
                "id": source_id,
                "path": rel_path.as_posix(),
                "role": role,
                "exists": exists,
                "sha256": sha256_file(path) if exists else None,
            }
        )
    return rows


def default_llama_factory(**kwargs: Any) -> Any:  # pragma: no cover - real GGUF runtime path.
    """Build llama.cpp lazily so unit tests never import or load GGUF weights."""

    from llama_cpp import Llama

    return Llama(**kwargs)


def default_cuda_probe() -> JsonDict:  # pragma: no cover - environment probe.
    """Return CUDA availability using the shared experiment-template probe."""

    from scripts.experiment_template import _cuda_is_available, _detect_gpu_count_rocm_aware

    return {
        "cuda_available": _cuda_is_available(),
        "gpu_count": _detect_gpu_count_rocm_aware(),
    }


def default_gpu_inventory() -> JsonDict:  # pragma: no cover - environment probe.
    """Return a lightweight nvidia-smi inventory when available."""

    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total,memory.free",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return {"available": False, "gpus": []}
    gpus = []
    for line in result.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) == 4:
            gpus.append(
                {
                    "index": int(parts[0]),
                    "name": parts[1],
                    "memory_total_mib": int(parts[2]),
                    "memory_free_mib": int(parts[3]),
                }
            )
    return {"available": result.returncode == 0, "gpus": gpus}


def repo_commit(repo_root: Path) -> str:  # pragma: no cover - environment probe.
    """Return the current git commit or `unknown` outside a git checkout."""

    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
        timeout=5,
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def python_environment() -> JsonDict:  # pragma: no cover - environment probe.
    """Return Python executable and version metadata for the artifact."""

    return {"executable": sys.executable, "version": sys.version.split()[0]}


def safe_load_json(path: Path) -> JsonDict:
    """Load a JSON object, returning empty for missing or malformed input."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Write stable JSON so reruns produce deterministic checksums."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    """Write a stable row transcript."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def sha256_file(path: Path) -> str:
    """Return SHA-256 for a local file."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    """Return SHA-256 for a UTF-8 string."""

    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def relative_path(root: Path, path: Path) -> str:
    """Return a stable repo-relative path when possible."""

    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def ratio(numerator: float, denominator: int) -> float:
    """Return a rounded ratio, using 0.0 for empty denominators."""

    if denominator == 0:
        return 0.0
    return round(float(numerator) / float(denominator), 6)


def float_or_none(value: Any) -> float | None:
    """Parse a float defensively."""

    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def clamp01(value: float) -> float:
    """Clamp a numeric confidence into [0, 1]."""

    return min(1.0, max(0.0, value))


def main() -> int:  # pragma: no cover - CLI wrapper.
    """Write the requested Exp 3099 artifact from the repository root."""

    artifact = run_experiment()
    print(
        json.dumps(
            {"path": OUTPUT_REL_PATH.as_posix(), "honest_verdict": artifact["honest_verdict"]}
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI wrapper.
    raise SystemExit(main())
