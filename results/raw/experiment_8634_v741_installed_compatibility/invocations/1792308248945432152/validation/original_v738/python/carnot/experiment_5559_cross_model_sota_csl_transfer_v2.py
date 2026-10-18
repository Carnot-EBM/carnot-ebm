"""Exp5559 causal cross-model local-SOTA CSL transfer v2.

Spec refs: REQ-LEARN-5559,
SCENARIO-LEARN-5559-UPSTREAM-GATE,
SCENARIO-LEARN-5559-CROSS-FAMILY,
SCENARIO-LEARN-5559-STALE-AND-NEGATIVE-GATES,
SCENARIO-LEARN-5559-NO-WEIGHT-MUTATION,
SCENARIO-LEARN-5559-ARTIFACT.

This retry deliberately starts from the Exp5558 causal memory fixture rather
than the older shuffled-memory transfer fixture. The only state that can move
between model families is external memory: source GGUF calls emit candidate
memory rows, a verifier accepts exact causal rows, and target GGUF calls are
scored against independent Exp5558 decision labels. Model file receipts are
checked before and after so a positive gate cannot be explained by hidden
training or adapter writes.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re
import time
from typing import Any

from carnot import experiment_5544_cross_model_sota_csl_transfer as exp5544


JsonDict = dict[str, Any]
ModelResolver = Callable[[str, str], str | None]
CachedPairFn = Callable[[], list[Mapping[str, Any]] | None]
RuntimeProbe = Callable[..., Mapping[str, Any]]
GenerationRunner = Callable[..., Mapping[str, Any]]

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path(
    "results/experiment_5559_cross_model_sota_csl_transfer_v2.json"
)
UPSTREAM_CAUSAL_MEMORY_PATH = Path(
    "results/experiment_5558_causal_write_manage_read_csl_memory.json"
)
SPEC_RELATIVE_PATH = Path("openspec/capabilities/self-learning/spec.md")
MODULE_RELATIVE_PATH = Path(
    "python/carnot/experiment_5559_cross_model_sota_csl_transfer_v2.py"
)
TEST_RELATIVE_PATH = Path(
    "tests/python/test_experiment_5559_cross_model_sota_csl_transfer_v2.py"
)

SCHEMA = "carnot.experiment_5559.cross_model_sota_csl_transfer_v2.v1"
EXPERIMENT_ID = "experiment_5559_cross_model_sota_csl_transfer_v2"
TASK_ID = "exp5559-cross-model-sota-csl-transfer-v2"
MILESTONE = "2026.07.503"
RUN_DATE = "2026-07-10"
RANDOM_SEED = 5559
N_GPU_LAYERS = -1
MAX_TOKENS = 16
INFERENCE_SUBSTRATE = "live_local_sota_gguf_cross_model_csl_transfer_or_gate_skip"

QWEN_HF_ID = exp5544.QWEN_HF_ID
GEMMA_26_HF_ID = exp5544.GEMMA_26_HF_ID
GEMMA_31_HF_ID = exp5544.GEMMA_31_HF_ID
MANDATED_HF_IDS = exp5544.MANDATED_HF_IDS
DEFAULT_QUANTIZATION = exp5544.DEFAULT_QUANTIZATION

NO_MEMORY_ARM = "no_memory"
SHUFFLED_MEMORY_ARM = "shuffled_memory"
STALE_MEMORY_ARM = "stale_memory"
ALIGNED_CAUSAL_MEMORY_ARM = "aligned_causal_memory"
TARGET_ARMS = (
    NO_MEMORY_ARM,
    SHUFFLED_MEMORY_ARM,
    STALE_MEMORY_ARM,
    ALIGNED_CAUSAL_MEMORY_ARM,
)
ARM_SCORE_FIELDS = {
    NO_MEMORY_ARM: "no_memory_score",
    SHUFFLED_MEMORY_ARM: "shuffled_memory_score",
    STALE_MEMORY_ARM: "stale_memory_score",
    ALIGNED_CAUSAL_MEMORY_ARM: "aligned_memory_score",
}

SPEC_REFS = (
    "REQ-LEARN-5559",
    "SCENARIO-LEARN-5559-UPSTREAM-GATE",
    "SCENARIO-LEARN-5559-CROSS-FAMILY",
    "SCENARIO-LEARN-5559-STALE-AND-NEGATIVE-GATES",
    "SCENARIO-LEARN-5559-NO-WEIGHT-MUTATION",
    "SCENARIO-LEARN-5559-ARTIFACT",
)
REQUIRED_ARTIFACT_FIELDS = (
    "model_specs",
    "upstream_causal_memory",
    "live_model_invoked",
    "source_models",
    "target_models",
    "no_memory_score",
    "shuffled_memory_score",
    "stale_memory_score",
    "aligned_memory_score",
    "cross_family_delta_over_shuffled",
    "negative_transfer_rate",
    "stale_evidence_rejection_rate",
    "no_weight_mutation",
    "measured_duration_s",
    "gpu_offload_evidence",
    "csl_claim_allowed",
    "tests_added_or_reused",
    "field_principles",
    "inference_substrate",
    "honest_verdict",
)
DEFAULT_TESTS_ADDED_OR_REUSED = (
    ".venv/bin/pytest "
    "tests/python/test_experiment_5559_cross_model_sota_csl_transfer_v2.py "
    "-q --no-cov -n 0",
    ".venv/bin/coverage run "
    "--include=python/carnot/experiment_5559_cross_model_sota_csl_transfer_v2.py "
    "-m pytest tests/python/test_experiment_5559_cross_model_sota_csl_transfer_v2.py "
    "-q --no-cov -n 0 && .venv/bin/coverage report "
    "--include=python/carnot/experiment_5559_cross_model_sota_csl_transfer_v2.py "
    "--fail-under=100",
    ".venv/bin/pytest tests/python -q",
)
FIELD_PRINCIPLES: JsonDict = {
    "model_specs": "Lists mandated SOTA GGUF models with local availability and receipts.",
    "upstream_causal_memory": "Exact Exp5558 causal memory artifact path used as the source fixture.",
    "live_model_invoked": "Bare boolean proving whether the live local GGUF path ran or was gate-skipped.",
    "source_models": "Names frozen source-model families used to emit external causal memory.",
    "target_models": "Names frozen target-model families evaluated on Exp5558 decisions.",
    "no_memory_score": "Baseline target score when no causal memory is supplied.",
    "shuffled_memory_score": "Control target score when memory context links are broken.",
    "stale_memory_score": "Control target score when stale or contradicted memory is exposed.",
    "aligned_memory_score": "Headline target score for aligned causal memory from another family.",
    "cross_family_delta_over_shuffled": "Gate delta proving aligned causal memory beats shuffled memory.",
    "negative_transfer_rate": "Blocks claims when aligned cross-family memory harms target decisions.",
    "stale_evidence_rejection_rate": "Blocks claims unless stale or contradicted memory is rejected.",
    "no_weight_mutation": "Bare frozen-weight gate from before/after GGUF file receipts.",
    "measured_duration_s": "Measured wall-clock duration for the artifact-producing run.",
    "gpu_offload_evidence": "Records runtime backend, CUDA visibility, offload receipts, and blockers.",
    "csl_claim_allowed": "Final gate requiring causal transfer benefit and safety controls.",
    "tests_added_or_reused": "Lists focused, coverage, and full-suite verification commands.",
    "field_principles": "Explains why each required headline and gate field exists.",
    "inference_substrate": "Declares live local SOTA GGUF cross-model CSL or honest gate skip.",
    "honest_verdict": "Terminal complete or blocked summary for reconciliation.",
}


def run(
    *,
    root: Path | str = REPO_ROOT,
    result_path: Path | str = RESULT_RELATIVE_PATH,
    tests_added_or_reused: Sequence[str] = DEFAULT_TESTS_ADDED_OR_REUSED,
    model_resolver: ModelResolver = exp5544.resolve_cached_gguf,
    cached_pair_fn: CachedPairFn = exp5544.cached_sota_pair,
    runtime_probe: RuntimeProbe | None = None,
    generation_runner: GenerationRunner | None = None,
    write: bool = True,
    random_seed: int = RANDOM_SEED,
) -> JsonDict:
    """Run the causal transfer gate and optionally write its JSON receipt."""

    started = time.perf_counter()
    root_path = Path(root)
    upstream = load_upstream_causal_memory(root_path)
    fixture = fixture_from_upstream(upstream)
    model_specs, cache_receipt = exp5544.resolve_model_specs(
        model_resolver=model_resolver,
        cached_pair_fn=cached_pair_fn,
    )
    roles = select_cross_family_roles(model_specs)
    runtime_fn = runtime_probe or exp5544.default_runtime_probe
    runtime_receipt = dict(runtime_fn(model_specs=model_specs, n_gpu_layers=N_GPU_LAYERS))
    preconditions = evaluate_preconditions(
        upstream=upstream,
        fixture=fixture,
        model_specs=model_specs,
        roles=roles,
        runtime_receipt=runtime_receipt,
    )
    selected_specs = selected_role_specs(roles)
    before_receipts = {
        spec["hf_id"]: exp5544.model_file_receipt(spec.get("model_path"))
        for spec in selected_specs
    }

    runner = generation_runner
    if preconditions["all_passed"] and runner is None:  # pragma: no cover - live GGUF path.
        runner = exp5544.LiveLlamaGenerationRunner(n_gpu_layers=N_GPU_LAYERS)

    source_attempts: list[JsonDict] = []
    memory_entries: JsonDict = {"aligned": [], "shuffled": [], "stale": []}
    target_evaluations = {arm: [] for arm in TARGET_ARMS}
    row_results: list[JsonDict] = []
    live_model_invoked = False

    if preconditions["all_passed"] and runner is not None:
        source_attempts = run_source_attempts(
            fixture=fixture,
            source_model=roles["source"],
            generation_runner=runner,
            random_seed=random_seed,
        )
        memory_entries = build_memory_entries(fixture, source_attempts)
        target_evaluations = run_target_evaluations(
            fixture=fixture,
            target_model=roles["target"],
            memory_entries=memory_entries,
            generation_runner=runner,
            random_seed=random_seed,
        )
        row_results = [row for arm in TARGET_ARMS for row in target_evaluations[arm]]
        live_model_invoked = bool(source_attempts or row_results)
        runtime_receipt.update(getattr(runner, "runtime_receipt", {}))

    after_receipts = {
        spec["hf_id"]: exp5544.model_file_receipt(spec.get("model_path"))
        for spec in selected_specs
    }
    artifact = build_artifact(
        upstream=upstream,
        fixture=fixture,
        model_specs=model_specs,
        cache_receipt=cache_receipt,
        roles=roles,
        runtime_receipt=runtime_receipt,
        preconditions=preconditions,
        source_attempts=source_attempts,
        memory_entries=memory_entries,
        target_evaluations=target_evaluations,
        row_results=row_results,
        before_receipts=before_receipts,
        after_receipts=after_receipts,
        live_model_invoked=live_model_invoked,
        tests_added_or_reused=tests_added_or_reused,
        measured_duration_s=time.perf_counter() - started,
        random_seed=random_seed,
    )
    validate_artifact(artifact)
    if write:
        write_json(resolve_path(root_path, result_path), artifact)
    return artifact


def load_upstream_causal_memory(root: Path | str) -> JsonDict:
    """Load Exp5558 and keep its fixture available for causal-memory replay."""

    path = resolve_path(Path(root), UPSTREAM_CAUSAL_MEMORY_PATH)
    try:
        artifact = load_json(path)
    except (OSError, json.JSONDecodeError):
        return {
            "path": UPSTREAM_CAUSAL_MEMORY_PATH.as_posix(),
            "loadable": False,
            "csl_claim_allowed": False,
            "csl_memory_ready": False,
            "fixture_hash": None,
            "artifact": None,
        }
    fixture_payload = {
        "events": artifact.get("events", []),
        "decisions": artifact.get("decisions", []),
        "managed_memory": artifact.get("managed_memory", {}),
        "write_evidence": artifact.get("write_evidence", {}),
    }
    return {
        "path": UPSTREAM_CAUSAL_MEMORY_PATH.as_posix(),
        "loadable": True,
        "csl_claim_allowed": artifact.get("csl_claim_allowed") is True,
        "csl_memory_ready": artifact.get("csl_memory_ready") is True,
        "honest_verdict": str(artifact.get("honest_verdict", "")),
        "inference_substrate": str(artifact.get("inference_substrate", "")),
        "fixture_hash": "sha256:" + sha256_json(fixture_payload),
        "artifact": artifact,
    }


def fixture_from_upstream(upstream: Mapping[str, Any]) -> JsonDict:
    """Extract the Exp5558 decisions and managed memory rows used by this retry."""

    artifact = upstream.get("artifact")
    if not isinstance(artifact, Mapping):
        return {
            "events": [],
            "decisions": [],
            "active_entries": [],
            "forgotten_entries": [],
            "accepted_entries": [],
            "fixture_hash": None,
        }
    fixture = {
        "events": deepcopy(artifact.get("events", [])),
        "decisions": deepcopy(artifact.get("decisions", [])),
        "active_entries": deepcopy(
            artifact.get("managed_memory", {}).get("active_entries", [])
        ),
        "forgotten_entries": deepcopy(
            artifact.get("managed_memory", {}).get("forgotten_entries", [])
        ),
        "accepted_entries": deepcopy(
            artifact.get("write_evidence", {}).get("accepted_entries", [])
        ),
        "fixture_hash": upstream.get("fixture_hash"),
    }
    return fixture


def public_upstream_status(upstream: Mapping[str, Any]) -> JsonDict:
    """Return upstream status without embedding the full Exp5558 artifact again."""

    return {key: value for key, value in upstream.items() if key != "artifact"}


def select_cross_family_roles(model_specs: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Select one Qwen source and one Gemma target when both families are local."""

    available = [
        dict(row)
        for row in model_specs
        if row.get("local_path_available") is True
    ]
    by_id = {str(row.get("hf_id")): row for row in available}
    source = by_id.get(QWEN_HF_ID)
    target = by_id.get(GEMMA_31_HF_ID) or by_id.get(GEMMA_26_HF_ID)
    if source and target and source.get("family") != target.get("family"):
        return {"source": source, "target": target}

    for candidate_source in available:
        for candidate_target in available:
            if (
                candidate_source.get("hf_id") != candidate_target.get("hf_id")
                and candidate_source.get("family") != candidate_target.get("family")
            ):
                return {"source": candidate_source, "target": candidate_target}
    return {"source": None, "target": None}


def evaluate_preconditions(
    *,
    upstream: Mapping[str, Any],
    fixture: Mapping[str, Any],
    model_specs: Sequence[Mapping[str, Any]],
    roles: Mapping[str, Any],
    runtime_receipt: Mapping[str, Any],
) -> JsonDict:
    """Combine upstream, fixture, model-family, and runtime checks."""

    blocked = [str(item) for item in runtime_receipt.get("blocked_preconditions", [])]
    model_ids = {str(row.get("hf_id")) for row in model_specs}
    if upstream.get("csl_claim_allowed") is not True:
        blocked.append("upstream_exp5558_csl_claim_not_allowed")
    if upstream.get("csl_memory_ready") is not True:
        blocked.append("upstream_exp5558_csl_memory_not_ready")
    if not fixture.get("decisions") or not fixture.get("active_entries"):
        blocked.append("upstream_causal_fixture_unavailable")
    if model_ids != set(MANDATED_HF_IDS):
        blocked.append("mandated_model_specs_missing")
    if not roles.get("source") or not roles.get("target"):
        blocked.append("required_cross_family_roles_unavailable")
    elif roles["source"].get("family") == roles["target"].get("family"):
        blocked.append("source_target_family_not_distinct")
    if runtime_receipt.get("cuda_visible") is not True:
        blocked.append("cuda_not_visible")
    if runtime_receipt.get("offload_evidence") is not True:
        blocked.append("gpu_offload_evidence_missing")
    if "llama" not in str(runtime_receipt.get("runtime_backend", "")):
        blocked.append("llama_cpp_gguf_runtime_missing")
    return {
        "upstream_causal_claim_allowed": upstream.get("csl_claim_allowed") is True,
        "upstream_causal_memory_ready": upstream.get("csl_memory_ready") is True,
        "causal_fixture_available": bool(
            fixture.get("decisions") and fixture.get("active_entries")
        ),
        "mandated_model_specs_present": model_ids == set(MANDATED_HF_IDS),
        "cross_family_roles_available": bool(roles.get("source") and roles.get("target")),
        "cuda_visible": runtime_receipt.get("cuda_visible") is True,
        "gpu_offload_preflight": runtime_receipt.get("offload_evidence") is True,
        "blocked_preconditions": sorted(set(blocked)),
        "all_passed": not blocked,
    }


def run_source_attempts(
    *,
    fixture: Mapping[str, Any],
    source_model: Mapping[str, Any],
    generation_runner: GenerationRunner,
    random_seed: int,
) -> list[JsonDict]:
    """Ask the source family to emit external causal-memory action tokens."""

    attempts: list[JsonDict] = []
    for index, candidate in enumerate(candidate_memory_rows(fixture)):
        seed = random_seed + index
        prompt = source_prompt(candidate, source_model)
        generation = generation_runner(
            stage="source_attempt",
            prompt=prompt,
            candidate_memory=candidate,
            model_spec=source_model,
            seed=seed,
            max_tokens=MAX_TOKENS,
            n_gpu_layers=N_GPU_LAYERS,
        )
        attempts.append(
            score_source_attempt(
                candidate=candidate,
                source_model=source_model,
                generation=generation,
                prompt=prompt,
                seed=seed,
            )
        )
    return attempts


def candidate_memory_rows(fixture: Mapping[str, Any]) -> list[JsonDict]:
    """Return current and forgotten Exp5558 rows eligible for source replay."""

    rows: list[JsonDict] = []
    for entry in fixture.get("active_entries", []):
        item = dict(entry)
        item["candidate_status"] = "current"
        rows.append(item)
    for entry in fixture.get("forgotten_entries", []):
        item = dict(entry)
        item["candidate_status"] = str(entry.get("forget_reason", "stale"))
        rows.append(item)
    return sorted(rows, key=lambda row: (str(row["context_key"]), str(row["memory_id"])))


def source_prompt(candidate: Mapping[str, Any], source_model: Mapping[str, Any]) -> str:
    """Build a source prompt whose exact output can become external memory."""

    return "\n".join(
        [
            "Return exactly one causal memory action token and no explanation.",
            f"Source model family: {source_model['family']}",
            f"Memory id: {candidate['memory_id']}",
            f"Context key: {candidate['context_key']}",
            f"Memory status: {candidate['candidate_status']}",
            f"Verified causal action token: {candidate['selected_action']}",
            "Final answer:",
        ]
    )


def score_source_attempt(
    *,
    candidate: Mapping[str, Any],
    source_model: Mapping[str, Any],
    generation: Mapping[str, Any],
    prompt: str,
    seed: int,
) -> JsonDict:
    """Verifier-gate one source output before it can enter memory."""

    output_text = str(generation.get("output_text", ""))
    selected = extract_answer(output_text, [candidate["selected_action"], "unknown"])
    row: JsonDict = {
        "schema": "carnot.experiment_5559.source_attempt.v1",
        "stage": "source_attempt",
        "source_model": str(source_model["hf_id"]),
        "source_family": str(source_model["family"]),
        "memory_id": str(candidate["memory_id"]),
        "event_id": str(candidate.get("event_id", "")),
        "context_key": str(candidate["context_key"]),
        "candidate_status": str(candidate["candidate_status"]),
        "expected_memory_action": str(candidate["selected_action"]),
        "selected_action": selected,
        "verifier_accepted": selected == candidate["selected_action"],
        "external_memory_only": True,
        "random_seed": seed,
        "prompt_text": prompt,
        "prompt_hash": sha256_text(prompt),
        "output_text": output_text,
        "output_hash": sha256_text(output_text),
        "backend_details": deepcopy(generation.get("backend_details", {})),
    }
    row["row_checksum"] = row_checksum(row)
    return row


def build_memory_entries(
    fixture: Mapping[str, Any],
    source_attempts: Sequence[Mapping[str, Any]],
) -> JsonDict:
    """Build aligned, shuffled, and stale target-memory views."""

    accepted = [row for row in source_attempts if row.get("verifier_accepted") is True]
    aligned = [
        memory_entry_from_source(row, "current")
        for row in accepted
        if row.get("candidate_status") == "current"
    ]
    stale = [
        memory_entry_from_source(row, str(row.get("candidate_status", "stale")))
        for row in accepted
        if row.get("candidate_status") != "current"
    ]
    return {
        "aligned": sorted(aligned, key=lambda row: row["memory_id"]),
        "shuffled": shuffled_memory_entries(aligned, fixture.get("decisions", [])),
        "stale": sorted(stale, key=lambda row: row["memory_id"]),
    }


def memory_entry_from_source(source_row: Mapping[str, Any], status: str) -> JsonDict:
    """Copy verifier-accepted source output into external memory shape."""

    return {
        "memory_id": str(source_row["memory_id"]),
        "source_memory_id": str(source_row["memory_id"]),
        "context_key": str(source_row["context_key"]),
        "source_context_key": str(source_row["context_key"]),
        "selected_action": str(source_row["selected_action"]),
        "memory_status": status,
        "source_model": str(source_row["source_model"]),
        "source_family": str(source_row["source_family"]),
        "source_prompt_hash": str(source_row["prompt_hash"]),
        "source_output_hash": str(source_row["output_hash"]),
        "aligned_to_decision": status == "current",
        "external_memory_only": True,
    }


def shuffled_memory_entries(
    aligned_entries: Sequence[Mapping[str, Any]],
    decisions: Sequence[Mapping[str, Any]],
) -> list[JsonDict]:
    """Rotate current memory actions while preserving target decision contexts."""

    entries = sorted([dict(entry) for entry in aligned_entries], key=lambda row: row["context_key"])
    if not entries:
        return []
    rotated = entries[1:] + entries[:1]
    decision_contexts = [str(decision["context_key"]) for decision in decisions]
    shuffled: list[JsonDict] = []
    for context_key in decision_contexts:
        replacement = next(
            (entry for entry in rotated if entry["source_context_key"] != context_key),
            rotated[0],
        )
        item = deepcopy(replacement)
        item["memory_id"] = f"shuffled-{context_key}"
        item["context_key"] = context_key
        item["source_memory_id"] = str(replacement["memory_id"])
        item["memory_status"] = "shuffled"
        item["aligned_to_decision"] = False
        shuffled.append(item)
    return shuffled


def run_target_evaluations(
    *,
    fixture: Mapping[str, Any],
    target_model: Mapping[str, Any],
    memory_entries: Mapping[str, Sequence[Mapping[str, Any]]],
    generation_runner: GenerationRunner,
    random_seed: int,
) -> dict[str, list[JsonDict]]:
    """Evaluate one target family across all target memory arms."""

    results = {arm: [] for arm in TARGET_ARMS}
    for decision_index, decision in enumerate(fixture["decisions"]):
        for arm_index, arm in enumerate(TARGET_ARMS):
            seed = random_seed + 1000 + decision_index * 37 + arm_index
            memory = memory_for_arm(memory_entries, arm, str(decision["context_key"]))
            prompt = target_prompt(decision, target_model, arm, memory)
            generation = generation_runner(
                stage="target_evaluation",
                arm=arm,
                prompt=prompt,
                decision=decision,
                model_spec=target_model,
                memory_entry=memory,
                seed=seed,
                max_tokens=MAX_TOKENS,
                n_gpu_layers=N_GPU_LAYERS,
            )
            results[arm].append(
                score_target_row(
                    decision=decision,
                    target_model=target_model,
                    arm=arm,
                    memory=memory,
                    generation=generation,
                    prompt=prompt,
                    seed=seed,
                )
            )
    return results


def memory_for_arm(
    memory_entries: Mapping[str, Sequence[Mapping[str, Any]]],
    arm: str,
    context_key: str,
) -> Mapping[str, Any] | None:
    """Return the external memory row visible to one target arm."""

    key = {
        ALIGNED_CAUSAL_MEMORY_ARM: "aligned",
        SHUFFLED_MEMORY_ARM: "shuffled",
        STALE_MEMORY_ARM: "stale",
    }.get(arm)
    if key is None:
        return None
    for entry in memory_entries.get(key, []):
        if entry.get("context_key") == context_key:
            return entry
    return None


def target_prompt(
    decision: Mapping[str, Any],
    target_model: Mapping[str, Any],
    arm: str,
    memory: Mapping[str, Any] | None,
) -> str:
    """Build the target prompt with explicit causal-memory validity checks."""

    allowed = allowed_action_tokens(decision, memory)
    lines = [
        "Return exactly one allowed action token and no explanation.",
        f"Target model family: {target_model['family']}",
        f"Arm: {arm}",
        f"Decision id: {decision['decision_id']}",
        f"Decision context key: {decision['context_key']}",
        f"Baseline action: {decision['baseline_action']}",
        f"Allowed action tokens: {' | '.join(allowed)}",
    ]
    if memory is None:
        lines.extend(
            [
                "Retrieved causal memory: none.",
                "Instruction: no current causal memory is available, so return the baseline action.",
            ]
        )
    else:
        lines.extend(
            [
                f"retrieved_memory_id: {memory['memory_id']}",
                f"retrieved_source_context_key: {memory['source_context_key']}",
                f"retrieved_memory_status: {memory['memory_status']}",
                f"verified_memory_action: {memory['selected_action']}",
                (
                    "Instruction: copy verified_memory_action only when "
                    "retrieved_memory_status is current and "
                    "retrieved_source_context_key equals Decision context key; "
                    "otherwise return unknown."
                ),
            ]
        )
    lines.append("Final answer:")
    return "\n".join(lines)


def allowed_action_tokens(
    decision: Mapping[str, Any],
    memory: Mapping[str, Any] | None,
) -> list[str]:
    """Return deterministic literal tokens accepted by the target scorer."""

    tokens = [
        str(decision["expected_action"]),
        str(decision["baseline_action"]),
        "unknown",
    ]
    if memory is not None:
        tokens.append(str(memory["selected_action"]))
    deduped: list[str] = []
    for token in tokens:
        if token not in deduped:
            deduped.append(token)
    return deduped


def score_target_row(
    *,
    decision: Mapping[str, Any],
    target_model: Mapping[str, Any],
    arm: str,
    memory: Mapping[str, Any] | None,
    generation: Mapping[str, Any],
    prompt: str,
    seed: int,
) -> JsonDict:
    """Score one target output against Exp5558 independent decision labels."""

    output_text = str(generation.get("output_text", ""))
    selected = extract_answer(output_text, allowed_action_tokens(decision, memory))
    memory_status = str(memory.get("memory_status")) if memory else None
    stale_opportunity = memory_status in {"stale", "contradicted"}
    row: JsonDict = {
        "schema": "carnot.experiment_5559.target_row.v1",
        "stage": "target_evaluation",
        "arm": arm,
        "decision_id": str(decision["decision_id"]),
        "context_key": str(decision["context_key"]),
        "target_model": str(target_model["hf_id"]),
        "target_family": str(target_model["family"]),
        "random_seed": seed,
        "prompt_text": prompt,
        "prompt_hash": sha256_text(prompt),
        "output_text": output_text,
        "output_hash": sha256_text(output_text),
        "selected_action": selected,
        "expected_action": str(decision["expected_action"]),
        "baseline_action": str(decision["baseline_action"]),
        "accepted_by_independent_label": selected == decision["expected_action"],
        "memory_receipt": deepcopy(dict(memory)) if memory else None,
        "memory_context_aligned": bool(
            memory
            and memory.get("memory_status") == "current"
            and memory.get("source_context_key") == decision["context_key"]
        ),
        "stale_evidence_opportunity": stale_opportunity,
        "stale_evidence_rejected": (
            selected != memory.get("selected_action") if memory and stale_opportunity else None
        ),
        "negative_transfer_detected": (
            arm == ALIGNED_CAUSAL_MEMORY_ARM and selected != decision["expected_action"]
        ),
        "backend_details": deepcopy(generation.get("backend_details", {})),
    }
    row["row_checksum"] = row_checksum(row)
    return row


def build_artifact(
    *,
    upstream: Mapping[str, Any],
    fixture: Mapping[str, Any],
    model_specs: Sequence[Mapping[str, Any]],
    cache_receipt: Mapping[str, Any],
    roles: Mapping[str, Any],
    runtime_receipt: Mapping[str, Any],
    preconditions: Mapping[str, Any],
    source_attempts: Sequence[Mapping[str, Any]],
    memory_entries: Mapping[str, Sequence[Mapping[str, Any]]],
    target_evaluations: Mapping[str, Sequence[Mapping[str, Any]]],
    row_results: Sequence[Mapping[str, Any]],
    before_receipts: Mapping[str, Mapping[str, Any]],
    after_receipts: Mapping[str, Mapping[str, Any]],
    live_model_invoked: bool,
    tests_added_or_reused: Sequence[str],
    measured_duration_s: float,
    random_seed: int,
) -> JsonDict:
    """Assemble the Exp5559 receipt with recomputable gates."""

    scores = arm_scores(target_evaluations)
    cross_delta = round(
        scores[ALIGNED_CAUSAL_MEMORY_ARM] - scores[SHUFFLED_MEMORY_ARM],
        10,
    )
    stale_rate = stale_evidence_rejection_rate(
        target_evaluations.get(STALE_MEMORY_ARM, [])
    )
    negative_rate = negative_transfer_rate(
        target_evaluations.get(ALIGNED_CAUSAL_MEMORY_ARM, [])
    )
    weight_evidence = exp5544.weight_mutation_evidence(before_receipts, after_receipts)
    offload_evidence = gpu_offload_evidence(runtime_receipt, preconditions)
    same_decisions = same_decision_set(target_evaluations)
    source_models = [str(roles["source"]["hf_id"])] if isinstance(roles.get("source"), Mapping) else []
    target_models = [str(roles["target"]["hf_id"])] if isinstance(roles.get("target"), Mapping) else []
    unavailable = [
        str(row["hf_id"])
        for row in model_specs
        if row.get("local_path_available") is not True
    ]
    claim_allowed = expected_claim_from_parts(
        upstream=upstream,
        preconditions=preconditions,
        scores=scores,
        same_decisions=same_decisions,
        stale_rate=stale_rate,
        negative_rate=negative_rate,
        no_weight_mutation=weight_evidence["no_weight_mutation"],
        offload_evidence=offload_evidence,
        live_model_invoked=live_model_invoked,
        source_models=source_models,
        target_models=target_models,
        source_attempts=source_attempts,
        row_results=row_results,
    )
    artifact: JsonDict = {
        "experiment": 5559,
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "task_id": TASK_ID,
        "milestone": MILESTONE,
        "run_date": RUN_DATE,
        "random_seed": random_seed,
        "spec_refs": list(SPEC_REFS),
        "upstream_causal_memory": UPSTREAM_CAUSAL_MEMORY_PATH.as_posix(),
        "upstream_causal_status": public_upstream_status(upstream),
        "fixture": deepcopy(dict(fixture)),
        "fixture_hash": fixture.get("fixture_hash"),
        "model_specs": [dict(row) for row in model_specs],
        "model_cache_evidence": deepcopy(dict(cache_receipt)),
        "unavailable_models": unavailable,
        "precondition_details": deepcopy(dict(preconditions)),
        "source_models": source_models,
        "target_models": target_models,
        "source_attempts": [dict(row) for row in source_attempts],
        "memory_entries": deepcopy(dict(memory_entries)),
        "target_evaluations": {
            arm: [dict(row) for row in target_evaluations.get(arm, [])]
            for arm in TARGET_ARMS
        },
        "row_results": [dict(row) for row in row_results],
        "same_decision_set": same_decisions,
        "no_memory_score": scores[NO_MEMORY_ARM],
        "shuffled_memory_score": scores[SHUFFLED_MEMORY_ARM],
        "stale_memory_score": scores[STALE_MEMORY_ARM],
        "aligned_memory_score": scores[ALIGNED_CAUSAL_MEMORY_ARM],
        "cross_family_delta_over_shuffled": cross_delta,
        "negative_transfer_rate": negative_rate,
        "stale_evidence_rejection_rate": stale_rate,
        "weight_mutation_evidence": weight_evidence,
        "no_weight_mutation": weight_evidence["no_weight_mutation"],
        "measured_duration_s": round(float(measured_duration_s), 10),
        "gpu_offload_evidence": offload_evidence,
        "live_model_invoked": live_model_invoked,
        "no_weight_mutation_method": "before_after_gguf_file_receipts_no_training",
        "no_weight_mutation_scope": "external_causal_memory_only",
        "csl_claim_allowed": claim_allowed,
        "tests_added_or_reused": list(tests_added_or_reused),
        "field_principles": dict(FIELD_PRINCIPLES),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "source_files": {
            "module": MODULE_RELATIVE_PATH.as_posix(),
            "spec": SPEC_RELATIVE_PATH.as_posix(),
            "test": TEST_RELATIVE_PATH.as_posix(),
        },
        "source_file_checksums": source_file_checksums(REPO_ROOT),
        "methodology_note": (
            "Exp5558 causal memory is reused as external memory only. Source "
            "and target GGUF model files remain frozen; no adapter or training "
            "step is run."
        ),
    }
    artifact["honest_verdict"] = honest_verdict(artifact)
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    return artifact


def expected_claim_from_parts(
    *,
    upstream: Mapping[str, Any],
    preconditions: Mapping[str, Any],
    scores: Mapping[str, float],
    same_decisions: bool,
    stale_rate: float,
    negative_rate: float,
    no_weight_mutation: bool,
    offload_evidence: Mapping[str, Any],
    live_model_invoked: bool,
    source_models: Sequence[str],
    target_models: Sequence[str],
    source_attempts: Sequence[Mapping[str, Any]],
    row_results: Sequence[Mapping[str, Any]],
) -> bool:
    """Return the final claim gate from primitive values."""

    return bool(
        upstream.get("csl_claim_allowed") is True
        and upstream.get("csl_memory_ready") is True
        and preconditions.get("all_passed") is True
        and source_models
        and target_models
        and set(source_models).isdisjoint(target_models)
        and live_model_invoked
        and source_attempts
        and row_results
        and same_decisions
        and scores[ALIGNED_CAUSAL_MEMORY_ARM] > scores[SHUFFLED_MEMORY_ARM]
        and stale_rate == 1.0
        and negative_rate == 0.0
        and no_weight_mutation
        and clean_gpu_offload(offload_evidence)
    )


def validate_artifact(artifact: Mapping[str, Any]) -> bool:
    """Raise when Exp5559 evidence or gate logic is inconsistent."""

    errors = artifact_errors(artifact)
    if errors:
        raise ValueError("invalid Exp5559 artifact: " + "; ".join(errors))
    return True


def artifact_errors(artifact: Mapping[str, Any]) -> list[str]:
    """Return validation errors while allowing honest blocked gate skips."""

    errors: list[str] = []
    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    if missing:
        errors.append(f"missing required fields: {missing}")
    if artifact.get("upstream_causal_memory") != UPSTREAM_CAUSAL_MEMORY_PATH.as_posix():
        errors.append("upstream_causal_memory")
    if artifact.get("random_seed") != RANDOM_SEED:
        errors.append("random_seed")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate")

    target_evaluations = artifact.get("target_evaluations", {})
    scores = arm_scores(target_evaluations)
    for arm, field in ARM_SCORE_FIELDS.items():
        if float(artifact.get(field, -1.0)) != scores[arm]:
            errors.append(field)
    if float(artifact.get("cross_family_delta_over_shuffled", -1.0)) != round(
        scores[ALIGNED_CAUSAL_MEMORY_ARM] - scores[SHUFFLED_MEMORY_ARM],
        10,
    ):
        errors.append("cross_family_delta_over_shuffled")
    if float(artifact.get("stale_evidence_rejection_rate", -1.0)) != stale_evidence_rejection_rate(
        target_evaluations.get(STALE_MEMORY_ARM, []) if isinstance(target_evaluations, Mapping) else []
    ):
        errors.append("stale_evidence_rejection_rate")
    if float(artifact.get("negative_transfer_rate", -1.0)) != negative_transfer_rate(
        target_evaluations.get(ALIGNED_CAUSAL_MEMORY_ARM, []) if isinstance(target_evaluations, Mapping) else []
    ):
        errors.append("negative_transfer_rate")

    same_decisions = same_decision_set(target_evaluations)
    if artifact.get("same_decision_set") is not same_decisions:
        errors.append("same_decision_set")
    if artifact.get("live_model_invoked") is not bool(
        artifact.get("source_attempts") or artifact.get("row_results")
    ):
        errors.append("live_model_invoked")
    if artifact.get("no_weight_mutation") is not recompute_no_weight_mutation(artifact):
        errors.append("no_weight_mutation")
    if not exp5544.weight_evidence_consistent(artifact.get("weight_mutation_evidence", {})):
        errors.append("weight_mutation_evidence")
    gpu_evidence = artifact.get("gpu_offload_evidence", {})
    if not gpu_evidence_consistent(gpu_evidence):
        errors.append("gpu_offload_evidence")
    if artifact.get("csl_claim_allowed") is True and not clean_gpu_offload(gpu_evidence):
        errors.append("gpu_offload_evidence")

    expected_claim = expected_claim_from_artifact(artifact, scores, same_decisions)
    if artifact.get("csl_claim_allowed") is not expected_claim:
        errors.append("csl_claim_allowed")
    if artifact.get("honest_verdict") != honest_verdict(artifact):
        errors.append("honest_verdict")

    principles = artifact.get("field_principles", {})
    missing_principles = [
        field
        for field in REQUIRED_ARTIFACT_FIELDS
        if not isinstance(principles, Mapping) or not principles.get(field)
    ]
    if missing_principles:
        errors.append(f"field_principles missing: {missing_principles}")

    model_ids = {str(row.get("hf_id")) for row in artifact.get("model_specs", [])}
    if model_ids != set(MANDATED_HF_IDS):
        errors.append("model_specs")
    for row in list(artifact.get("source_attempts", [])) + list(artifact.get("row_results", [])):
        if isinstance(row, Mapping) and row.get("row_checksum") != row_checksum(row):
            errors.append("row_checksum")
            break
    checksum = artifact.get("reproducibility_checksum")
    if checksum and checksum != reproducibility_checksum(artifact):
        errors.append("reproducibility_checksum")
    return errors


def expected_claim_from_artifact(
    artifact: Mapping[str, Any],
    scores: Mapping[str, float],
    same_decisions: bool,
) -> bool:
    """Recompute the final claim gate from stored artifact fields."""

    return expected_claim_from_parts(
        upstream=artifact.get("upstream_causal_status", {}),
        preconditions=artifact.get("precondition_details", {}),
        scores=scores,
        same_decisions=same_decisions,
        stale_rate=float(artifact.get("stale_evidence_rejection_rate", 0.0)),
        negative_rate=float(artifact.get("negative_transfer_rate", 1.0)),
        no_weight_mutation=artifact.get("no_weight_mutation") is True,
        offload_evidence=artifact.get("gpu_offload_evidence", {}),
        live_model_invoked=artifact.get("live_model_invoked") is True,
        source_models=list(artifact.get("source_models", [])),
        target_models=list(artifact.get("target_models", [])),
        source_attempts=list(artifact.get("source_attempts", [])),
        row_results=list(artifact.get("row_results", [])),
    )


def honest_verdict(artifact: Mapping[str, Any]) -> str:
    """Return the terminal verdict required by conductor receipts."""

    if artifact.get("csl_claim_allowed") is True:
        return "complete: causal_cross_model_sota_csl_transfer_v2_claim_allowed"
    return "blocked: causal_cross_model_sota_csl_transfer_v2_claim_not_allowed"


def selected_role_specs(roles: Mapping[str, Any]) -> list[JsonDict]:
    """Return unique source and target specs for before/after receipts."""

    specs: list[JsonDict] = []
    seen: set[str] = set()
    for key in ("source", "target"):
        row = roles.get(key)
        if isinstance(row, Mapping) and str(row.get("hf_id")) not in seen:
            specs.append(dict(row))
            seen.add(str(row.get("hf_id")))
    return specs


def arm_scores(target_evaluations: Any) -> JsonDict:
    """Recompute target arm scores from row evidence."""

    scores: JsonDict = {}
    for arm in TARGET_ARMS:
        rows = target_evaluations.get(arm, []) if isinstance(target_evaluations, Mapping) else []
        scores[arm] = score_rows(rows)
    return scores


def score_rows(rows: Sequence[Mapping[str, Any]]) -> float:
    """Return a rounded pass-rate score, with empty rows scored as zero."""

    if not rows:
        return 0.0
    return round(
        sum(row.get("accepted_by_independent_label") is True for row in rows) / len(rows),
        10,
    )


def stale_evidence_rejection_rate(rows: Sequence[Mapping[str, Any]]) -> float:
    """Measure stale-memory opportunities where the stale action was not used."""

    opportunities = [row for row in rows if row.get("stale_evidence_opportunity") is True]
    return safe_rate(
        sum(row.get("stale_evidence_rejected") is True for row in opportunities),
        len(opportunities),
    )


def negative_transfer_rate(rows: Sequence[Mapping[str, Any]]) -> float:
    """Measure aligned causal rows where transfer made the target miss the label."""

    return safe_rate(
        sum(row.get("negative_transfer_detected") is True for row in rows),
        len(rows),
    )


def same_decision_set(target_evaluations: Any) -> bool:
    """Check that every target arm used the same ordered decision IDs."""

    if not isinstance(target_evaluations, Mapping):
        return False
    decision_sets = [
        [row.get("decision_id") for row in target_evaluations.get(arm, [])]
        for arm in TARGET_ARMS
    ]
    return bool(decision_sets and decision_sets[0]) and all(
        decision_ids == decision_sets[0] for decision_ids in decision_sets
    )


def safe_rate(numerator: int, denominator: int) -> float:
    """Return a deterministic rate and avoid divide-by-zero in gate skips."""

    return 0.0 if denominator == 0 else round(numerator / denominator, 10)


def extract_answer(output_text: str, candidates: Sequence[str]) -> str | None:
    """Return the first literal candidate token found in model output."""

    best: tuple[int, int, str] | None = None
    for index, candidate in enumerate(candidates):
        match = re.search(
            rf"(?<![A-Za-z0-9_-]){re.escape(str(candidate).lower())}(?![A-Za-z0-9_-])",
            output_text.lower(),
        )
        if match is not None:
            current = (match.start(), index, str(candidate))
            best = current if best is None or current < best else best
    return best[2] if best else None


def gpu_offload_evidence(
    runtime_receipt: Mapping[str, Any],
    preconditions: Mapping[str, Any],
) -> JsonDict:
    """Normalize runtime and offload evidence without making blocked skips invalid."""

    return {
        "runtime_backend": str(runtime_receipt.get("runtime_backend", "unavailable")),
        "cuda_visible": runtime_receipt.get("cuda_visible") is True,
        "llama_cpp_import_ok": runtime_receipt.get("llama_cpp_import_ok") is True,
        "gpu_offload_supported": runtime_receipt.get("gpu_offload_supported") is True,
        "offload_evidence": runtime_receipt.get("offload_evidence") is True,
        "n_gpu_layers": int(runtime_receipt.get("n_gpu_layers", N_GPU_LAYERS)),
        "gpu_memory_delta_mb": int(runtime_receipt.get("gpu_memory_delta_mb", 0) or 0),
        "load_receipts": deepcopy(runtime_receipt.get("load_receipts", [])),
        "blocked_preconditions": list(preconditions.get("blocked_preconditions", [])),
    }


def gpu_evidence_consistent(evidence: Any) -> bool:
    """Check the GPU evidence shape; clean offload is required only for claims."""

    return (
        isinstance(evidence, Mapping)
        and isinstance(evidence.get("runtime_backend"), str)
        and isinstance(evidence.get("cuda_visible"), bool)
        and isinstance(evidence.get("offload_evidence"), bool)
        and isinstance(evidence.get("blocked_preconditions"), list)
    )


def clean_gpu_offload(evidence: Any) -> bool:
    """Return true only for CUDA-visible llama.cpp offload evidence."""

    return (
        isinstance(evidence, Mapping)
        and evidence.get("offload_evidence") is True
        and evidence.get("cuda_visible") is True
        and evidence.get("llama_cpp_import_ok") is True
        and "llama" in str(evidence.get("runtime_backend", ""))
    )


def recompute_no_weight_mutation(artifact: Mapping[str, Any]) -> bool:
    """Recompute the bare frozen-weight gate from stored receipts."""

    evidence = artifact.get("weight_mutation_evidence", {})
    return isinstance(evidence, Mapping) and exp5544.weight_evidence_consistent(evidence)


def resolve_path(root: Path | str, path: Path | str) -> Path:
    """Resolve repository-relative paths while preserving absolute paths."""

    candidate = Path(path)
    return candidate if candidate.is_absolute() else Path(root) / candidate


def load_json(path: Path | str) -> JsonDict:
    """Read a JSON object from disk."""

    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: Path | str, payload: Mapping[str, Any]) -> None:
    """Write stable JSON so artifact diffs are meaningful."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def row_checksum(row: Mapping[str, Any]) -> str:
    """Hash one row with its checksum field removed."""

    payload = {key: value for key, value in row.items() if key != "row_checksum"}
    return "sha256:" + sha256_json(payload)


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    """Hash the artifact with its checksum field removed."""

    payload = {
        key: value for key, value in artifact.items() if key != "reproducibility_checksum"
    }
    return "sha256:" + sha256_json(payload)


def source_file_checksums(root: Path) -> JsonDict:
    """Record source-file hashes for replay and audit."""

    return {
        "module": file_sha256(root / MODULE_RELATIVE_PATH),
        "spec": file_sha256(root / SPEC_RELATIVE_PATH),
        "test": file_sha256(root / TEST_RELATIVE_PATH),
    }


def file_sha256(path: Path) -> str:
    """Hash a local file as sha256 text."""

    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def sha256_text(text: str) -> str:
    """Hash text with the same prefix used by artifacts."""

    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_json(payload: Any) -> str:
    """Hash a JSON-serializable payload deterministically."""

    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def main() -> None:  # pragma: no cover - thin CLI wrapper.
    """CLI entrypoint used by the conductor."""

    artifact = run()
    print(
        json.dumps(
            {
                "path": RESULT_RELATIVE_PATH.as_posix(),
                "csl_claim_allowed": artifact["csl_claim_allowed"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":  # pragma: no cover - thin CLI wrapper.
    main()
