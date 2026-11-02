"""Exp5513 bounded local SOTA GGUF hard/soft structured evidence panel.

Spec refs: REQ-VERIFY-5513, SCENARIO-VERIFY-5513.

This module is deliberately a panel wrapper, not a new verifier. Exp5512 owns
the structured row schema and parser handoff, while Exp5499 owns the exact
hard/soft validators. Exp5513 only gates local GGUF inference, prompts the
mandated headline models, preserves missing or malformed rows as evidence, and
computes aggregate rates from those exact validators.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import gc
import hashlib
import json
from pathlib import Path
import subprocess
import time
from typing import Any

from carnot import experiment_5499_preference_maxsat_minimal_fixture_v499 as fixture_mod
from carnot import experiment_5512_structured_output_positive_control as positive
from carnot.inference.sota_models import cached_sota_pair, resolve_cached_gguf


JsonDict = dict[str, Any]
CacheResolver = Callable[[str, str], str | None]
PanelRunner = Callable[[Mapping[str, Any], str], Mapping[str, Any] | str]
PairResolver = Callable[[], Sequence[Mapping[str, Any]] | None]

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path("results/experiment_5513_sota_hard_soft_structured_panel.json")
STRUCTURED_POSITIVE_CONTROL_ARTIFACT_RELATIVE_PATH = positive.RESULT_RELATIVE_PATH

SCHEMA = "carnot.experiment_5513.sota_hard_soft_structured_panel.v500"
EXPERIMENT = 5513
EXPERIMENT_ID = "exp5513-sota-hard-soft-structured-panel"
MILESTONE = "2026.07.500"
RUN_DATE = "2026-07-10"
RANDOM_SEED = 5513
PREFERRED_QUANT = "Q4_K_M"
N_GPU_LAYERS = -1
INFERENCE_SUBSTRATE = "live_llm_inference"
SPEC_REFS = ("REQ-VERIFY-5513", "SCENARIO-VERIFY-5513")

MANDATED_HEADLINE_MODEL_IDS = positive.MANDATED_HEADLINE_MODEL_IDS
MODEL_SPECS = positive.MODEL_SPECS

REQUIRED_ARTIFACT_FIELDS = (
    "model_specs",
    "headline_models_used",
    "legacy_smoke_models_used",
    "cached_models_missing",
    "llama_cpp_cuda_available",
    "gpu_offload_verified",
    "gpu_memory_delta_mb",
    "structured_positive_control_artifact",
    "exact_validator_accuracy",
    "hard_constraint_violation_rate",
    "preference_optimality_rate",
    "schema_validity_rate",
    "abstention_rate",
    "missing_candidate_rows",
    "sota_rows_emitted",
    "sota_structured_panel_ready",
    "inference_substrate",
    "honest_verdict",
)


def canonical_json(value: Any) -> str:
    """Serialize JSON deterministically so prompts and checksums are stable."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_json(value: Any) -> str:
    """Hash a JSON-compatible value after stable serialization."""

    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def payload_checksum(payload: Mapping[str, Any]) -> str:
    """Hash an artifact while blanking its self-referential checksum field."""

    stable = dict(payload)
    stable["reproducibility_checksum"] = ""
    return sha256_json(stable)


def load_structured_positive_control(path: Path) -> JsonDict:
    """Load the Exp5512 gate artifact and return the fields Exp5513 consumes."""

    if not path.exists():
        return {
            "structured_output_positive_control_ready": False,
            "sota_panel_gate_open": False,
            "honest_verdict": "blocked: structured_positive_control_artifact_missing",
        }
    return json.loads(path.read_text(encoding="utf-8"))


def build_reason_then_structure_prompt(target_rows: Sequence[Mapping[str, Any]]) -> str:
    """Build the bounded prompt: short reasoning first, exact schema rows last."""

    return (
        "You are evaluating hard/soft claim candidates. Write brief reasoning "
        "about hard constraints before soft preferences, then output one JSON "
        "object with candidate_rows matching this schema. Do not omit rows; "
        "abstain only when hard constraints are infeasible.\n\n"
        f"Row schema version: {positive.CANDIDATE_SCHEMA_VERSION}\n"
        f"JSON schema: {canonical_json(positive.candidate_schema())}\n"
        f"Candidate rows to transcribe and verify: {canonical_json(list(target_rows))}\n"
        "Final answer shape: {\"candidate_rows\": [...], \"proof_claims\": "
        "[{\"candidate_id\": \"...\", \"claimed_exact_validator_verdict\": \"...\"}]}\n"
    )


def extract_candidate_payloads(text: str) -> JsonDict:
    """Extract candidate row payloads from reason-then-JSON model output."""

    parsed = _first_json_payload(text)
    if parsed is None:
        return {
            "candidate_payloads": [],
            "proof_claims": [],
            "parse_failures": [{"parse_status": "no_json_payload", "detail": text[:200]}],
        }
    if isinstance(parsed, list):
        return {"candidate_payloads": [row for row in parsed if isinstance(row, Mapping)], "proof_claims": [], "parse_failures": []}
    if isinstance(parsed.get("candidate_rows"), list):
        rows = [row for row in parsed["candidate_rows"] if isinstance(row, Mapping)]
        claims = parsed.get("proof_claims", [])
        return {
            "candidate_payloads": rows,
            "proof_claims": [row for row in claims if isinstance(row, Mapping)],
            "parse_failures": [],
        }
    if parsed.get("candidate_schema_version") == positive.CANDIDATE_SCHEMA_VERSION:
        return {"candidate_payloads": [dict(parsed)], "proof_claims": [], "parse_failures": []}
    return {
        "candidate_payloads": [],
        "proof_claims": [],
        "parse_failures": [{"parse_status": "candidate_rows_missing"}],
    }


def resolve_model_specs(cache_resolver: CacheResolver = resolve_cached_gguf) -> list[JsonDict]:
    """Resolve mandated GGUF files through the existing local-GGUF helper path."""

    return positive.resolve_model_specs(cache_resolver)


def build_artifact(
    *,
    structured_positive_control_path: Path = REPO_ROOT / STRUCTURED_POSITIVE_CONTROL_ARTIFACT_RELATIVE_PATH,
    runtime_status: Mapping[str, Any] | None = None,
    cache_resolver: CacheResolver = resolve_cached_gguf,
    pair_resolver: PairResolver = cached_sota_pair,
    panel_runner: PanelRunner | None = None,
    max_headline_models: int = 1,
    tests_run: Sequence[Mapping[str, Any]] = (),
) -> JsonDict:
    """Build the Exp5513 artifact, running headline inference only after gates pass."""

    gate = load_structured_positive_control(Path(structured_positive_control_path))
    fixture = positive.load_fixture_artifact()["fixture"]
    target_rows = positive.build_fixture_candidate_payloads(fixture)
    model_specs = resolve_model_specs(cache_resolver)
    runtime = dict(runtime_status or probe_llama_runtime(model_specs))
    selected = _select_headline_specs(
        model_specs,
        pair_resolver=pair_resolver,
        max_headline_models=max_headline_models,
    )
    blockers = _preflight_blockers(gate=gate, model_specs=model_specs, runtime=runtime)
    model_runs: list[JsonDict] = []
    candidate_rows: list[JsonDict] = []
    if not blockers:
        prompt = build_reason_then_structure_prompt(target_rows)
        runner = panel_runner or default_panel_runner
        for spec in selected:
            model_run = _run_one_model(spec=spec, prompt=prompt, fixture=fixture, runner=runner)
            model_runs.append(model_run)
            candidate_rows.extend(model_run["candidate_rows"])

    metrics = _aggregate_metrics(
        model_runs=model_runs,
        fixture=fixture,
        runtime=runtime,
        preflight_blockers=blockers,
    )
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "run_date": RUN_DATE,
        "random_seed": RANDOM_SEED,
        "spec_refs": list(SPEC_REFS),
        "result_path": RESULT_RELATIVE_PATH.as_posix(),
        "model_specs": model_specs,
        "headline_models_used": metrics["headline_models_used"],
        "legacy_smoke_models_used": [],
        "cached_models_missing": [
            str(row["hf_id"]) for row in model_specs if row.get("local_model_present") is not True
        ],
        "llama_cpp_cuda_available": bool(runtime.get("llama_cpp_cuda_available")),
        "gpu_offload_verified": bool(runtime.get("gpu_offload_verified")),
        "gpu_memory_delta_mb": metrics["gpu_memory_delta_mb"],
        "structured_positive_control_artifact": (
            STRUCTURED_POSITIVE_CONTROL_ARTIFACT_RELATIVE_PATH.as_posix()
        ),
        "exact_validator_accuracy": metrics["exact_validator_accuracy"],
        "hard_constraint_violation_rate": metrics["hard_constraint_violation_rate"],
        "preference_optimality_rate": metrics["preference_optimality_rate"],
        "schema_validity_rate": metrics["schema_validity_rate"],
        "abstention_rate": metrics["abstention_rate"],
        "missing_candidate_rows": metrics["missing_candidate_rows"],
        "sota_rows_emitted": metrics["sota_rows_emitted"],
        "sota_structured_panel_ready": metrics["sota_structured_panel_ready"],
        "inference_substrate": INFERENCE_SUBSTRATE,
        "honest_verdict": honest_verdict(metrics["sota_structured_panel_ready"], metrics["readiness_blockers"]),
        "structured_positive_control_ready": bool(
            gate.get("structured_output_positive_control_ready")
        ),
        "sota_panel_gate_open": bool(gate.get("sota_panel_gate_open")),
        "structured_positive_control_honest_verdict": str(gate.get("honest_verdict", "")),
        "candidate_schema_version": positive.CANDIDATE_SCHEMA_VERSION,
        "fixture_artifact": positive.FIXTURE_ARTIFACT_RELATIVE_PATH.as_posix(),
        "fixture_sha256": fixture_mod.sha256_json(fixture),
        "model_runs": model_runs,
        "candidate_rows": candidate_rows,
        "parse_failure_counts": metrics["parse_failure_counts"],
        "proof_claim_consistency_rate": metrics["proof_claim_consistency_rate"],
        "readiness_blockers": metrics["readiness_blockers"],
        "runtime_status": runtime,
        "offload_diagnostics": list(runtime.get("offload_diagnostics", [])),
        "reason_then_structure_prompt_sha256": sha256_json(
            build_reason_then_structure_prompt(target_rows)
        ),
        "no_autotokenizer_on_gguf": True,
        "research_conductor_modified": False,
        "tests_run": [dict(row) for row in tests_run],
        "reproducibility_checksum": "",
    }
    artifact["reproducibility_checksum"] = payload_checksum(artifact)
    validate_artifact(artifact)
    return artifact


def run(
    *,
    result_path: Path | str = REPO_ROOT / RESULT_RELATIVE_PATH,
    structured_positive_control_path: Path = REPO_ROOT
    / STRUCTURED_POSITIVE_CONTROL_ARTIFACT_RELATIVE_PATH,
    runtime_status: Mapping[str, Any] | None = None,
    cache_resolver: CacheResolver = resolve_cached_gguf,
    pair_resolver: PairResolver = cached_sota_pair,
    panel_runner: PanelRunner | None = None,
    max_headline_models: int = 1,
    tests_run: Sequence[Mapping[str, Any]] = (),
    write: bool = True,
) -> JsonDict:
    """Build and optionally write the Exp5513 deliverable JSON."""

    artifact = build_artifact(
        structured_positive_control_path=structured_positive_control_path,
        runtime_status=runtime_status,
        cache_resolver=cache_resolver,
        pair_resolver=pair_resolver,
        panel_runner=panel_runner,
        max_headline_models=max_headline_models,
        tests_run=tests_run,
    )
    if write:
        output = Path(result_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(artifact, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
            encoding="utf-8",
        )
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Validate the terminal artifact and fail closed on overclaiming."""

    for field in REQUIRED_ARTIFACT_FIELDS:
        _require(field in artifact, field)
    _require(
        artifact.get("structured_positive_control_artifact")
        == STRUCTURED_POSITIVE_CONTROL_ARTIFACT_RELATIVE_PATH.as_posix(),
        "structured_positive_control_artifact",
    )
    _require(artifact.get("inference_substrate") == INFERENCE_SUBSTRATE, "inference_substrate")
    _require(artifact.get("no_autotokenizer_on_gguf") is True, "no_autotokenizer_on_gguf")
    _require(artifact.get("research_conductor_modified") is False, "research_conductor_modified")
    _require(str(artifact.get("honest_verdict", "")).startswith(("complete:", "blocked:")), "honest_verdict")
    _require(
        [row.get("hf_id") for row in artifact.get("model_specs", [])]
        == list(MANDATED_HEADLINE_MODEL_IDS),
        "model_specs",
    )
    _require(
        set(artifact.get("headline_models_used", [])).issubset(MANDATED_HEADLINE_MODEL_IDS),
        "headline_models_used",
    )
    _require(artifact.get("legacy_smoke_models_used") == [], "legacy_smoke_models_used")
    _require(
        set(artifact.get("cached_models_missing", [])).issubset(MANDATED_HEADLINE_MODEL_IDS),
        "cached_models_missing",
    )
    for field in (
        "exact_validator_accuracy",
        "hard_constraint_violation_rate",
        "preference_optimality_rate",
        "schema_validity_rate",
        "abstention_rate",
        "proof_claim_consistency_rate",
    ):
        _require(0.0 <= float(artifact.get(field, -1.0)) <= 1.0, field)
    _require(isinstance(artifact.get("llama_cpp_cuda_available"), bool), "llama_cpp_cuda_available")
    _require(isinstance(artifact.get("gpu_offload_verified"), bool), "gpu_offload_verified")
    _require(float(artifact.get("gpu_memory_delta_mb", -1.0)) >= 0.0, "gpu_memory_delta_mb")
    _require(int(artifact.get("missing_candidate_rows", -1)) >= 0, "missing_candidate_rows")
    _require(int(artifact.get("sota_rows_emitted", -1)) >= 0, "sota_rows_emitted")
    _require(isinstance(artifact.get("sota_structured_panel_ready"), bool), "ready")
    if artifact.get("sota_structured_panel_ready") is True:
        _require(bool(artifact.get("headline_models_used")), "headline_models_used")
        _require(artifact.get("gpu_offload_verified") is True, "gpu_offload_verified")
        _require(artifact.get("exact_validator_accuracy") == 1.0, "exact_validator_accuracy")
        _require(artifact.get("schema_validity_rate") == 1.0, "schema_validity_rate")
        _require(artifact.get("missing_candidate_rows") == 0, "missing_candidate_rows")
        _require(artifact.get("readiness_blockers") == [], "readiness_blockers")
    _require(artifact.get("reproducibility_checksum") == payload_checksum(artifact), "checksum")


def honest_verdict(ready: bool, blockers: Sequence[str]) -> str:
    """Return a terminal verdict that keeps blocked evidence explicit."""

    if ready:
        return "complete: sota_hard_soft_structured_panel_ready_exact_validators_authoritative"
    suffix = "_".join(blockers) if blockers else "insufficient_visible_evidence"
    return f"blocked: sota_hard_soft_structured_panel_not_ready_{suffix}"


def probe_llama_runtime(model_specs: Sequence[Mapping[str, Any]]) -> JsonDict:  # pragma: no cover
    """Check llama.cpp CUDA and observed GPU offload before headline inference."""

    cuda_available = _llama_cpp_cuda_available()
    first_cached = next((row for row in model_specs if row.get("local_model_present") is True), None)
    if not cuda_available or first_cached is None:
        return {
            "llama_cpp_cuda_available": cuda_available,
            "gpu_offload_verified": False,
            "gpu_memory_delta_mb": 0.0,
            "offload_diagnostics": [
                {
                    "resource": "llama_cpp_gpu_offload",
                    "available": False,
                    "detail": "cuda unavailable or no cached model",
                }
            ],
        }
    return _verify_gpu_offload(str(first_cached["model_path"]))


def default_panel_runner(spec: Mapping[str, Any], prompt: str) -> JsonDict:  # pragma: no cover
    """Run one bounded llama.cpp completion against a cached mandated GGUF."""

    from llama_cpp import Llama  # noqa: PLC0415

    before = _gpu_memory_total_mb()
    start = time.monotonic()
    llm = Llama(
        model_path=str(spec["model_path"]),
        n_ctx=4096,
        n_batch=128,
        n_gpu_layers=N_GPU_LAYERS,
        seed=RANDOM_SEED,
        verbose=False,
    )
    try:
        prompt_tokens = len(llm.tokenize(prompt.encode("utf-8")))
        result = llm.create_chat_completion(
            messages=[
                {
                    "role": "system",
                    "content": "Return brief reasoning followed by the requested JSON object.",
                },
                {"role": "user", "content": prompt},
            ],
            max_tokens=2048,
            temperature=0.0,
            top_p=1.0,
            seed=RANDOM_SEED,
        )
        choices = result.get("choices", []) if isinstance(result, Mapping) else []
        message = choices[0].get("message", {}) if choices else {}
        raw_output = str(message.get("content", "")) if isinstance(message, Mapping) else ""
        completion_tokens = len(llm.tokenize(raw_output.encode("utf-8"))) if raw_output else 0
        after = _gpu_memory_total_mb()
        return {
            "raw_output": raw_output,
            "llama_cpp_binding": "llama_cpp.Llama.create_completion",
            "llama_cpp_command": None,
            "n_gpu_layers": N_GPU_LAYERS,
            "gpu_memory_before_mb": before,
            "gpu_memory_after_mb": after,
            "gpu_memory_delta_mb": max(0.0, after - before),
            "wall_time_s": round(time.monotonic() - start, 6),
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
        }
    finally:
        llm = None
        gc.collect()


def _run_one_model(
    *,
    spec: Mapping[str, Any],
    prompt: str,
    fixture: Mapping[str, Any],
    runner: PanelRunner,
) -> JsonDict:
    start = time.monotonic()
    try:
        runner_output = runner(spec, prompt)
        telemetry = _normalize_runner_output(runner_output)
        runtime_error = None
    except Exception as exc:  # noqa: BLE001
        telemetry = _normalize_runner_output("")
        runtime_error = f"{type(exc).__name__}: {exc}"
    parsed = extract_candidate_payloads(str(telemetry["raw_output"]))
    candidate_rows = [
        positive.classify_candidate_payload(payload, fixture=fixture)
        for payload in parsed["candidate_payloads"]
    ]
    _attach_proof_consistency(candidate_rows, parsed["proof_claims"])
    parse_failures = list(parsed["parse_failures"])
    if runtime_error is not None:
        parse_failures.append({"parse_status": "runtime_error", "detail": runtime_error})
    expected_ids = {str(row["instance_id"]) for row in fixture["instances"]}
    seen_ids = {
        str(row["instance_id"])
        for row in candidate_rows
        if row.get("instance_id") and row.get("parse_status") != "unknown_instance"
    }
    missing_ids = sorted(expected_ids - seen_ids)
    for row in candidate_rows:
        row["model_hf_id"] = str(spec["hf_id"])
    return {
        "model_hf_id": str(spec["hf_id"]),
        "model_file": str(spec.get("model_path")),
        "quant": str(spec.get("preferred_quant", PREFERRED_QUANT)),
        "llama_cpp_binding": telemetry["llama_cpp_binding"],
        "llama_cpp_command": telemetry["llama_cpp_command"],
        "n_gpu_layers": int(telemetry["n_gpu_layers"]),
        "gpu_memory_delta_mb": float(telemetry["gpu_memory_delta_mb"]),
        "wall_time_s": float(telemetry["wall_time_s"]) or round(time.monotonic() - start, 6),
        "prompt_tokens": int(telemetry["prompt_tokens"]),
        "completion_tokens": int(telemetry["completion_tokens"]),
        "raw_output_preview": str(telemetry["raw_output"])[:500],
        "parse_failures": parse_failures,
        "candidate_rows": candidate_rows,
        "missing_instance_ids": missing_ids,
        "runtime_error": runtime_error,
    }


def _aggregate_metrics(
    *,
    model_runs: Sequence[Mapping[str, Any]],
    fixture: Mapping[str, Any],
    runtime: Mapping[str, Any],
    preflight_blockers: Sequence[str],
) -> JsonDict:
    expected_instances = list(fixture["instances"])
    expected_slots = len(model_runs) * len(expected_instances)
    feasible_ids = {
        str(row["instance_id"]) for row in expected_instances if row["expected_status"] == "optimal"
    }
    expected_feasible_slots = len(model_runs) * len(feasible_ids)
    rows = [row for run in model_runs for row in run.get("candidate_rows", [])]
    parse_failures = [row for run in model_runs for row in run.get("parse_failures", [])]
    missing_rows = sum(len(run.get("missing_instance_ids", [])) for run in model_runs)
    schema_valid = sum(int(row.get("schema_valid") is True) for row in rows)
    exact_correct = sum(int(row.get("exact_validator_correct") is True) for row in rows)
    hard_violations = sum(
        int(row.get("exact_validator_verdict") == "hard_constraint_violation") for row in rows
    )
    soft_optimal = sum(int(row.get("soft_optimal") is True) for row in rows)
    abstentions = sum(int(row.get("parse_status") == "schema_valid_abstention") for row in rows)
    proof_consistent = [
        row for row in rows if row.get("proof_claim_consistent") is not None
    ]
    proof_rate = _rate(
        sum(int(row.get("proof_claim_consistent") is True) for row in proof_consistent),
        len(proof_consistent),
        empty=1.0,
    )
    parse_counts = _parse_failure_counts(rows, parse_failures)
    blockers = list(preflight_blockers)
    if model_runs and not rows:
        blockers.append("no_candidate_rows_emitted")
    if missing_rows:
        blockers.append("missing_candidate_rows")
    if schema_valid < expected_slots:
        blockers.append("schema_invalid_or_missing_rows")
    if exact_correct < expected_slots:
        blockers.append("exact_validator_mismatch")
    if hard_violations:
        blockers.append("hard_constraint_violation")
    if soft_optimal < expected_feasible_slots:
        blockers.append("preference_suboptimal_or_missing")
    if parse_counts:
        blockers.append("parse_failures")
    if proof_rate < 1.0:
        blockers.append("proof_claim_mismatch")
    headline_models_used = [str(run["model_hf_id"]) for run in model_runs if run.get("runtime_error") is None]
    ready = bool(model_runs) and not blockers
    max_run_delta = max([float(run.get("gpu_memory_delta_mb", 0.0)) for run in model_runs] or [0.0])
    gpu_delta = max(float(runtime.get("gpu_memory_delta_mb", 0.0)), max_run_delta)
    return {
        "headline_models_used": headline_models_used,
        "exact_validator_accuracy": _rate(exact_correct, expected_slots),
        "hard_constraint_violation_rate": _rate(hard_violations, expected_slots),
        "preference_optimality_rate": _rate(soft_optimal, expected_feasible_slots),
        "schema_validity_rate": _rate(schema_valid, expected_slots),
        "abstention_rate": _rate(abstentions, expected_slots),
        "missing_candidate_rows": missing_rows,
        "sota_rows_emitted": len(rows),
        "parse_failure_counts": parse_counts,
        "proof_claim_consistency_rate": proof_rate,
        "gpu_memory_delta_mb": round(gpu_delta, 6),
        "readiness_blockers": sorted(set(blockers)),
        "sota_structured_panel_ready": ready,
    }


def _select_headline_specs(
    model_specs: Sequence[Mapping[str, Any]],
    *,
    pair_resolver: PairResolver,
    max_headline_models: int,
) -> list[JsonDict]:
    selected: list[JsonDict] = []
    by_id = {str(row["hf_id"]): row for row in model_specs}
    pair = pair_resolver() or []
    for pair_row in pair:
        spec = by_id.get(str(pair_row.get("hf_id")))
        if spec and spec.get("local_model_present") is True:
            selected.append(dict(spec))
    selected_ids = {str(row["hf_id"]) for row in selected}
    for spec in model_specs:
        if spec.get("local_model_present") is True and str(spec["hf_id"]) not in selected_ids:
            selected.append(dict(spec))
            selected_ids.add(str(spec["hf_id"]))
    return selected[: max(0, max_headline_models)]


def _preflight_blockers(
    *,
    gate: Mapping[str, Any],
    model_specs: Sequence[Mapping[str, Any]],
    runtime: Mapping[str, Any],
) -> list[str]:
    blockers = []
    if gate.get("structured_output_positive_control_ready") is not True:
        blockers.append("structured_positive_control_not_ready")
    if not any(row.get("local_model_present") is True for row in model_specs):
        blockers.append("no_cached_mandated_gguf")
    if runtime.get("llama_cpp_cuda_available") is not True:
        blockers.append("llama_cpp_cuda_unavailable")
    if runtime.get("gpu_offload_verified") is not True:
        blockers.append("gpu_offload_unverified")
    return blockers


def _normalize_runner_output(output: Mapping[str, Any] | str) -> JsonDict:
    if isinstance(output, Mapping):
        return {
            "raw_output": str(output.get("raw_output", "")),
            "llama_cpp_binding": output.get("llama_cpp_binding"),
            "llama_cpp_command": output.get("llama_cpp_command"),
            "n_gpu_layers": int(output.get("n_gpu_layers", N_GPU_LAYERS)),
            "gpu_memory_delta_mb": float(output.get("gpu_memory_delta_mb", 0.0)),
            "wall_time_s": float(output.get("wall_time_s", 0.0)),
            "prompt_tokens": int(output.get("prompt_tokens", 0)),
            "completion_tokens": int(output.get("completion_tokens", 0)),
        }
    return {
        "raw_output": str(output),
        "llama_cpp_binding": "unknown",
        "llama_cpp_command": None,
        "n_gpu_layers": N_GPU_LAYERS,
        "gpu_memory_delta_mb": 0.0,
        "wall_time_s": 0.0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
    }


def _attach_proof_consistency(
    rows: Sequence[JsonDict],
    proof_claims: Sequence[Mapping[str, Any]],
) -> None:
    claims = {
        str(row.get("candidate_id")): str(row.get("claimed_exact_validator_verdict"))
        for row in proof_claims
        if row.get("candidate_id") and row.get("claimed_exact_validator_verdict")
    }
    for row in rows:
        claim = claims.get(str(row.get("candidate_id")))
        row["proof_claimed_exact_validator_verdict"] = claim
        row["proof_claim_consistent"] = (
            None if claim is None else claim == row.get("exact_validator_verdict")
        )


def _first_json_payload(text: str) -> Any:
    decoder = json.JSONDecoder()
    payloads = []
    starts = [idx for idx, char in enumerate(text) if char in "[{"]
    for idx in starts:
        try:
            payload, _end = decoder.raw_decode(text[idx:])
            payloads.append(payload)
            if _payload_has_candidate_rows(payload):
                return payload
            if (
                isinstance(payload, Mapping)
                and payload.get("candidate_schema_version")
                == positive.CANDIDATE_SCHEMA_VERSION
            ):
                return payload
        except json.JSONDecodeError:
            continue
    for payload in reversed(payloads):
        if isinstance(payload, Mapping):
            return payload
    return payloads[-1] if payloads else None


def _payload_has_candidate_rows(payload: Any) -> bool:
    if isinstance(payload, Mapping):
        return isinstance(payload.get("candidate_rows"), list)
    if isinstance(payload, list):
        return any(
            isinstance(row, Mapping)
            and row.get("candidate_schema_version") == positive.CANDIDATE_SCHEMA_VERSION
            for row in payload
        )
    return False


def _parse_failure_counts(
    rows: Sequence[Mapping[str, Any]],
    parse_failures: Sequence[Mapping[str, Any]],
) -> JsonDict:
    counts: dict[str, int] = {}
    for failure in parse_failures:
        status = str(failure.get("parse_status"))
        counts[status] = counts.get(status, 0) + 1
    for row in rows:
        if row.get("parseable") is True:
            continue
        status = str(row.get("parse_status"))
        counts[status] = counts.get(status, 0) + 1
    return counts


def _rate(numerator: int | float, denominator: int, *, empty: float = 0.0) -> float:
    return round(float(numerator) / float(denominator), 6) if denominator else empty


def _llama_cpp_cuda_available() -> bool:  # pragma: no cover
    try:
        from llama_cpp import llama_cpp  # noqa: PLC0415

        return bool(llama_cpp.llama_supports_gpu_offload())
    except Exception:
        return False


def _verify_gpu_offload(model_path: str) -> JsonDict:  # pragma: no cover
    from llama_cpp import Llama  # noqa: PLC0415

    before = _gpu_memory_total_mb()
    llm = Llama(
        model_path=model_path,
        n_ctx=512,
        n_batch=32,
        n_gpu_layers=N_GPU_LAYERS,
        seed=RANDOM_SEED,
        verbose=False,
    )
    try:
        after = _gpu_memory_total_mb()
        delta = max(0.0, after - before)
        verified = delta > 128.0
        return {
            "llama_cpp_cuda_available": True,
            "gpu_offload_verified": verified,
            "gpu_memory_delta_mb": round(delta, 6),
            "offload_diagnostics": [
                {
                    "resource": "llama_cpp_gpu_offload",
                    "available": verified,
                    "detail": f"nvidia-smi memory delta {delta:.3f} MB",
                    "model_path": model_path,
                    "n_gpu_layers": N_GPU_LAYERS,
                }
            ],
        }
    finally:
        llm = None
        gc.collect()


def _gpu_memory_total_mb() -> float:  # pragma: no cover
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )
    if result.returncode != 0:
        return 0.0
    values = [float(line.strip()) for line in result.stdout.splitlines() if line.strip()]
    return sum(values)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def main() -> int:  # pragma: no cover
    artifact = run()
    print(
        json.dumps(
            {
                "result": RESULT_RELATIVE_PATH.as_posix(),
                "honest_verdict": artifact["honest_verdict"],
                "headline_models_used": artifact["headline_models_used"],
                "sota_structured_panel_ready": artifact["sota_structured_panel_ready"],
            }
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
