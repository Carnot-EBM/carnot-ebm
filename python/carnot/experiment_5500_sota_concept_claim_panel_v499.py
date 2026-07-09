"""Exp5500 local SOTA GGUF concept/claim panel over the Exp5499 fixture.

Spec refs: REQ-VERIFY-5500, SCENARIO-VERIFY-5500.

The model is allowed to write a complete candidate state or an explanation, but
the Exp5499 exact hard/soft validators remain the final authority. This keeps
the LLM role narrow: propose claim states, then let deterministic validators
decide whether the proposal is feasible, optimal, or an abstention.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import gc
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
from typing import Any

from carnot import experiment_5499_preference_maxsat_minimal_fixture_v499 as fixture_mod
from carnot.inference.sota_models import SOTA_GGUF_MODELS, cached_sota_pair, resolve_cached_gguf


JsonDict = dict[str, Any]
CacheResolver = Callable[[str, str], str | None]
RuntimeProbe = Callable[[], Mapping[str, Any]]
PairResolver = Callable[[], Sequence[Mapping[str, Any]] | None]
RuntimeFactory = Callable[[Mapping[str, Any]], Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path("results/experiment_5500_sota_concept_claim_panel_v499.json")
FIXTURE_ARTIFACT_RELATIVE_PATH = fixture_mod.RESULT_RELATIVE_PATH

SCHEMA = "carnot.experiment_5500.sota_concept_claim_panel.v499"
EXPERIMENT = 5500
EXPERIMENT_ID = "exp5500-sota-concept-claim-panel-v499"
MILESTONE = "2026.07.499"
RUN_DATE = "2026-07-09"
RANDOM_SEED = 5500
N_GPU_LAYERS = -1
PREFERRED_QUANT = "Q4_K_M"
INFERENCE_SUBSTRATE = "live_llm_inference"
SPEC_REFS = ("REQ-VERIFY-5500", "SCENARIO-VERIFY-5500")

MANDATED_HEADLINE_MODEL_IDS = (
    "unsloth/Qwen3.6-35B-A3B-GGUF",
    "unsloth/gemma-4-31B-it-GGUF",
    "unsloth/gemma-4-26B-A4B-it-GGUF",
)

_REGISTRY_BY_ID = {row["hf_id"]: row for row in SOTA_GGUF_MODELS}
MODEL_SPECS: list[JsonDict] = [
    {
        "name": _REGISTRY_BY_ID[hf_id]["name"],
        "hf_id": hf_id,
        "role": _REGISTRY_BY_ID[hf_id]["role"],
        "preferred_quant": PREFERRED_QUANT,
        "headline_eligible": True,
    }
    for hf_id in MANDATED_HEADLINE_MODEL_IDS
]

REQUIRED_ARTIFACT_FIELDS = (
    "model_specs",
    "headline_models_used",
    "legacy_smoke_models_used",
    "cached_models_missing",
    "llama_cpp_cuda_available",
    "gpu_offload_verified",
    "gpu_memory_delta_mb",
    "fixture_artifact",
    "exact_validator_accuracy",
    "hard_constraint_violation_rate",
    "preference_optimality_rate",
    "concept_claim_telemetry_rows",
    "guided_decoding_used",
    "inference_substrate",
    "honest_verdict",
)


def canonical_json(payload: Mapping[str, Any]) -> str:
    """Serialize JSON deterministically for reproducible checksums."""

    return json.dumps(dict(payload), sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def payload_checksum(payload: Mapping[str, Any]) -> str:
    """Hash an artifact while blanking its self-referential checksum field."""

    stable = dict(payload)
    stable["reproducibility_checksum"] = ""
    return hashlib.sha256(canonical_json(stable).encode("utf-8")).hexdigest()


def resolve_model_specs(cache_resolver: CacheResolver = resolve_cached_gguf) -> list[JsonDict]:
    """Resolve the mandated MODEL_SPECS to cached local GGUF file paths."""

    resolved: list[JsonDict] = []
    for base in MODEL_SPECS:
        model_path = cache_resolver(str(base["hf_id"]), str(base["preferred_quant"]))
        local_model_present = bool(model_path and Path(model_path).is_file())
        receipt = model_file_receipt(model_path) if local_model_present else {}
        resolved.append(
            {
                **base,
                "model_path": model_path,
                "local_model_present": local_model_present,
                "model_file": receipt.get("model_file"),
                "model_filename": receipt.get("model_filename"),
                "model_size_bytes": receipt.get("model_size_bytes"),
                "quant": receipt.get("quant", base["preferred_quant"]),
            }
        )
    return resolved


def model_file_receipt(model_path: str | None) -> JsonDict:
    """Return auditable file metadata without hashing multi-GB weights."""

    path = Path(str(model_path))
    return {
        "model_file": str(path),
        "model_filename": path.name,
        "model_size_bytes": path.stat().st_size if path.exists() else None,
        "quant": quant_from_filename(path.name),
    }


def quant_from_filename(filename: str) -> str:
    """Extract the GGUF quant token used by the local model file."""

    for token in (
        "UD-Q8_K_XL",
        "UD-Q6_K_XL",
        "UD-Q5_K_M",
        "UD-Q5_K_S",
        "UD-Q4_K_M",
        "UD-Q4_K_S",
        "UD-Q4_K_XL",
        "UD-Q3_K_M",
        "UD-Q3_K_S",
        "UD-Q2_K_XL",
        "Q8_0",
        "Q6_K",
        "Q5_K_M",
        "Q4_K_M",
        "Q3_K_M",
        "Q2_K",
        "BF16",
        "MXFP4_MOE",
    ):
        if token.lower() in filename.lower():
            return token
    return "unknown"


def select_headline_specs(
    model_specs: Sequence[Mapping[str, Any]],
    *,
    pair_resolver: PairResolver = cached_sota_pair,
    max_headline_models: int = 1,
) -> list[JsonDict]:
    """Select cached mandated headline models, preferring cached_sota_pair order."""

    selected: list[JsonDict] = []
    by_id = {str(row["hf_id"]): row for row in model_specs}
    pair = pair_resolver() or []
    for pair_row in pair:
        hf_id = str(pair_row.get("hf_id"))
        spec = by_id.get(hf_id)
        if spec and spec.get("local_model_present") is True and hf_id not in {row["hf_id"] for row in selected}:
            selected.append(dict(spec))
    for spec in model_specs:
        hf_id = str(spec["hf_id"])
        if spec.get("local_model_present") is True and hf_id not in {row["hf_id"] for row in selected}:
            selected.append(dict(spec))
    return selected[: max(0, max_headline_models)]


def build_prompt(fixture: Mapping[str, Any]) -> str:
    """Build a free-form generation prompt without grammar or token steering."""

    lines = [
        "/no_think",
        "Output must begin with { and must be valid compact JSON only.",
        "No analysis, no markdown, no prose before or after the JSON object.",
        "Propose candidate claim states for this tiny hard/soft validation fixture.",
        "Hard constraints are final. Soft preferences only rank hard-feasible states.",
        "Return a JSON object with an instances list. Each row needs instance_id and either",
        "an assignment object or abstain=true when hard constraints conflict.",
    ]
    for instance in fixture["instances"]:
        domains = fixture_mod.domains_from_instance(instance)
        hard = [
            {
                "id": constraint["id"],
                "any_literal": [
                    {literal["variable"]: literal["equals"]} for literal in constraint["literals"]
                ],
            }
            for constraint in instance["hard_constraints"]
        ]
        soft = [
            {
                "id": preference["id"],
                "reward": {preference["variable"]: preference["value"]},
                "weight": preference["weight"],
            }
            for preference in instance["soft_preferences"]
        ]
        lines.append(
            canonical_json(
                {
                    "instance_id": instance["instance_id"],
                    "domains": domains,
                    "hard_constraints": hard,
                    "soft_preferences": soft,
                    "expected_status_hint": instance["expected_status"],
                }
            )
        )
    return "\n".join(lines)


def parse_candidate_payload(text: str) -> JsonDict:
    """Parse the first JSON object from unconstrained model text."""

    decoder = json.JSONDecoder()
    for index, char in enumerate(text):
        if char != "{":
            continue
        try:
            payload, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    return {}


def candidate_rows_by_instance(payload: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    """Index parsed model rows by fixture instance id."""

    rows = payload.get("instances", [])
    if not isinstance(rows, list):
        return {}
    return {
        str(row.get("instance_id")): row
        for row in rows
        if isinstance(row, Mapping) and row.get("instance_id")
    }


def coerce_assignment(
    instance: Mapping[str, Any],
    candidate_row: Mapping[str, Any] | None,
) -> tuple[dict[str, str] | None, bool, str]:
    """Return a normalized assignment, abstention flag, and parse status."""

    if not isinstance(candidate_row, Mapping):
        return None, True, "missing_candidate"
    if candidate_row.get("abstain") is True:
        return None, True, "model_abstained"
    raw_assignment = candidate_row.get("assignment")
    if not isinstance(raw_assignment, Mapping):
        return None, True, "missing_assignment"
    domains = fixture_mod.domains_from_instance(instance)
    assignment = {str(key): str(value) for key, value in raw_assignment.items()}
    if set(assignment) != set(domains):
        return None, True, "invalid_assignment_keys"
    if any(assignment[name] not in domains[name] for name in domains):
        return None, True, "invalid_assignment_domain"
    return assignment, False, "parsed_assignment"


def score_instance(
    *,
    model_spec: Mapping[str, Any],
    instance: Mapping[str, Any],
    candidate_row: Mapping[str, Any] | None,
) -> JsonDict:
    """Score one model-produced claim state with the Exp5499 exact validators."""

    reference = fixture_mod.solve_reference(instance)
    assignment, abstained, parse_status = coerce_assignment(instance, candidate_row)
    base = {
        "model_hf_id": model_spec["hf_id"],
        "instance_id": instance["instance_id"],
        "expected_status": instance["expected_status"],
        "reference_status": reference["status"],
        "reference_assignment": reference["assignment"],
        "parse_status": parse_status,
        "abstained": abstained,
        "claim_types": [claim["claim_type"] for claim in instance["typed_claims"]],
        "hard_constraint_ids": [row["id"] for row in instance["hard_constraints"]],
        "soft_preference_ids": [row["id"] for row in instance["soft_preferences"]],
    }
    if abstained:
        correct = reference["status"] == "infeasible"
        return {
            **base,
            "assignment": None,
            "hard_constraints_pass": False,
            "soft_score": None,
            "soft_optimal": False,
            "reference_agreement": correct,
            "exact_validator_correct": correct,
            "exact_validator_verdict": "correct_abstention" if correct else "abstention",
        }

    hard_ok = fixture_mod.hard_constraints_pass(instance, assignment)
    soft_score = fixture_mod.soft_score(instance, assignment)
    soft_optimal = bool(
        reference["status"] == "optimal"
        and hard_ok
        and soft_score == reference["objective_score"]
    )
    reference_agreement = bool(
        soft_optimal
        and assignment == reference["assignment"]
        and fixture_mod.assignment_hash(assignment) == reference["assignment_hash"]
    )
    if not hard_ok:
        verdict = "hard_constraint_violation"
    elif reference_agreement:
        verdict = "exact_match"
    else:
        verdict = "soft_suboptimal"
    return {
        **base,
        "assignment": assignment,
        "hard_constraints_pass": hard_ok,
        "soft_score": soft_score,
        "soft_optimal": soft_optimal,
        "reference_agreement": reference_agreement,
        "exact_validator_correct": reference_agreement,
        "exact_validator_verdict": verdict,
    }


def score_model_generation(
    *,
    model_spec: Mapping[str, Any],
    generation: Mapping[str, Any],
    fixture: Mapping[str, Any],
) -> list[JsonDict]:
    """Score all Exp5499 instances for one model generation."""

    payload = parse_candidate_payload(str(generation.get("output_text", "")))
    rows_by_id = candidate_rows_by_instance(payload)
    return [
        score_instance(
            model_spec=model_spec,
            instance=instance,
            candidate_row=rows_by_id.get(str(instance["instance_id"])),
        )
        for instance in fixture["instances"]
    ]


def aggregate_metrics(
    *,
    telemetry: Sequence[Mapping[str, Any]],
    generation_receipts: Sequence[Mapping[str, Any]],
    load_receipts: Sequence[Mapping[str, Any]],
) -> JsonDict:
    """Aggregate exact-validator and runtime telemetry into artifact fields."""

    attempts = [row for row in telemetry if row.get("abstained") is not True]
    expected_optimal = [row for row in telemetry if row.get("expected_status") == "optimal"]
    token_counts = {
        "prompt_tokens": sum(int(row.get("prompt_tokens", 0)) for row in generation_receipts),
        "completion_tokens": sum(int(row.get("completion_tokens", 0)) for row in generation_receipts),
        "total_tokens": sum(int(row.get("total_tokens", 0)) for row in generation_receipts),
    }
    return {
        "exact_validator_accuracy": _rate(
            sum(int(row.get("exact_validator_correct") is True) for row in telemetry),
            len(telemetry),
        ),
        "hard_constraint_violation_rate": _rate(
            sum(int(row.get("hard_constraints_pass") is False) for row in attempts),
            len(attempts),
        ),
        "preference_optimality_rate": _rate(
            sum(int(row.get("soft_optimal") is True) for row in expected_optimal),
            len(expected_optimal),
        ),
        "concept_claim_telemetry_rows": len(telemetry),
        "abstention_count": sum(int(row.get("abstained") is True) for row in telemetry),
        "token_counts": token_counts,
        "gpu_memory_delta_mb": max(
            [float(row.get("gpu_memory_delta_mb", 0.0) or 0.0) for row in load_receipts] or [0.0]
        ),
        "gpu_offload_verified": any(row.get("gpu_offload_verified") is True for row in load_receipts),
    }


def run(
    *,
    result_path: Path | str = REPO_ROOT / RESULT_RELATIVE_PATH,
    cache_resolver: CacheResolver = resolve_cached_gguf,
    runtime_probe: RuntimeProbe = None,
    runtime_factory: RuntimeFactory | None = None,
    pair_resolver: PairResolver = cached_sota_pair,
    max_headline_models: int = 1,
    tests_run: Sequence[Mapping[str, Any]] = (),
    write: bool = True,
) -> JsonDict:
    """Run the bounded SOTA panel or emit a blocked precondition artifact."""

    started = time.perf_counter()
    fixture_artifact = load_fixture_artifact()
    fixture = fixture_artifact["fixture"]
    model_specs = resolve_model_specs(cache_resolver=cache_resolver)
    preflight = dict((runtime_probe or default_runtime_probe)())
    selected_specs = select_headline_specs(
        model_specs,
        pair_resolver=pair_resolver,
        max_headline_models=max_headline_models,
    )
    cached_missing = [str(row["hf_id"]) for row in model_specs if row["local_model_present"] is not True]
    cached_present = [row for row in model_specs if row["local_model_present"] is True]
    blocked_reasons: list[str] = []
    if not cached_present:
        blocked_reasons.append("no_cached_mandated_sota_gguf")
    if preflight.get("runtime_ready") is not True:
        blocked_reasons.extend(str(reason) for reason in preflight.get("blocked_reasons", []))

    telemetry: list[JsonDict] = []
    generation_receipts: list[JsonDict] = []
    load_receipts: list[JsonDict] = []
    runtime_errors: list[JsonDict] = []
    headline_models_used: list[str] = []

    if not blocked_reasons:
        prompt = build_prompt(fixture)
        factory = runtime_factory or default_runtime_factory
        for spec in selected_specs:
            runtime = None
            try:
                runtime = factory(spec)
                load_receipt = dict(runtime.load_receipt)
                load_receipt.setdefault("model_hf_id", spec["hf_id"])
                load_receipts.append(load_receipt)
                if load_receipt.get("gpu_offload_verified") is not True:
                    runtime_errors.append(
                        {"model_hf_id": spec["hf_id"], "error": "gpu_offload_not_verified_after_load"}
                    )
                    continue
                generation = dict(runtime.generate(prompt))
                generation.setdefault("model_hf_id", spec["hf_id"])
                generation_receipts.append(generation)
                telemetry.extend(
                    score_model_generation(model_spec=spec, generation=generation, fixture=fixture)
                )
                headline_models_used.append(str(spec["hf_id"]))
            except Exception as exc:
                runtime_errors.append(
                    {"model_hf_id": spec["hf_id"], "error": f"{type(exc).__name__}: {exc}"}
                )
            finally:
                if runtime is not None and hasattr(runtime, "close"):
                    runtime.close()
        if not headline_models_used:
            blocked_reasons.append("no_gpu_offloaded_headline_model_completed")

    metrics = aggregate_metrics(
        telemetry=telemetry,
        generation_receipts=generation_receipts,
        load_receipts=load_receipts,
    )
    exact_validator_verdicts = [
        {
            "model_hf_id": row["model_hf_id"],
            "instance_id": row["instance_id"],
            "verdict": row["exact_validator_verdict"],
            "correct": row["exact_validator_correct"],
        }
        for row in telemetry
    ]
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "run_date": RUN_DATE,
        "spec_refs": list(SPEC_REFS),
        "model_specs": model_specs,
        "headline_models_used": headline_models_used,
        "legacy_smoke_models_used": [],
        "cached_models_missing": cached_missing,
        "llama_cpp_cuda_available": bool(preflight.get("llama_cpp_cuda_available")),
        "gpu_offload_verified": metrics["gpu_offload_verified"],
        "gpu_memory_delta_mb": metrics["gpu_memory_delta_mb"],
        "fixture_artifact": FIXTURE_ARTIFACT_RELATIVE_PATH.as_posix(),
        "exact_validator_accuracy": metrics["exact_validator_accuracy"],
        "hard_constraint_violation_rate": metrics["hard_constraint_violation_rate"],
        "preference_optimality_rate": metrics["preference_optimality_rate"],
        "concept_claim_telemetry_rows": metrics["concept_claim_telemetry_rows"],
        "guided_decoding_used": False,
        "token_steering_used": False,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "honest_verdict": honest_verdict(blocked_reasons, metrics["exact_validator_accuracy"]),
        "abstention_count": metrics["abstention_count"],
        "token_counts": metrics["token_counts"],
        "exact_validator_verdicts": exact_validator_verdicts,
        "concept_claim_telemetry": telemetry,
        "generation_receipts": generation_receipts,
        "runtime_load_receipts": load_receipts,
        "runtime_errors": runtime_errors,
        "runtime_preflight": preflight,
        "blocked_reasons": sorted(set(blocked_reasons)),
        "llama_cpp_command_or_binding": command_or_binding(load_receipts, generation_receipts),
        "n_gpu_layers": n_gpu_layers_used(load_receipts),
        "wall_time_s": round(time.perf_counter() - started, 6),
        "fixture_sha256": fixture_mod.sha256_json(fixture),
        "guided_decoding_policy": "not_used_no_grammar_no_token_steering",
        "research_conductor_modified": False,
        "tests_run": [dict(row) for row in tests_run],
        "reproducibility_checksum": "",
    }
    artifact["reproducibility_checksum"] = payload_checksum(artifact)
    validate_artifact(artifact)
    if write:
        output = Path(result_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(artifact, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
            encoding="utf-8",
        )
    return artifact


def load_fixture_artifact(path: Path = REPO_ROOT / FIXTURE_ARTIFACT_RELATIVE_PATH) -> JsonDict:
    """Load Exp5499's artifact because this panel depends on its fixture."""

    artifact = json.loads(path.read_text(encoding="utf-8"))
    fixture_mod.validate_fixture(artifact["fixture"])
    return artifact


def honest_verdict(blocked_reasons: Sequence[str], accuracy: float) -> str:
    """Return a terminal verdict that distinguishes blocked and measured runs."""

    if blocked_reasons:
        return "blocked: sota_concept_claim_panel_" + "_".join(sorted(set(blocked_reasons)))
    return f"complete: live_sota_claim_panel_measured_accuracy_{accuracy}"


def command_or_binding(
    load_receipts: Sequence[Mapping[str, Any]],
    generation_receipts: Sequence[Mapping[str, Any]],
) -> str | None:
    """Return the llama.cpp binding or command recorded by the runtime."""

    for row in generation_receipts:
        value = row.get("llama_cpp_command_or_binding")
        if value:
            return str(value)
    for row in load_receipts:
        value = row.get("llama_cpp_binding") or row.get("llama_cpp_command_or_binding")
        if value:
            return str(value)
    return None


def n_gpu_layers_used(load_receipts: Sequence[Mapping[str, Any]]) -> int | None:
    """Return the first recorded n_gpu_layers value from runtime receipts."""

    for row in load_receipts:
        if row.get("n_gpu_layers") is not None:
            return int(row["n_gpu_layers"])
    return None


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Validate the required Exp5500 artifact contract."""

    for field in REQUIRED_ARTIFACT_FIELDS:
        _require(field in artifact, field)
    _require(artifact.get("fixture_artifact") == FIXTURE_ARTIFACT_RELATIVE_PATH.as_posix(), "fixture_artifact")
    _require(artifact.get("guided_decoding_used") is False, "guided_decoding_used")
    _require(artifact.get("token_steering_used") is False, "token_steering_used")
    _require(artifact.get("inference_substrate") == INFERENCE_SUBSTRATE, "inference_substrate")
    _require(artifact.get("legacy_smoke_models_used") == [], "legacy_smoke_models_used")
    _require(artifact.get("research_conductor_modified") is False, "research_conductor_modified")
    _require(str(artifact.get("honest_verdict", "")).startswith(("complete:", "blocked:")), "honest_verdict")
    _require(
        [row.get("hf_id") for row in artifact.get("model_specs", [])]
        == list(MANDATED_HEADLINE_MODEL_IDS),
        "model_specs",
    )
    for field in (
        "exact_validator_accuracy",
        "hard_constraint_violation_rate",
        "preference_optimality_rate",
    ):
        value = float(artifact.get(field, -1.0))
        _require(0.0 <= value <= 1.0, field)
    _require(isinstance(artifact.get("concept_claim_telemetry_rows"), int), "concept_claim_telemetry_rows")
    _require(artifact.get("reproducibility_checksum") == payload_checksum(artifact), "checksum")


def default_runtime_probe() -> JsonDict:  # pragma: no cover
    """Check CUDA and llama.cpp offload before any headline model load."""

    cuda_available = False
    cuda_device_count = 0
    torch_error = None
    try:
        import torch  # noqa: PLC0415

        cuda_available = bool(torch.cuda.is_available())
        cuda_device_count = int(torch.cuda.device_count())
    except Exception as exc:  # noqa: BLE001
        torch_error = f"{type(exc).__name__}: {exc}"

    llama_cpp_import_ok = False
    llama_cpp_cuda_available = False
    llama_error = None
    system_info = None
    try:
        from llama_cpp import llama_cpp  # noqa: PLC0415

        llama_cpp_import_ok = True
        llama_cpp_cuda_available = bool(llama_cpp.llama_supports_gpu_offload())
        raw_info = llama_cpp.llama_print_system_info()
        system_info = raw_info.decode("utf-8", "replace") if isinstance(raw_info, bytes) else str(raw_info)
    except Exception as exc:  # noqa: BLE001
        llama_error = f"{type(exc).__name__}: {exc}"

    nvidia_smi = nvidia_smi_snapshot()
    blocked = []
    if not cuda_available or cuda_device_count <= 0:
        blocked.append("cuda_unavailable")
    if not llama_cpp_import_ok:
        blocked.append("llama_cpp_import_failed")
    if not llama_cpp_cuda_available:
        blocked.append("llama_cpp_gpu_offload_unavailable")
    if not nvidia_smi.get("ok"):
        blocked.append("nvidia_smi_unavailable")
    return {
        "cuda_available": cuda_available,
        "cuda_device_count": cuda_device_count,
        "torch_error": torch_error,
        "llama_cpp_import_ok": llama_cpp_import_ok,
        "llama_cpp_cuda_available": llama_cpp_cuda_available,
        "gpu_offload_supported": llama_cpp_cuda_available,
        "llama_cpp_error": llama_error,
        "system_info": system_info,
        "nvidia_smi": nvidia_smi,
        "runtime_ready": not blocked,
        "blocked_reasons": blocked,
    }


def nvidia_smi_snapshot() -> JsonDict:  # pragma: no cover
    """Collect lightweight GPU memory diagnostics from nvidia-smi."""

    try:
        proc = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.used,memory.total,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}", "gpus": []}
    if proc.returncode != 0:
        return {"ok": False, "error": proc.stderr.strip(), "gpus": []}
    gpus = []
    for line in proc.stdout.splitlines():
        index, name, used, total, util = [part.strip() for part in line.split(",", maxsplit=4)]
        gpus.append(
            {
                "index": int(index),
                "name": name,
                "memory_used_mb": float(used),
                "memory_total_mb": float(total),
                "utilization_gpu_pct": float(util),
            }
        )
    return {"ok": True, "gpus": gpus}


def total_gpu_memory_used_mb() -> float | None:  # pragma: no cover
    """Return total used GPU memory in MiB, or None when nvidia-smi is absent."""

    snapshot = nvidia_smi_snapshot()
    if not snapshot.get("ok"):
        return None
    return float(sum(float(row["memory_used_mb"]) for row in snapshot["gpus"]))


def default_runtime_factory(spec: Mapping[str, Any]) -> "LlamaCppClaimPanelRuntime":  # pragma: no cover
    """Create the real llama-cpp-python runtime for local delivery runs."""

    return LlamaCppClaimPanelRuntime(spec)


class LlamaCppClaimPanelRuntime:  # pragma: no cover
    """Small llama-cpp-python wrapper for bounded live local generation."""

    def __init__(self, spec: Mapping[str, Any]) -> None:
        from llama_cpp import Llama  # noqa: PLC0415

        self.spec = dict(spec)
        before = total_gpu_memory_used_mb()
        started = time.perf_counter()
        self.llm = Llama(
            model_path=str(spec["model_path"]),
            n_ctx=2048,
            n_batch=128,
            n_gpu_layers=N_GPU_LAYERS,
            seed=RANDOM_SEED,
            verbose=False,
        )
        after = total_gpu_memory_used_mb()
        delta = 0.0 if before is None or after is None else max(0.0, after - before)
        self.load_receipt = {
            "model_hf_id": spec["hf_id"],
            "model_file": spec["model_path"],
            "runtime_backend": "llama_cpp_python_cuda_gguf",
            "llama_cpp_binding": "llama_cpp.Llama",
            "n_gpu_layers": N_GPU_LAYERS,
            "gpu_memory_before_mb": before,
            "gpu_memory_after_mb": after,
            "gpu_memory_delta_mb": delta,
            "gpu_offload_verified": delta > 256.0,
            "load_wall_time_s": round(time.perf_counter() - started, 6),
        }

    def generate(self, prompt: str) -> JsonDict:
        started = time.perf_counter()
        result = self.llm.create_completion(
            prompt=prompt,
            max_tokens=768,
            temperature=0.0,
            top_p=1.0,
            seed=RANDOM_SEED,
            echo=False,
            stop=["</s>", "<end_of_turn>"],
        )
        choices = result.get("choices", []) if isinstance(result, Mapping) else []
        text = str(choices[0].get("text", "")) if choices else ""
        usage = result.get("usage", {}) if isinstance(result, Mapping) else {}
        prompt_tokens = len(self.llm.tokenize(prompt.encode("utf-8")))
        completion_tokens = int(usage.get("completion_tokens", 0) or 0)
        return {
            "model_hf_id": self.spec["hf_id"],
            "output_text": text,
            "wall_time_s": round(time.perf_counter() - started, 6),
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "llama_cpp_command_or_binding": (
                "llama_cpp.Llama.create_completion(max_tokens=768,"
                " temperature=0.0, top_p=1.0, n_gpu_layers=-1)"
            ),
        }

    def close(self) -> None:
        self.llm = None
        gc.collect()


def _rate(numerator: int | float, denominator: int) -> float:
    return round(float(numerator) / float(denominator), 6) if denominator else 0.0


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def main() -> int:  # pragma: no cover
    artifact = run()
    print(json.dumps({"result": RESULT_RELATIVE_PATH.as_posix(), "honest_verdict": artifact["honest_verdict"]}))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
