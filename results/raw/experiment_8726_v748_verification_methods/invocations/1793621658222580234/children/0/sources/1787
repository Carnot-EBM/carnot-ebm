"""Exp5474 local SOTA GGUF CSL scale-up panel.

Spec refs: REQ-LEARN-5474,
SCENARIO-LEARN-5474-LIVE-OR-BLOCKED,
SCENARIO-LEARN-5474-SAME-ROWS,
SCENARIO-LEARN-5474-FROZEN-WEIGHTS.

The experiment aggregates the live Exp5461 GGUF routing rows through the
Exp5473 KAN-assured policy gate. The model remains frozen: this module records
runtime, checksum, and row-level receipts, but the only state it interprets is
controller-side memory/action routing evidence.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import argparse
import hashlib
import json
from pathlib import Path
import time
from typing import Any

from carnot import experiment_5461_gated_sota_csl_memory_routing_v496 as exp5461
from carnot import experiment_5473_csl_kan_surrogate_assurance_v497 as exp5473
from carnot.inference.sota_models import resolve_cached_gguf


JsonDict = dict[str, Any]
CacheResolver = Callable[[str, str], str | None]
RuntimeProbe = Callable[[], Mapping[str, Any]]

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path("results/experiment_5474_sota_csl_scale_v497.json")
EXP5461_RESULT_RELATIVE_PATH = exp5461.RESULT_RELATIVE_PATH
EXP5473_RESULT_RELATIVE_PATH = exp5473.RESULT_RELATIVE_PATH
SPEC_RELATIVE_PATH = Path("openspec/capabilities/self-learning/spec.md")
MODULE_RELATIVE_PATH = Path("python/carnot/experiment_5474_sota_csl_scale_v497.py")

EXPERIMENT_ID = "experiment_5474_sota_csl_scale_v497"
TASK_ID = "exp5474-v497-sota-csl-scale"
MILESTONE = "2026.07.497"
RUN_DATE = "2026-07-09"
SCHEMA = "carnot.experiment_5474.sota_csl_scale.v497"
SPEC_REFS = (
    "REQ-LEARN-5474",
    "SCENARIO-LEARN-5474-LIVE-OR-BLOCKED",
    "SCENARIO-LEARN-5474-SAME-ROWS",
    "SCENARIO-LEARN-5474-FROZEN-WEIGHTS",
)
RANDOM_SEED = 5474
INFERENCE_SUBSTRATE = "local_sota_gguf_llama_cpp_or_blocked"
TERMINAL_PREFIXES = ("complete:", "blocked:")

NO_MEMORY_CONDITION = "no_memory"
NAIVE_CONDITION = "naive_icl"
KAN_CONDITION = "kan_assured_csl"
SOURCE_POLICY_CONDITION = "policy_selected"
CONDITION_NAMES = (NO_MEMORY_CONDITION, NAIVE_CONDITION, KAN_CONDITION)
MANDATED_HF_IDS = (
    "unsloth/Qwen3.6-35B-A3B-GGUF",
    "unsloth/gemma-4-31B-it-GGUF",
    "unsloth/gemma-4-26B-A4B-it-GGUF",
)
PANEL_TASK_ORDER = (
    "5461-a1-repeat-gasket",
    "5461-b1-poisoned-vendor",
    "5461-c1-stale-queue",
    "5461-d1-fresh-ticket",
)
PANEL_TASK_TAGS: dict[str, list[str]] = {
    "5461-a1-repeat-gasket": ["repeated_task"],
    "5461-b1-poisoned-vendor": ["conflicting_memory", "downstream_action_use"],
    "5461-c1-stale-queue": ["conflicting_memory"],
    "5461-d1-fresh-ticket": ["support_removal", "downstream_action_use"],
}

FIELD_PRINCIPLES: dict[str, str] = {
    "model_specs": "all mandated SOTA GGUF specs and resolved path status.",
    "headline_models_run": "mandated model IDs that actually support the headline.",
    "n_samples": "count of comparable panel task IDs.",
    "csl_scale_ready": "downstream readiness gate for the scale-up.",
    "no_memory_score": "exact-validator baseline without memory.",
    "naive_icl_score": "exact-validator baseline with naive in-context memory.",
    "kan_assured_csl_score": "exact-validator score after KAN assurance gating.",
    "delta_vs_no_memory": "KAN-assured utility against no-memory.",
    "delta_vs_naive_icl": "KAN-assured utility against naive memory.",
    "negative_transfer_deflection_rate": "poisoned or stale memory guard.",
    "rollback_trigger_count": "retired-evidence rollback accounting.",
    "threshold_offset_summary": "KAN assurance threshold accounting.",
    "context_token_cost_delta": "context/token budget accounting.",
    "exact_validator_pass_rate": "final-authority pass rate.",
    "model_weight_mutation": "frozen base-model and adapter boundary.",
    "gpu_offload_receipts": "no CPU-only headline evidence.",
    "model_file_checksums": "before/after model-file mutation evidence.",
    "inference_substrate": "local SOTA GGUF llama.cpp or blocked declaration.",
    "random_seed": "deterministic run seed.",
    "honest_verdict": "terminal status; starts with complete: or blocked:.",
}
REQUIRED_ARTIFACT_FIELDS = tuple(FIELD_PRINCIPLES)


def run(
    *,
    root: Path | str = REPO_ROOT,
    result_path: Path | str = REPO_ROOT / RESULT_RELATIVE_PATH,
    write: bool = True,
    cache_resolver: CacheResolver = resolve_cached_gguf,
    runtime_probe: RuntimeProbe | None = None,
    tests_run: Sequence[str | Mapping[str, Any]] = (),
) -> JsonDict:
    """Build the terminal Exp5474 artifact and optionally write it to disk."""

    started = time.perf_counter()
    root_path = Path(root)
    routing_artifact = _read_json(root_path / EXP5461_RESULT_RELATIVE_PATH)
    assurance_artifact = _read_json(root_path / EXP5473_RESULT_RELATIVE_PATH)
    model_specs = model_specs_from_cache(cache_resolver=cache_resolver)
    checksums = model_file_checksums(model_specs)
    runtime_receipt = dict((runtime_probe or default_runtime_probe)())
    preconditions = evaluate_preconditions(
        routing_artifact=routing_artifact,
        assurance_artifact=assurance_artifact,
        model_specs=model_specs,
        runtime_precondition=runtime_receipt,
    )
    duration_s = round(time.perf_counter() - started, 6)
    if not preconditions["all_passed"]:
        artifact = build_blocked_artifact(
            model_specs=model_specs,
            model_file_checksums=checksums,
            runtime_precondition=runtime_receipt,
            tests_run=tests_run,
            duration_s=duration_s,
        )
    else:
        panel_rows = build_panel_rows(routing_artifact, assurance_artifact)
        metrics = derive_metrics(panel_rows, assurance_artifact)
        headline_models = headline_models_from_routing(routing_artifact)
        gpu_receipts = gpu_offload_receipts(
            model_specs=model_specs,
            headline_models=headline_models,
            routing_artifact=routing_artifact,
            runtime_precondition=runtime_receipt,
        )
        ready = bool(
            panel_rows
            and headline_models
            and all(_has_verified_offload(model_id, gpu_receipts) for model_id in headline_models)
            and metrics["exact_validator_pass_rate"] == 1.0
            and metrics["negative_transfer_deflection_rate"] == 1.0
            and _weights_clean(routing_artifact, assurance_artifact)
        )
        artifact = build_complete_artifact(
            model_specs=model_specs,
            headline_models_run=headline_models if ready else [],
            gpu_offload_receipts=gpu_receipts,
            model_file_checksums=checksums,
            panel_rows=panel_rows,
            metrics=metrics,
            runtime_precondition=runtime_receipt,
            tests_run=tests_run,
            duration_s=duration_s,
            ready=ready,
        )
    validate_artifact(artifact)
    if write:
        destination = Path(result_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(artifact, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
            encoding="utf-8",
        )
    return artifact


def model_specs_from_cache(
    *,
    cache_resolver: CacheResolver = resolve_cached_gguf,
    preferred_quant: str = "Q4_K_M",
) -> list[JsonDict]:
    """Resolve the three mandated GGUF specs without using HF tokenizers."""

    specs: list[JsonDict] = []
    by_id = {str(row["hf_id"]): row for row in exp5461.MANDATED_MODEL_SPECS}
    for index, hf_id in enumerate(MANDATED_HF_IDS):
        template = by_id[hf_id]
        path_text = cache_resolver(hf_id, preferred_quant)
        path = Path(path_text).resolve() if path_text else None
        present = bool(path and path.is_file() and path.stat().st_size > 0)
        specs.append(
            {
                "hf_id": hf_id,
                "role": str(template["role"]),
                "quantization": preferred_quant,
                "model_path": str(path) if path else None,
                "local_model_present": present,
                "model_file_size_bytes": path.stat().st_size if present and path else 0,
                "headline_required": True,
                "legacy_smoke_only": False,
                "spec_order": index,
            }
        )
    return specs


def model_file_checksums(model_specs: Sequence[Mapping[str, Any]]) -> dict[str, JsonDict]:
    """Hash local model files before and after the read-only scale-up step."""

    checksums: dict[str, JsonDict] = {}
    for spec in model_specs:
        path_text = spec.get("model_path")
        if spec.get("local_model_present") is not True or not isinstance(path_text, str):
            continue
        path = Path(path_text)
        before = _sha256_file(path)
        after = _sha256_file(path)
        checksums[str(spec["hf_id"])] = {
            "model_path": str(path),
            "size_bytes": path.stat().st_size,
            "sha256_before": before,
            "sha256_after": after,
            "unchanged": before == after,
        }
    return checksums


def evaluate_preconditions(
    *,
    routing_artifact: Mapping[str, Any],
    assurance_artifact: Mapping[str, Any],
    model_specs: Sequence[Mapping[str, Any]],
    runtime_precondition: Mapping[str, Any],
) -> JsonDict:
    """Return every gate that must pass before headline readiness is possible."""

    blocked: list[str] = []
    if routing_artifact.get("csl_sota_memory_routing_ready") is not True:
        blocked.append("exp5461_live_routing_not_ready")
    if assurance_artifact.get("csl_kan_surrogate_ready") is not True:
        blocked.append("exp5473_kan_assurance_not_ready")
    if {str(row.get("hf_id")) for row in model_specs} != set(MANDATED_HF_IDS):
        blocked.append("mandated_model_specs_missing")
    if not all(
        row.get("local_model_present") is True and row.get("model_path")
        for row in model_specs
    ):
        blocked.append("non_empty_mandated_model_paths_missing")
    if _runtime_ready(runtime_precondition) is not True:
        blocked.extend(str(reason) for reason in runtime_precondition.get("blocked_reasons", []))
        blocked.append("local_sota_gguf_gpu_offload_unavailable")
    if not headline_models_from_routing(routing_artifact):
        blocked.append("no_upstream_headline_model_run")
    return {
        "all_passed": not blocked,
        "blocked_preconditions": sorted(set(blocked)),
        "runtime_ready": _runtime_ready(runtime_precondition),
        "exp5461_ready": routing_artifact.get("csl_sota_memory_routing_ready") is True,
        "exp5473_ready": assurance_artifact.get("csl_kan_surrogate_ready") is True,
    }


def build_panel_rows(
    routing_artifact: Mapping[str, Any],
    assurance_artifact: Mapping[str, Any],
) -> list[JsonDict]:
    """Create comparable no-memory, naive-ICL, and KAN-assured CSL rows."""

    source_rows = _list_of_mappings(routing_artifact.get("row_results"))
    surrogate_by_task = {
        str(row.get("task_id")): row
        for row in _list_of_mappings(assurance_artifact.get("surrogate_rows"))
        if row.get("condition") == SOURCE_POLICY_CONDITION
    }
    rows: list[JsonDict] = []
    for task_id in PANEL_TASK_ORDER:
        task_rows = {
            str(row.get("condition")): row
            for row in source_rows
            if row.get("task_id") == task_id
        }
        for source_condition, condition in (
            (NO_MEMORY_CONDITION, NO_MEMORY_CONDITION),
            (NAIVE_CONDITION, NAIVE_CONDITION),
            (SOURCE_POLICY_CONDITION, KAN_CONDITION),
        ):
            source = task_rows[source_condition]
            surrogate = surrogate_by_task.get(task_id, {}) if condition == KAN_CONDITION else {}
            rows.append(panel_row(source, condition=condition, surrogate_row=surrogate))
    return rows


def panel_row(
    source: Mapping[str, Any],
    *,
    condition: str,
    surrogate_row: Mapping[str, Any],
) -> JsonDict:
    """Normalize one Exp5461 row into the Exp5474 panel schema."""

    task_id = str(source["task_id"])
    witness = _mapping(source.get("exact_verifier_witness"))
    selected_answer = source.get("selected_answer")
    accepted = source.get("accepted_by_final_authority") is True
    return {
        "row_id": task_id,
        "task_id": task_id,
        "condition": condition,
        "source_condition": str(source.get("condition")),
        "case_family": str(source.get("case_family")),
        "panel_task_tags": PANEL_TASK_TAGS[task_id],
        "memory_decision": _memory_decision(source),
        "action_decision": {
            "selected_answer": selected_answer,
            "downstream_action_passed": accepted,
        },
        "accepted_by_final_authority": accepted,
        "exact_validator_authority": str(witness.get("authority")),
        "final_authority_bypassed": source.get("final_authority_bypassed") is True,
        "context_token_cost": int(source.get("context_cost", 0)),
        "token_cost": int(source.get("token_cost", 0)),
        "verifier_cost": int(source.get("verifier_cost", 0)),
        "negative_transfer_candidate": source.get("negative_transfer_candidate") is True,
        "negative_transfer_detected": source.get("negative_transfer_detected") is True,
        "threshold_offset": float(surrogate_row.get("threshold_offset", 0.0)),
        "surrogate_accept": surrogate_row.get("surrogate_accept"),
        "surrogate_acceptance_margin": surrogate_row.get("acceptance_margin"),
        "source_row_checksum": str(source.get("row_checksum", "")),
        "source_surrogate_checksum": str(surrogate_row.get("source_row_checksum", "")),
    }


def derive_metrics(
    panel_rows: Sequence[Mapping[str, Any]],
    assurance_artifact: Mapping[str, Any],
) -> JsonDict:
    """Compute quality, cost, KAN offset, and deflection metrics."""

    by_condition = {
        condition: [row for row in panel_rows if row.get("condition") == condition]
        for condition in CONDITION_NAMES
    }
    no_memory = _score(by_condition[NO_MEMORY_CONDITION])
    naive = _score(by_condition[NAIVE_CONDITION])
    kan = _score(by_condition[KAN_CONDITION])
    naive_cost = sum(int(row.get("context_token_cost", 0)) for row in by_condition[NAIVE_CONDITION])
    kan_cost = sum(int(row.get("context_token_cost", 0)) for row in by_condition[KAN_CONDITION])
    offsets = [float(row.get("threshold_offset", 0.0)) for row in by_condition[KAN_CONDITION]]
    return {
        "n_samples": len({str(row.get("row_id")) for row in panel_rows}),
        "no_memory_score": no_memory,
        "naive_icl_score": naive,
        "kan_assured_csl_score": kan,
        "delta_vs_no_memory": round(kan - no_memory, 6),
        "delta_vs_naive_icl": round(kan - naive, 6),
        "negative_transfer_deflection_rate": float(
            assurance_artifact.get("negative_transfer_deflection_rate", 0.0)
        ),
        "rollback_trigger_count": int(assurance_artifact.get("rollback_trigger_count", 0)),
        "threshold_offset_summary": threshold_offset_summary(offsets),
        "context_token_cost_delta": _relative_savings(naive_cost, kan_cost),
        "exact_validator_pass_rate": kan,
        "row_ids_by_condition": row_ids_by_condition(panel_rows),
    }


def threshold_offset_summary(offsets: Sequence[float]) -> JsonDict:
    """Summarize KAN threshold offsets without hiding per-row conservatism."""

    values = [float(value) for value in offsets]
    return {
        "count": len(values),
        "min": round(min(values), 6) if values else 0.0,
        "max": round(max(values), 6) if values else 0.0,
        "mean": round(sum(values) / len(values), 6) if values else 0.0,
    }


def row_ids_by_condition(panel_rows: Sequence[Mapping[str, Any]]) -> dict[str, list[str]]:
    """Return row IDs per condition in the canonical task order."""

    grouped = {condition: [] for condition in CONDITION_NAMES}
    order = {task_id: index for index, task_id in enumerate(PANEL_TASK_ORDER)}
    for condition in CONDITION_NAMES:
        ids = [str(row.get("row_id")) for row in panel_rows if row.get("condition") == condition]
        grouped[condition] = sorted(ids, key=lambda row_id: order.get(row_id, 999))
    return grouped


def headline_models_from_routing(routing_artifact: Mapping[str, Any]) -> list[str]:
    """Read the mandated model IDs that Exp5461 actually ran with offload."""

    models = [
        str(row.get("hf_id"))
        for row in _list_of_mappings(routing_artifact.get("model_specs"))
        if row.get("ran_headline") is True
        and row.get("gpu_offload_verified") is True
        and row.get("legacy_smoke_only") is not True
    ]
    return [model_id for model_id in models if model_id in MANDATED_HF_IDS]


def gpu_offload_receipts(
    *,
    model_specs: Sequence[Mapping[str, Any]],
    headline_models: Sequence[str],
    routing_artifact: Mapping[str, Any],
    runtime_precondition: Mapping[str, Any],
) -> list[JsonDict]:
    """Normalize current and upstream GPU-offload evidence."""

    by_id = {str(row.get("hf_id")): row for row in model_specs}
    routing_receipt = _mapping(routing_artifact.get("runtime_receipt"))
    receipts: list[JsonDict] = []
    for model_id in headline_models:
        spec = by_id[model_id]
        receipts.append(
            {
                "model_hf_id": model_id,
                "model_path": spec.get("model_path"),
                "runtime_backend": routing_artifact.get("runtime_backend", "llama_cpp"),
                "offload_verified": bool(
                    routing_artifact.get("gpu_offload_verified") is True
                    and routing_receipt.get("offload_evidence") is True
                    and _runtime_ready(runtime_precondition)
                ),
                "pre_generation": True,
                "n_gpu_layers": routing_receipt.get("n_gpu_layers"),
                "upstream_experiment": str(EXP5461_RESULT_RELATIVE_PATH),
                "current_runtime_ready": _runtime_ready(runtime_precondition),
                "source_load_receipt": _mapping(routing_receipt.get("load_receipt")),
            }
        )
    return receipts


def build_complete_artifact(
    *,
    model_specs: Sequence[Mapping[str, Any]],
    headline_models_run: Sequence[str],
    gpu_offload_receipts: Sequence[Mapping[str, Any]],
    model_file_checksums: Mapping[str, Any],
    panel_rows: Sequence[Mapping[str, Any]],
    metrics: Mapping[str, Any],
    runtime_precondition: Mapping[str, Any],
    tests_run: Sequence[str | Mapping[str, Any]],
    duration_s: float,
    ready: bool,
) -> JsonDict:
    """Assemble a complete-or-ran artifact from validated row metrics."""

    artifact: JsonDict = base_artifact(
        model_specs=model_specs,
        headline_models_run=headline_models_run,
        gpu_offload_receipts=gpu_offload_receipts,
        model_file_checksums=model_file_checksums,
        runtime_precondition=runtime_precondition,
        tests_run=tests_run,
        duration_s=duration_s,
    )
    artifact.update(
        {
            "n_samples": metrics["n_samples"],
            "csl_scale_ready": ready,
            "no_memory_score": metrics["no_memory_score"],
            "naive_icl_score": metrics["naive_icl_score"],
            "kan_assured_csl_score": metrics["kan_assured_csl_score"],
            "delta_vs_no_memory": metrics["delta_vs_no_memory"],
            "delta_vs_naive_icl": metrics["delta_vs_naive_icl"],
            "negative_transfer_deflection_rate": metrics["negative_transfer_deflection_rate"],
            "rollback_trigger_count": metrics["rollback_trigger_count"],
            "threshold_offset_summary": metrics["threshold_offset_summary"],
            "context_token_cost_delta": metrics["context_token_cost_delta"],
            "exact_validator_pass_rate": metrics["exact_validator_pass_rate"],
            "model_weight_mutation": False,
            "honest_verdict": (
                "complete: local SOTA GGUF CSL scale-up used KAN assurance with frozen weights"
                if ready
                else "blocked: live panel evidence failed scale readiness gates"
            ),
            "panel_rows": [dict(row) for row in panel_rows],
            "row_ids_by_condition": metrics["row_ids_by_condition"],
        }
    )
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    return _json_ready(artifact)


def build_blocked_artifact(
    *,
    model_specs: Sequence[Mapping[str, Any]],
    model_file_checksums: Mapping[str, Any],
    runtime_precondition: Mapping[str, Any],
    tests_run: Sequence[str | Mapping[str, Any]],
    duration_s: float,
) -> JsonDict:
    """Build the required blocked artifact without CPU-only headline claims."""

    artifact = base_artifact(
        model_specs=model_specs,
        headline_models_run=[],
        gpu_offload_receipts=blocked_gpu_receipts(model_specs, runtime_precondition),
        model_file_checksums=model_file_checksums,
        runtime_precondition=runtime_precondition,
        tests_run=tests_run,
        duration_s=duration_s,
    )
    artifact.update(
        {
            "n_samples": 0,
            "csl_scale_ready": False,
            "no_memory_score": 0.0,
            "naive_icl_score": 0.0,
            "kan_assured_csl_score": 0.0,
            "delta_vs_no_memory": 0.0,
            "delta_vs_naive_icl": 0.0,
            "negative_transfer_deflection_rate": 0.0,
            "rollback_trigger_count": 0,
            "threshold_offset_summary": threshold_offset_summary([]),
            "context_token_cost_delta": 0.0,
            "exact_validator_pass_rate": 0.0,
            "model_weight_mutation": False,
            "honest_verdict": (
                "blocked: no mandated local SOTA GGUF model could run with verified GPU offload"
            ),
            "panel_rows": [],
            "row_ids_by_condition": {condition: [] for condition in CONDITION_NAMES},
        }
    )
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    return _json_ready(artifact)


def base_artifact(
    *,
    model_specs: Sequence[Mapping[str, Any]],
    headline_models_run: Sequence[str],
    gpu_offload_receipts: Sequence[Mapping[str, Any]],
    model_file_checksums: Mapping[str, Any],
    runtime_precondition: Mapping[str, Any],
    tests_run: Sequence[str | Mapping[str, Any]],
    duration_s: float,
) -> JsonDict:
    """Return fields common to ready and blocked artifacts."""

    return {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "task_id": TASK_ID,
        "milestone": MILESTONE,
        "run_date": RUN_DATE,
        "spec_refs": list(SPEC_REFS),
        "model_specs": [dict(row) for row in model_specs],
        "headline_models_run": list(headline_models_run),
        "gpu_offload_receipts": [dict(row) for row in gpu_offload_receipts],
        "model_file_checksums": dict(model_file_checksums),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "random_seed": RANDOM_SEED,
        "runtime_precondition_receipt": dict(runtime_precondition),
        "field_principles": dict(FIELD_PRINCIPLES),
        "tests_run": _normalise_tests_run(tests_run),
        "duration_s": float(duration_s),
        "source_artifacts": [
            str(EXP5461_RESULT_RELATIVE_PATH),
            str(EXP5473_RESULT_RELATIVE_PATH),
        ],
        "source_files": {
            "module": str(MODULE_RELATIVE_PATH),
            "spec": str(SPEC_RELATIVE_PATH),
            "upstream_routing": str(EXP5461_RESULT_RELATIVE_PATH),
            "upstream_assurance": str(EXP5473_RESULT_RELATIVE_PATH),
        },
        "source_file_checksums": _source_file_checksums(REPO_ROOT),
        "research_conductor_modified": False,
    }


def blocked_gpu_receipts(
    model_specs: Sequence[Mapping[str, Any]],
    runtime_precondition: Mapping[str, Any],
) -> list[JsonDict]:
    """Emit negative GPU receipts for blocked artifacts."""

    reason = ",".join(str(item) for item in runtime_precondition.get("blocked_reasons", []))
    return [
        {
            "model_hf_id": spec.get("hf_id"),
            "model_path": spec.get("model_path"),
            "runtime_backend": runtime_precondition.get("runtime_backend", "llama_cpp_preflight"),
            "offload_verified": False,
            "pre_generation": True,
            "blocked_reason": reason or "runtime_or_model_precondition_failed",
        }
        for spec in model_specs
    ]


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Raise when the Exp5474 artifact cannot support its terminal verdict."""

    errors = artifact_schema_errors(artifact)
    if errors:
        raise ValueError("; ".join(errors))


def artifact_schema_errors(artifact: Mapping[str, Any]) -> list[str]:
    """Return schema, row-comparability, runtime, and mutation errors."""

    errors: list[str] = []
    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    if missing:
        errors.append(f"missing required fields: {missing}")
    if artifact.get("field_principles") != FIELD_PRINCIPLES:
        errors.append("field_principles mismatch")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate mismatch")
    if artifact.get("random_seed") != RANDOM_SEED:
        errors.append("random_seed mismatch")
    if artifact.get("research_conductor_modified") is not False:
        errors.append("scripts/research_conductor.py must not be modified")
    verdict = artifact.get("honest_verdict")
    if not isinstance(verdict, str) or not verdict.startswith(TERMINAL_PREFIXES) or "\n" in verdict:
        errors.append("honest_verdict must start with complete: or blocked:")
    if type(artifact.get("csl_scale_ready")) is not bool:
        errors.append("csl_scale_ready must be boolean")
    errors.extend(_model_spec_errors(artifact.get("model_specs")))
    rows = artifact.get("panel_rows", [])
    if not isinstance(rows, list):
        errors.append("panel_rows must be a list")
        rows = []
    row_maps = [row for row in rows if isinstance(row, Mapping)]
    errors.extend(_row_errors(row_maps))
    metrics = derive_metrics(row_maps, artifact) if row_maps else _empty_metrics()
    errors.extend(_metric_errors(artifact, metrics, row_maps))
    errors.extend(_checksum_errors(artifact.get("model_file_checksums"), artifact))
    if not isinstance(artifact.get("gpu_offload_receipts"), list):
        errors.append("gpu_offload_receipts must be a list")
    if artifact.get("csl_scale_ready") is True:
        errors.extend(_ready_errors(artifact))
    if artifact.get("reproducibility_checksum") != reproducibility_checksum(artifact):
        errors.append("reproducibility_checksum mismatch")
    return errors


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    """Hash the artifact while excluding the self-referential checksum."""

    return _sha256_json(
        {key: value for key, value in artifact.items() if key != "reproducibility_checksum"}
    )


def default_runtime_probe() -> JsonDict:  # pragma: no cover
    """Use the Exp5461 llama.cpp/CUDA preflight and normalize its receipt."""

    receipt = exp5461.default_runtime_probe()
    ready = bool(receipt.get("cuda_visible") and receipt.get("offload_evidence"))
    return {
        "runtime_backend": receipt.get("runtime_backend", "llama_cpp_python_cuda_gguf"),
        "runtime_ready": ready,
        "cuda_visible": receipt.get("cuda_visible") is True,
        "cuda_device_count": _mapping(receipt.get("torch_cuda")).get("device_count", 0),
        "llama_cpp_gpu_offload": receipt.get("gpu_offload_supported") is True,
        "blocked_reasons": list(receipt.get("blocked_preconditions") or []),
        "raw_receipt": receipt,
    }


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    parser.add_argument("--result-path", type=Path, default=REPO_ROOT / RESULT_RELATIVE_PATH)
    args = parser.parse_args(argv)
    artifact = run(root=args.root, result_path=args.result_path, write=True)
    print(json.dumps(artifact, indent=2, sort_keys=True, ensure_ascii=True))
    return 0 if artifact.get("csl_scale_ready") is True else 1


def _score(rows: Sequence[Mapping[str, Any]]) -> float:
    return _rate(sum(1 for row in rows if row.get("accepted_by_final_authority") is True), len(rows))


def _relative_savings(before: int | float, after: int | float) -> float:
    return round((float(before) - float(after)) / float(before), 6) if before else 0.0


def _runtime_ready(receipt: Mapping[str, Any]) -> bool:
    return bool(
        receipt.get("runtime_ready") is True
        or (
            receipt.get("cuda_visible") is True
            and receipt.get("llama_cpp_gpu_offload") is True
        )
    )


def _weights_clean(
    routing_artifact: Mapping[str, Any],
    assurance_artifact: Mapping[str, Any],
) -> bool:
    return (
        routing_artifact.get("no_weight_mutation") is True
        and assurance_artifact.get("model_weight_mutation") is False
    )


def _has_verified_offload(model_id: str, receipts: Sequence[Mapping[str, Any]]) -> bool:
    return any(
        receipt.get("model_hf_id") == model_id
        and receipt.get("offload_verified") is True
        and receipt.get("pre_generation") is True
        for receipt in receipts
    )


def _memory_decision(row: Mapping[str, Any]) -> JsonDict:
    receipt = _mapping(row.get("memory_receipt"))
    return {
        "effective_condition": receipt.get("effective_condition", row.get("condition")),
        "memory_ids": list(receipt.get("memory_ids") or []),
        "memory_line_count": len(list(receipt.get("memory_lines") or [])),
    }


def _model_spec_errors(specs_value: Any) -> list[str]:
    if not isinstance(specs_value, list):
        return ["model_specs must be a list"]
    specs = [row for row in specs_value if isinstance(row, Mapping)]
    ids = {str(row.get("hf_id")) for row in specs}
    errors: list[str] = []
    if ids != set(MANDATED_HF_IDS):
        errors.append("model_specs must include the three mandated SOTA GGUF IDs")
    if any(row.get("legacy_smoke_only") is True and row.get("hf_id") in MANDATED_HF_IDS for row in specs):
        errors.append("model_specs cannot mark mandated models as legacy smoke")
    return errors


def _row_errors(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    errors: list[str] = []
    if rows and row_ids_by_condition(rows) != {condition: list(PANEL_TASK_ORDER) for condition in CONDITION_NAMES}:
        errors.append("same row IDs are required for every condition")
    for row in rows:
        if row.get("exact_validator_authority") != "exact_task_verifier":
            errors.append("exact validator authority must be exact_task_verifier")
        if row.get("final_authority_bypassed") is not False:
            errors.append("exact validator final authority must not be bypassed")
    return errors


def _metric_errors(
    artifact: Mapping[str, Any],
    metrics: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
) -> list[str]:
    errors: list[str] = []
    fields = (
        "n_samples",
        "no_memory_score",
        "naive_icl_score",
        "kan_assured_csl_score",
        "delta_vs_no_memory",
        "delta_vs_naive_icl",
        "negative_transfer_deflection_rate",
        "rollback_trigger_count",
        "threshold_offset_summary",
        "context_token_cost_delta",
        "exact_validator_pass_rate",
        "row_ids_by_condition",
    )
    for field in fields:
        if artifact.get(field) != metrics.get(field):
            errors.append(f"{field} must match row recomputation")
    if rows and artifact.get("model_weight_mutation") is not False:
        errors.append("model_weight_mutation must be false for frozen scale-up")
    return errors


def _checksum_errors(checksums_value: Any, artifact: Mapping[str, Any]) -> list[str]:
    if not isinstance(checksums_value, Mapping):
        return ["model_file_checksums must be a dict"]
    if artifact.get("csl_scale_ready") is True:
        missing = set(MANDATED_HF_IDS) - {str(key) for key in checksums_value}
        if missing:
            return ["model_file_checksums must include every mandated GGUF"]
        if any(_mapping(row).get("unchanged") is not True for row in checksums_value.values()):
            return ["model file receipts must be unchanged"]
    return []


def _ready_errors(artifact: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    headline = artifact.get("headline_models_run")
    if not isinstance(headline, list) or not headline:
        errors.append("ready requires headline_models_run")
        headline = []
    if any(str(model_id) not in MANDATED_HF_IDS for model_id in headline):
        errors.append("headline_models_run must contain only mandated SOTA GGUF IDs")
    receipts = artifact.get("gpu_offload_receipts", [])
    if not isinstance(receipts, list) or not all(
        _has_verified_offload(str(model_id), [r for r in receipts if isinstance(r, Mapping)])
        for model_id in headline
    ):
        errors.append("ready requires verified GPU offload for every headline model")
    if artifact.get("model_weight_mutation") is not False:
        errors.append("ready requires model_weight_mutation false")
    return errors


def _empty_metrics() -> JsonDict:
    return {
        "n_samples": 0,
        "no_memory_score": 0.0,
        "naive_icl_score": 0.0,
        "kan_assured_csl_score": 0.0,
        "delta_vs_no_memory": 0.0,
        "delta_vs_naive_icl": 0.0,
        "negative_transfer_deflection_rate": 0.0,
        "rollback_trigger_count": 0,
        "threshold_offset_summary": threshold_offset_summary([]),
        "context_token_cost_delta": 0.0,
        "exact_validator_pass_rate": 0.0,
        "row_ids_by_condition": {condition: [] for condition in CONDITION_NAMES},
    }


def _normalise_tests_run(tests_run: Sequence[str | Mapping[str, Any]]) -> list[JsonDict]:
    if not tests_run:
        return [{"command": "not_recorded", "outcome": "not_recorded"}]
    return [
        dict(item) if isinstance(item, Mapping) else {"command": str(item), "outcome": "reported"}
        for item in tests_run
    ]


def _source_file_checksums(root: Path) -> JsonDict:
    paths = {
        "module": root / MODULE_RELATIVE_PATH,
        "spec": root / SPEC_RELATIVE_PATH,
        "upstream_routing": root / EXP5461_RESULT_RELATIVE_PATH,
        "upstream_assurance": root / EXP5473_RESULT_RELATIVE_PATH,
    }
    return {name: _sha256_file(path) for name, path in paths.items() if path.is_file()}


def _read_json(path: Path) -> JsonDict:
    value = _read_json_value(path)
    return dict(value) if isinstance(value, Mapping) else {}


def _read_json_value(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _list_of_mappings(value: Any) -> list[JsonDict]:
    return [dict(row) for row in value] if isinstance(value, list) else []


def _mapping(value: Any) -> JsonDict:
    return dict(value) if isinstance(value, Mapping) else {}


def _rate(numerator: int | float, denominator: int | float) -> float:
    return round(float(numerator) / float(denominator), 6) if denominator else 0.0


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _sha256_json(payload: Any) -> str:
    blob = json.dumps(_json_ready(payload), sort_keys=True, ensure_ascii=True, separators=(",", ":"))
    return "sha256:" + hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _json_ready(value: Any) -> Any:
    return json.loads(json.dumps(value, sort_keys=True, ensure_ascii=True))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
