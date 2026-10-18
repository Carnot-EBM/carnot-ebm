#!/usr/bin/env python3
"""Exp 5378 structured methodology-duration receipt.

Spec refs: REQ-VERIFY-5378, SCENARIO-VERIFY-5378.

This experiment repairs the receipt boundary from Exp 5366: the structured
protocol rows were clean, but the live methodology duration excluded the model
load/runtime envelope.  The code below keeps the benchmark fixtures and
thresholds unchanged, fails closed on CPU-only GGUF runtime evidence, and emits
one receipt whose duration covers live local SOTA runtime work rather than
planning prose.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
import hashlib
import json
from pathlib import Path
import time
from typing import Any

from carnot import experiment_5365_grammar_budget_protocol_preflight_v489 as exp5365
from carnot import experiment_5366_live_grammar_budgeted_sota_protocol_v489 as exp5366
from carnot.inference.sota_models import resolve_cached_gguf


JsonDict = dict[str, Any]
ModelResolver = Callable[[str, str], str | None]
RuntimeProbe = Callable[..., JsonDict]
LiveRunner = Callable[..., JsonDict]

REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_ID = "experiment_5378_structured_methodology_duration_receipt_v490"
MILESTONE = "2026.07.490"
RESULT_RELATIVE_PATH = Path("results/experiment_5378_structured_methodology_duration_receipt_v490.json")
SCHEMA = "carnot.experiment_5378.structured_methodology_duration_receipt.v490"
SPEC_REFS = ("REQ-VERIFY-5378", "SCENARIO-VERIFY-5378")
RANDOM_SEED = 5378
TERMINAL_PREFIXES = ("complete:", "blocked_")
MANDATED_HF_IDS = exp5366.MANDATED_HF_IDS

REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "live_sota_receipt_ready",
    "grammar_budget_protocol_ready",
    "MODEL_SPECS",
    "selected_model_spec",
    "inference_substrate",
    "gpu_or_offload_receipt",
    "no_autotokenizer_used",
    "prompt_count",
    "parse_success_rate",
    "schema_success_rate",
    "final_json_extraction_rate",
    "semantic_success_rate",
    "truncation_failure_rate",
    "completion_slack_min_tokens",
    "unsafe_false_accepts",
    "methodology_duration_s",
    "active_roadmap_modified",
    "conductor_modified",
    "honest_verdict",
)

FIELD_PRINCIPLES: dict[str, str] = {
    "status": "complete only if live local SOTA inference ran; blocked if runtime preconditions fail.",
    "live_sota_receipt_ready": (
        "true only if a mandated SOTA GGUF produced non-retired live evidence "
        "with methodology_duration_s>=60."
    ),
    "grammar_budget_protocol_ready": "copied or recomputed from the .489 preflight.",
    "MODEL_SPECS": "list containing all mandated local GGUF model specs considered for headline results.",
    "selected_model_spec": "exact model spec used for headline measurements.",
    "inference_substrate": "concrete runtime path, GPU/offload status, and GGUF loader family.",
    "gpu_or_offload_receipt": "machine-readable evidence that this was not the retired CPU-only headline path.",
    "no_autotokenizer_used": "must be true for GGUF repositories.",
    "prompt_count": "number of live prompts evaluated.",
    "parse_success_rate": "fraction of responses with parseable JSON.",
    "schema_success_rate": "fraction of responses satisfying the schema.",
    "final_json_extraction_rate": "fraction with unambiguous final JSON extraction.",
    "semantic_success_rate": "fraction satisfying task semantics after schema validity.",
    "truncation_failure_rate": "fraction classified as token-budget truncation failures.",
    "completion_slack_min_tokens": "minimum JSON completion slack observed.",
    "unsafe_false_accepts": "count of invalid/unsafe outputs accepted as valid.",
    "methodology_duration_s": "live measurement duration excluding planning prose; must be >=60 for ready.",
    "active_roadmap_modified": "must be false.",
    "conductor_modified": "must be false.",
    "honest_verdict": "one-line ready/block verdict.",
}

ACCEPTANCE_THRESHOLDS: JsonDict = {
    "parse_success_rate": exp5366.MIN_PARSE_SUCCESS_RATE,
    "schema_success_rate": exp5366.MIN_SCHEMA_SUCCESS_RATE,
    "final_json_extraction_rate": exp5366.MIN_FINAL_JSON_EXTRACTION_RATE,
    "methodology_duration_s": exp5366.MIN_CLEAN_METHODOLOGY_DURATION_S,
    "unsafe_false_accepts": 0,
}


def field_provenance() -> dict[str, JsonDict]:
    """Return principle annotations for every required Exp 5378 field."""

    return {
        field: {
            "principle": principle,
            "satisfied_by": "Exp 5378 structured methodology-duration receipt builder",
        }
        for field, principle in FIELD_PRINCIPLES.items()
    }


def run(
    *,
    root: Path | str = REPO_ROOT,
    artifact_path: Path | str | None = None,
    exp5365_path: Path | str | None = None,
    exp5365_artifact: Mapping[str, Any] | None = None,
    model_resolver: ModelResolver = resolve_cached_gguf,
    runtime_probe: RuntimeProbe | None = None,
    live_runner: LiveRunner | None = None,
    tests_run: Sequence[Any] | None = None,
    write: bool = True,
) -> JsonDict:
    """Build the Exp 5378 receipt or a blocked artifact when preconditions fail."""

    started = time.perf_counter()
    root_path = Path(root)
    destination = _destination(root_path, artifact_path)
    gate_path = Path(exp5365_path) if exp5365_path is not None else root_path / exp5365.RESULT_RELATIVE_PATH
    gate = dict(exp5365_artifact or _load_json(gate_path))
    grammar_ready = bool(gate.get("grammar_budget_protocol_ready"))
    completion_slack = _completion_slack_from_gate(gate)
    model_specs = exp5366.default_model_specs_unresolved()
    selected_model: JsonDict | None = None
    runtime_receipt = _blocked_runtime_receipt(["exp5365_grammar_budget_protocol_not_ready"])
    blockers: list[str] = []
    live_artifact: JsonDict | None = None
    runtime_probe = exp5366.default_runtime_probe if runtime_probe is None else runtime_probe
    live_runner = _default_live_runner if live_runner is None else live_runner

    if not grammar_ready:
        blockers.append("exp5365_grammar_budget_protocol_not_ready")
    else:
        model_specs = exp5366.resolve_model_specs(model_resolver)
        selected_model = exp5366.select_headline_model(model_specs)
        if selected_model is None:
            blockers.append("no_mandated_sota_gguf_resolved")
        runtime_receipt = _normalise_runtime_receipt(
            runtime_probe(
                model_specs=model_specs,
                selected_model_spec=selected_model,
                exp5365_artifact=gate,
            )
        )
        blockers.extend(str(item) for item in runtime_receipt.get("blocked_preconditions", ()))
        if not runtime_receipt.get("non_retired_gpu_or_offload_path"):
            blockers.append("non_retired_gpu_or_offload_path_unavailable")
        blockers = _unique(blockers)
        runtime_receipt["blocked_preconditions"] = blockers
        if not blockers and selected_model is not None:
            live_started = time.perf_counter()
            live_artifact = dict(
                live_runner(
                    root=root_path,
                    artifact_path=destination,
                    exp5365_artifact=gate,
                    model_resolver=model_resolver,
                    runtime_probe=runtime_probe,
                    tests_run=[],
                    write=False,
                )
            )
            live_elapsed_s = time.perf_counter() - live_started
            runtime_receipt = _merge_runtime_receipt(runtime_receipt, live_artifact)
            selected_model = _selected_model_from_live(live_artifact, selected_model)
            model_specs = _model_specs_from_live(live_artifact, model_specs)
            methodology_duration_s = _methodology_duration(live_artifact, live_elapsed_s)
            return _finalize_artifact(
                started=started,
                destination=destination,
                write=write,
                grammar_ready=grammar_ready,
                completion_slack=completion_slack,
                model_specs=model_specs,
                selected_model=selected_model,
                runtime_receipt=runtime_receipt,
                live_artifact=live_artifact,
                methodology_duration_s=methodology_duration_s,
                tests_run=tests_run,
            )

    runtime_receipt["blocked_preconditions"] = _unique(blockers)
    return _finalize_artifact(
        started=started,
        destination=destination,
        write=write,
        grammar_ready=grammar_ready,
        completion_slack=completion_slack,
        model_specs=model_specs,
        selected_model=None,
        runtime_receipt=runtime_receipt,
        live_artifact=None,
        methodology_duration_s=0.0,
        tests_run=tests_run,
    )


def artifact_schema_errors(artifact: Mapping[str, Any]) -> list[str]:
    """Return schema and readiness errors for an Exp 5378 artifact."""

    errors: list[str] = []
    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    if missing:
        errors.append(f"missing required fields: {missing}")
    if artifact.get("status") not in {"complete", "blocked"}:
        errors.append("status must be complete or blocked")
    for field in ("live_sota_receipt_ready", "grammar_budget_protocol_ready"):
        if not isinstance(artifact.get(field), bool):
            errors.append(f"{field} must be boolean")
    if not _model_specs_cover_mandated(artifact.get("MODEL_SPECS")):
        errors.append("MODEL_SPECS must contain all mandated SOTA GGUF specs")
    selected = artifact.get("selected_model_spec")
    if selected is not None and (
        not isinstance(selected, Mapping) or selected.get("hf_id") not in MANDATED_HF_IDS
    ):
        errors.append("selected_model_spec must be null or one mandated model spec")
    substrate = artifact.get("inference_substrate")
    if not isinstance(substrate, Mapping):
        errors.append("inference_substrate must be object")
        substrate = {}
    receipt = artifact.get("gpu_or_offload_receipt")
    if not isinstance(receipt, Mapping):
        errors.append("gpu_or_offload_receipt must be object")
        receipt = {}
    if artifact.get("no_autotokenizer_used") is not True:
        errors.append("no_autotokenizer_used must be true")
    if not _non_negative_int(artifact.get("prompt_count")):
        errors.append("prompt_count must be non-negative integer")
    for field in (
        "parse_success_rate",
        "schema_success_rate",
        "final_json_extraction_rate",
        "semantic_success_rate",
        "truncation_failure_rate",
    ):
        if not _rate_is_valid(artifact.get(field)):
            errors.append(f"{field} must be in [0, 1]")
    if not isinstance(artifact.get("completion_slack_min_tokens"), int):
        errors.append("completion_slack_min_tokens must be integer")
    if not _non_negative_int(artifact.get("unsafe_false_accepts")):
        errors.append("unsafe_false_accepts must be non-negative integer")
    if not isinstance(artifact.get("methodology_duration_s"), int | float):
        errors.append("methodology_duration_s must be numeric")
    if artifact.get("active_roadmap_modified") is not False:
        errors.append("active_roadmap_modified must be false")
    if artifact.get("conductor_modified") is not False:
        errors.append("conductor_modified must be false")
    honest = artifact.get("honest_verdict")
    if not isinstance(honest, str) or not honest.startswith(TERMINAL_PREFIXES):
        errors.append("honest_verdict must start with complete: or blocked_")
    provenance = artifact.get("field_provenance")
    if not isinstance(provenance, Mapping) or any(field not in provenance for field in REQUIRED_ARTIFACT_FIELDS):
        errors.append("field_provenance must cover required fields")
    if artifact.get("status") == "complete" and selected is None:
        errors.append("complete status requires selected_model_spec")
    if artifact.get("status") == "complete" and artifact.get("live_sota_receipt_ready") is not True:
        errors.append("complete status requires live_sota_receipt_ready")
    if artifact.get("live_sota_receipt_ready") is True:
        if _numeric(artifact.get("methodology_duration_s")) < exp5366.MIN_CLEAN_METHODOLOGY_DURATION_S:
            errors.append("live_sota_receipt_ready requires methodology_duration_s>=60")
        if receipt.get("non_retired_gpu_or_offload_path") is not True:
            errors.append("ready receipt requires non-retired GPU/offload evidence")
        if substrate.get("live_local_sota_inference_ran") is not True:
            errors.append("ready receipt requires live local SOTA inference")
        if selected is None:
            errors.append("ready receipt requires selected_model_spec")
    return errors


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Raise when an Exp 5378 artifact cannot support downstream audit."""

    errors = artifact_schema_errors(artifact)
    if errors:
        raise AssertionError("; ".join(errors))


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - CLI wrapper
    """CLI entry point for producing the Exp 5378 receipt artifact."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    parser.add_argument("--artifact-path", type=Path, default=None)
    args = parser.parse_args(argv)
    artifact = run(root=args.root, artifact_path=args.artifact_path)
    print(json.dumps(artifact, indent=2, sort_keys=True, ensure_ascii=True))
    return 0 if artifact["status"] == "complete" else 1


def _default_live_runner(**kwargs: Any) -> JsonDict:  # pragma: no cover - live GGUF runtime
    return exp5366.run(**kwargs)


def _destination(root: Path, artifact_path: Path | str | None) -> Path:
    destination = Path(artifact_path) if artifact_path is not None else root / RESULT_RELATIVE_PATH
    return destination if destination.is_absolute() else root / destination


def _completion_slack_from_gate(gate: Mapping[str, Any]) -> int:
    value = gate.get("completion_slack_min_tokens")
    return int(value) if isinstance(value, int) and not isinstance(value, bool) else -1


def _blocked_runtime_receipt(blockers: Sequence[str]) -> JsonDict:
    return {
        "gpu_visible": False,
        "gguf_runtime_available": False,
        "gguf_loader_family": "not_checked",
        "offload_evidence": False,
        "non_retired_gpu_or_offload_path": False,
        "blocked_preconditions": list(blockers),
    }


def _normalise_runtime_receipt(receipt: Mapping[str, Any]) -> JsonDict:
    out = dict(receipt)
    out.setdefault("gpu_visible", False)
    out.setdefault("gguf_runtime_available", False)
    out.setdefault("gguf_loader_family", "llama.cpp")
    out.setdefault("offload_evidence", False)
    out.setdefault("non_retired_gpu_or_offload_path", False)
    out.setdefault("blocked_preconditions", [])
    out["blocked_preconditions"] = list(out.get("blocked_preconditions") or [])
    return out


def _merge_runtime_receipt(preflight_receipt: Mapping[str, Any], live_artifact: Mapping[str, Any]) -> JsonDict:
    live_receipt = live_artifact.get("gpu_or_offload_receipt")
    if isinstance(live_receipt, Mapping):
        merged = {**preflight_receipt, **live_receipt}
        return _normalise_runtime_receipt(merged)
    return _normalise_runtime_receipt(preflight_receipt)


def _selected_model_from_live(
    live_artifact: Mapping[str, Any],
    fallback: Mapping[str, Any] | None,
) -> JsonDict | None:
    selected = live_artifact.get("selected_model_spec")
    if isinstance(selected, Mapping) and selected.get("hf_id") in MANDATED_HF_IDS:
        return dict(selected)
    return dict(fallback) if fallback is not None else None


def _model_specs_from_live(live_artifact: Mapping[str, Any], fallback: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    model_specs = live_artifact.get("MODEL_SPECS")
    if _model_specs_cover_mandated(model_specs):
        return [dict(row) for row in model_specs]  # type: ignore[union-attr]
    return [dict(row) for row in fallback]


def _methodology_duration(live_artifact: Mapping[str, Any], elapsed_s: float) -> float:
    return round(
        max(
            _numeric(live_artifact.get("methodology_duration_s")),
            _numeric(live_artifact.get("duration_s")),
            float(elapsed_s),
        ),
        6,
    )


def _finalize_artifact(
    *,
    started: float,
    destination: Path,
    write: bool,
    grammar_ready: bool,
    completion_slack: int,
    model_specs: Sequence[Mapping[str, Any]],
    selected_model: Mapping[str, Any] | None,
    runtime_receipt: Mapping[str, Any],
    live_artifact: Mapping[str, Any] | None,
    methodology_duration_s: float,
    tests_run: Sequence[Any] | None,
) -> JsonDict:
    metrics = _metrics_from_live(live_artifact)
    no_autotokenizer_used = _no_autotokenizer_used(live_artifact, model_specs)
    live_ready = _live_sota_receipt_ready(
        selected_model=selected_model,
        runtime_receipt=runtime_receipt,
        live_artifact=live_artifact,
        methodology_duration_s=methodology_duration_s,
        no_autotokenizer_used=no_autotokenizer_used,
    )
    structured_clean = _structured_protocol_clean(metrics, methodology_duration_s)
    status = "complete" if live_ready else "blocked"
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "spec_refs": list(SPEC_REFS),
        "status": status,
        "live_sota_receipt_ready": live_ready,
        "grammar_budget_protocol_ready": grammar_ready,
        "grammar_budget_protocol_source": exp5365.RESULT_RELATIVE_PATH.as_posix(),
        "structured_protocol_clean": structured_clean,
        "MODEL_SPECS": [dict(row) for row in model_specs],
        "selected_model_spec": dict(selected_model) if live_ready and selected_model is not None else None,
        "inference_substrate": _inference_substrate(runtime_receipt, live_artifact, selected_model, live_ready),
        "gpu_or_offload_receipt": dict(runtime_receipt),
        "no_autotokenizer_used": no_autotokenizer_used,
        "prompt_count": metrics["prompt_count"],
        "parse_success_rate": metrics["parse_success_rate"],
        "schema_success_rate": metrics["schema_success_rate"],
        "final_json_extraction_rate": metrics["final_json_extraction_rate"],
        "semantic_success_rate": metrics["semantic_success_rate"],
        "truncation_failure_rate": metrics["truncation_failure_rate"],
        "completion_slack_min_tokens": metrics["completion_slack_min_tokens"]
        if live_artifact is not None
        else completion_slack,
        "unsafe_false_accepts": metrics["unsafe_false_accepts"],
        "methodology_duration_s": methodology_duration_s,
        "active_roadmap_modified": False,
        "conductor_modified": False,
        "acceptance_thresholds": dict(ACCEPTANCE_THRESHOLDS),
        "source_artifacts": _source_artifacts(grammar_ready, live_artifact),
        "tests_run": list(tests_run or []),
        "random_seed": RANDOM_SEED,
        "duration_s": round(time.perf_counter() - started, 6),
        "field_provenance": field_provenance(),
    }
    artifact["honest_verdict"] = _honest_verdict(artifact)
    artifact["reproducibility_checksum"] = _sha16(
        _stable_json(
            {
                "experiment_id": EXPERIMENT_ID,
                "grammar_ready": grammar_ready,
                "live_ready": live_ready,
                "methodology_duration_s": methodology_duration_s,
                "model_specs": artifact["MODEL_SPECS"],
                "selected_model": artifact["selected_model_spec"],
                "metrics": metrics,
                "seed": RANDOM_SEED,
            }
        )
    )
    validate_artifact(artifact)
    if write:
        _write_json(destination, artifact)
    return artifact


def _metrics_from_live(live_artifact: Mapping[str, Any] | None) -> JsonDict:
    if live_artifact is None:
        return {
            "prompt_count": 0,
            "parse_success_rate": 0.0,
            "schema_success_rate": 0.0,
            "final_json_extraction_rate": 0.0,
            "semantic_success_rate": 0.0,
            "truncation_failure_rate": 0.0,
            "completion_slack_min_tokens": -1,
            "unsafe_false_accepts": 0,
        }
    return {
        "prompt_count": int(live_artifact.get("prompt_count") or 0),
        "parse_success_rate": _numeric(live_artifact.get("parse_success_rate")),
        "schema_success_rate": _numeric(live_artifact.get("schema_success_rate")),
        "final_json_extraction_rate": _numeric(live_artifact.get("final_json_extraction_rate")),
        "semantic_success_rate": _numeric(live_artifact.get("semantic_success_rate")),
        "truncation_failure_rate": _numeric(live_artifact.get("truncation_failure_rate")),
        "completion_slack_min_tokens": _completion_slack_from_gate(live_artifact),
        "unsafe_false_accepts": int(live_artifact.get("unsafe_false_accepts") or 0),
    }


def _no_autotokenizer_used(
    live_artifact: Mapping[str, Any] | None,
    model_specs: Sequence[Mapping[str, Any]],
) -> bool:
    live_flag = True if live_artifact is None else live_artifact.get("no_autotokenizer_used") is True
    specs_clean = all(row.get("autotokenizer_used") is False for row in model_specs)
    return bool(live_flag and specs_clean)


def _live_sota_receipt_ready(
    *,
    selected_model: Mapping[str, Any] | None,
    runtime_receipt: Mapping[str, Any],
    live_artifact: Mapping[str, Any] | None,
    methodology_duration_s: float,
    no_autotokenizer_used: bool,
) -> bool:
    if live_artifact is None or selected_model is None:
        return False
    substrate = live_artifact.get("inference_substrate")
    substrate = substrate if isinstance(substrate, Mapping) else {}
    return bool(
        live_artifact.get("status") == "complete"
        and selected_model.get("hf_id") in MANDATED_HF_IDS
        and int(live_artifact.get("prompt_count") or 0) > 0
        and substrate.get("live_local_sota_inference_ran") is True
        and runtime_receipt.get("non_retired_gpu_or_offload_path") is True
        and no_autotokenizer_used
        and methodology_duration_s >= exp5366.MIN_CLEAN_METHODOLOGY_DURATION_S
    )


def _structured_protocol_clean(metrics: Mapping[str, Any], methodology_duration_s: float) -> bool:
    return bool(
        _numeric(metrics.get("parse_success_rate")) >= exp5366.MIN_PARSE_SUCCESS_RATE
        and _numeric(metrics.get("schema_success_rate")) >= exp5366.MIN_SCHEMA_SUCCESS_RATE
        and _numeric(metrics.get("final_json_extraction_rate")) >= exp5366.MIN_FINAL_JSON_EXTRACTION_RATE
        and metrics.get("unsafe_false_accepts") == 0
        and methodology_duration_s >= exp5366.MIN_CLEAN_METHODOLOGY_DURATION_S
    )


def _inference_substrate(
    runtime_receipt: Mapping[str, Any],
    live_artifact: Mapping[str, Any] | None,
    selected_model: Mapping[str, Any] | None,
    live_ready: bool,
) -> JsonDict:
    live_substrate = live_artifact.get("inference_substrate") if live_artifact is not None else None
    if isinstance(live_substrate, Mapping):
        out = dict(live_substrate)
    else:
        out = {}
    out.setdefault("kind", "live_llm_inference" if live_ready else "blocked_preconditions")
    out.setdefault("loader_family", str(runtime_receipt.get("gguf_loader_family") or "llama.cpp"))
    out.setdefault("gguf_loader_family", str(runtime_receipt.get("gguf_loader_family") or "llama.cpp"))
    out["gpu_or_offload_status"] = (
        "non_retired_gpu_or_offload_path"
        if runtime_receipt.get("non_retired_gpu_or_offload_path")
        else "blocked_or_cpu_only"
    )
    out["live_local_sota_inference_ran"] = bool(live_ready)
    out["selected_model_hf_id"] = None if selected_model is None else selected_model.get("hf_id")
    return out


def _source_artifacts(grammar_ready: bool, live_artifact: Mapping[str, Any] | None) -> list[JsonDict]:
    return [
        {
            "path": exp5365.RESULT_RELATIVE_PATH.as_posix(),
            "used": True,
            "grammar_budget_protocol_ready": grammar_ready,
        },
        {
            "path": exp5366.RESULT_RELATIVE_PATH.as_posix(),
            "used": live_artifact is not None,
            "purpose": "structured protocol fixtures, acceptance fields, and live GGUF receipt source",
        },
    ]


def _honest_verdict(artifact: Mapping[str, Any]) -> str:
    if artifact.get("grammar_budget_protocol_ready") is not True:
        return "blocked_exp5365_grammar_budget_protocol_not_ready"
    if artifact.get("status") == "complete" and artifact.get("structured_protocol_clean") is True:
        return "complete: live structured SOTA receipt ready with methodology_duration_s>=60"
    if artifact.get("live_sota_receipt_ready") is True:
        return "complete: live SOTA runtime receipt ready; structured protocol gate remains false"
    blockers = artifact.get("gpu_or_offload_receipt", {}).get("blocked_preconditions", [])
    first = blockers[0] if isinstance(blockers, list) and blockers else "preconditions_failed"
    return f"blocked_preconditions: {first}"


def _model_specs_cover_mandated(value: Any) -> bool:
    if not isinstance(value, list):
        return False
    return len(value) == len(MANDATED_HF_IDS) and {row.get("hf_id") for row in value if isinstance(row, Mapping)} == set(MANDATED_HF_IDS)


def _rate_is_valid(value: Any) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool) and 0.0 <= float(value) <= 1.0


def _non_negative_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _numeric(value: Any) -> float:
    return float(value) if isinstance(value, int | float) and not isinstance(value, bool) else -1.0


def _unique(values: Sequence[str]) -> list[str]:
    return list(dict.fromkeys(str(value) for value in values if value))


def _stable_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _sha16(value: str | bytes) -> str:
    data = value if isinstance(value, bytes) else value.encode("utf-8")
    return hashlib.sha256(data).hexdigest()[:16]


def _load_json(path: Path) -> JsonDict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n", encoding="utf-8")


if __name__ == "__main__":  # pragma: no cover - CLI wrapper
    raise SystemExit(main())
