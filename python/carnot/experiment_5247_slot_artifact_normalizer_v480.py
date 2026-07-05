"""Exp 5247 strict artifact schema/receipt normalizer.

Spec refs: REQ-REPORT-5247, SCENARIO-REPORT-5247-SAFE-REPAIR,
SCENARIO-REPORT-5247-UNSAFE-REJECTION,
SCENARIO-REPORT-5247-REPRESENTATIVE-479.

This module is intentionally a normalizer over artifact copies, not a historical
artifact migration. It makes shape-only repairs that preserve the original
evidence boundary and records every refused repair as an audit receipt.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import tempfile
import time
from typing import Any

from scripts import adversarial_verify as av


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path("results/experiment_5247_slot_artifact_normalizer_v480.json")
EXPERIMENT = "experiment_5247_slot_artifact_normalizer_v480"
EXPERIMENT_ID = "exp5247-slot-artifact-normalizer-v480"
MILESTONE = "2026.07.480"
RUN_DATE = "2026-07-05"
SCHEMA = "carnot.experiment_5247.slot_artifact_normalizer.v480"
INFERENCE_SUBSTRATE = "cached_fixture_replay_no_llm"
TERMINAL_PREFIXES = ("complete:", "blocked_")
SPEC_REFS = (
    "REQ-REPORT-5247",
    "SCENARIO-REPORT-5247-SAFE-REPAIR",
    "SCENARIO-REPORT-5247-UNSAFE-REJECTION",
    "SCENARIO-REPORT-5247-REPRESENTATIVE-479",
)
REPRESENTATIVE_RELATIVE_PATHS = (
    Path("results/experiment_5235_adversarial_qa_null_tautology_calibration_v479.json"),
    Path("results/experiment_5236_gap4_clean_status_after_qa_calibration_v479.json"),
    Path("results/experiment_5241_arc_gated_live_patch_attempt_v479.json"),
)

FIELD_PRINCIPLES: dict[str, str] = {
    "honest_verdict": (
        "Must start with complete: or blocked_ and state whether the strict "
        "normalizer is ready for gated consumers."
    ),
    "inference_substrate": (
        "Must be cached_fixture_replay_no_llm because Exp5247 only replays "
        "checked-in artifacts and tests the normalizer."
    ),
    "artifact_normalizer_ready": (
        "Bare bool gate for Exp5248; true only when strict safe-repair, "
        "unsafe-rejection, duration, gate, and principle tests pass."
    ),
    "artifact_normalizer_ready_principle": (
        "Explains why the bare gate is safe for downstream consumers."
    ),
    "safe_repairs_supported": (
        "Lists only copy/null/wrapper repairs that do not create methodology, "
        "duration, model, solve, or win evidence."
    ),
    "unsafe_repairs_rejected": "Lists missing-evidence and synthesis attempts the normalizer refuses.",
    "duration_policy_preserved": (
        "True only when compute-bound duration and methodology findings remain "
        "blocked after normalization."
    ),
    "conductor_modified": "False because the normalizer lives outside scripts/research_conductor.py.",
    "tests_run": (
        "Records command/outcome receipts for unit, coverage, spec, and artifact "
        "verification checks."
    ),
}

SAFE_REPAIRS_SUPPORTED = (
    "top_level_principle_wrapper_unwrap",
    "missing_explicit_null_for_declared_nullable_fields",
    "unambiguous_boolean_gate_field_extraction",
)
UNSAFE_REPAIRS_REJECTED = (
    "missing_methodology_receipts",
    "missing_duration_receipt",
    "subfloor_compute_duration",
    "missing_model_receipts",
    "missing_or_conflicting_gate_boolean",
    "missing_principle_for_required_fields",
    "solve_provenance_or_performance_win_synthesis",
)
REQUIRED_SCHEMA_FIELDS = (
    "schema",
    "experiment",
    "experiment_id",
    "milestone",
    "run_date",
    "spec_refs",
    "result_path",
    "duration_s",
    "field_principles",
    "representative_artifact_classifications",
    "safe_repair_receipts",
    "unsafe_rejection_receipts",
    "honest_verdict",
    "inference_substrate",
    "artifact_normalizer_ready",
    "artifact_normalizer_ready_principle",
    "safe_repairs_supported",
    "unsafe_repairs_rejected",
    "duration_policy_preserved",
    "conductor_modified",
    "tests_run",
    "reproducibility_checksum",
)

COMPUTE_BOUND_MARKERS = ("GGUF", "CUDA", "torch.cuda", "live_llm_inference")
LIVE_MODEL_MIN_DURATION_S = 60.0
METADATA_KEYS = {
    "principle",
    "field_principles",
    "field_provenance",
    "safe_repairs_supported",
    "unsafe_repairs_rejected",
}


@dataclass(frozen=True)
class NormalizationResult:
    """Normalized artifact copy plus receipts for every accepted/refused repair."""

    normalized: JsonDict
    safe_repairs: list[JsonDict]
    unsafe_rejections: list[JsonDict]
    ready_for_gated_consumers: bool


def _is_wrapper(value: Any) -> bool:
    return isinstance(value, Mapping) and "value" in value and "principle" in value


def _wrap(field: str, value: Any) -> JsonDict:
    return {"value": value, "principle": FIELD_PRINCIPLES[field]}


def _stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _checksum(payload: Mapping[str, Any]) -> str:
    stable = dict(payload)
    stable["reproducibility_checksum"] = ""
    return "sha256:" + hashlib.sha256(_stable_json(stable).encode("utf-8")).hexdigest()


def _field_principle(source: Mapping[str, Any], field: str) -> str | None:
    direct = source.get(field)
    if _is_wrapper(direct):
        principle = direct.get("principle")
        return principle if isinstance(principle, str) and principle.strip() else None
    sibling = source.get(f"{field}_principle")
    if isinstance(sibling, str) and sibling.strip():
        return sibling
    principles = source.get("field_principles")
    if isinstance(principles, Mapping):
        value = principles.get(field)
        if isinstance(value, str) and value.strip():
            return value
        if isinstance(value, Mapping):
            principle = value.get("principle")
            if isinstance(principle, str) and principle.strip():
                return principle
    return None


def _metadata_key(key: str) -> bool:
    return key in METADATA_KEYS or key.endswith("_principle")


def _gate_values(value: Any, wanted_key: str, path: tuple[str, ...] = ()) -> list[JsonDict]:
    rows: list[JsonDict] = []
    if isinstance(value, Mapping):
        for key, nested in value.items():
            key_text = str(key)
            if _metadata_key(key_text):
                continue
            nested_path = path + (key_text,)
            if key_text == wanted_key:
                gate_value = nested.get("value") if _is_wrapper(nested) else nested
                rows.append({"path": ".".join(nested_path), "value": gate_value})
            rows.extend(_gate_values(nested, wanted_key, nested_path))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            rows.extend(_gate_values(item, wanted_key, path + (f"[{index}]",)))
    return rows


def _finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _has_compute_marker(payload: Mapping[str, Any]) -> bool:
    text = json.dumps(payload, sort_keys=True, default=str)
    return any(marker in text for marker in COMPUTE_BOUND_MARKERS)


def _add_rejection(out: list[JsonDict], kind: str, field: str, detail: str) -> None:
    out.append({"kind": kind, "field": field, "detail": detail})


def _validate_duration_and_methodology(normalized: Mapping[str, Any], out: list[JsonDict]) -> None:
    if not _has_compute_marker(normalized):
        return
    duration = normalized.get("duration_s")
    if not _finite_number(duration):
        _add_rejection(out, "missing_duration_receipt", "duration_s", "compute-bound artifact")
    elif float(duration) < LIVE_MODEL_MIN_DURATION_S:
        _add_rejection(
            out,
            "duration_too_short",
            "duration_s",
            f"duration_s={duration} below {LIVE_MODEL_MIN_DURATION_S}s live-model floor",
        )
    has_model = bool(
        normalized.get("model_specs")
        or normalized.get("target_model")
        or normalized.get("models_tested")
    )
    has_seed = (
        normalized.get("random_seed") is not None
        or normalized.get("seed") is not None
        or bool(normalized.get("random_seeds_used"))
        or bool(normalized.get("seeds"))
    )
    has_checksum = bool(normalized.get("reproducibility_checksum"))
    missing = [
        name
        for name, present in (
            ("model_specs/target_model", has_model),
            ("random_seed", has_seed),
            ("reproducibility_checksum", has_checksum),
        )
        if not present
    ]
    if missing:
        _add_rejection(
            out,
            "missing_methodology_receipt",
            "methodology",
            "missing " + ", ".join(missing),
        )


def normalize_artifact(
    artifact: Mapping[str, Any],
    *,
    nullable_fields: Sequence[str] = (),
    gate_fields: Sequence[str] = (),
    required_principle_fields: Sequence[str] = (),
) -> NormalizationResult:
    """Return a strictly normalized artifact copy and auditable repair receipts."""

    normalized: JsonDict = {}
    safe_repairs: list[JsonDict] = []
    unsafe_rejections: list[JsonDict] = []

    for key, value in artifact.items():
        if _is_wrapper(value):
            normalized[key] = value["value"]
            safe_repairs.append({"kind": "top_level_wrapper_unwrapped", "field": key})
        else:
            normalized[key] = value

    for field in nullable_fields:
        if field not in normalized:
            normalized[field] = None
            safe_repairs.append({"kind": "missing_explicit_null_added", "field": field})

    for gate in gate_fields:
        if gate in normalized:
            if not isinstance(normalized[gate], bool):
                _add_rejection(
                    unsafe_rejections,
                    "nonboolean_gate_value",
                    gate,
                    f"top-level gate value is {type(normalized[gate]).__name__}",
                )
            continue
        candidates = _gate_values(artifact, gate)
        if not candidates:
            _add_rejection(unsafe_rejections, "missing_gate_boolean", gate, "no source value")
            continue
        nonbool = [row for row in candidates if not isinstance(row["value"], bool)]
        if nonbool:
            _add_rejection(
                unsafe_rejections,
                "nonboolean_gate_value",
                gate,
                f"non-boolean values at {[row['path'] for row in nonbool]}",
            )
            continue
        values = {row["value"] for row in candidates}
        if len(values) != 1:
            _add_rejection(
                unsafe_rejections,
                "conflicting_gate_boolean",
                gate,
                f"conflicting values at {[row['path'] for row in candidates]}",
            )
            continue
        normalized[gate] = values.pop()
        safe_repairs.append({"kind": "unambiguous_gate_boolean_extracted", "field": gate})

    substrate = normalized.get("inference_substrate")
    if not isinstance(substrate, str) or not substrate.strip():
        _add_rejection(
            unsafe_rejections,
            "missing_inference_substrate",
            "inference_substrate",
            "no explicit substrate declaration",
        )

    for field in required_principle_fields:
        if _field_principle(artifact, field) is None:
            _add_rejection(
                unsafe_rejections,
                "missing_principle",
                field,
                "required field has no wrapper, sibling, or field_principles entry",
            )

    _validate_duration_and_methodology(normalized, unsafe_rejections)

    if normalized.get("flagged_adversarial") is True:
        _add_rejection(
            unsafe_rejections,
            "source_flagged_adversarial",
            "flagged_adversarial",
            "existing source artifact marks itself adversarially flagged",
        )
    corrigendum = normalized.get("corrigendum_pending")
    if isinstance(corrigendum, list) and corrigendum:
        _add_rejection(
            unsafe_rejections,
            "source_corrigendum_pending",
            "corrigendum_pending",
            "existing source artifact carries corrigendum flags",
        )

    return NormalizationResult(
        normalized=normalized,
        safe_repairs=safe_repairs,
        unsafe_rejections=unsafe_rejections,
        ready_for_gated_consumers=not unsafe_rejections,
    )


def _flag_kinds(report: Mapping[str, Any]) -> list[str]:
    return sorted({str(flag.get("kind")) for flag in report.get("flags", []) if flag.get("kind")})


def _corrigendum_kinds(payload: Mapping[str, Any]) -> list[str]:
    rows = payload.get("corrigendum_pending")
    if not isinstance(rows, list):
        return []
    return sorted({str(row.get("kind")) for row in rows if isinstance(row, Mapping) and row.get("kind")})


def _verify_normalized_copy(normalized: Mapping[str, Any]) -> JsonDict:
    with tempfile.TemporaryDirectory(prefix="exp5247_normalized_") as tmp:
        path = Path(tmp) / "normalized.json"
        path.write_text(json.dumps(normalized, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return av.verify_artifact(path)


def classify_artifact_path(path: Path | str) -> JsonDict:
    """Classify one artifact before and after normalization without mutating it."""

    artifact_path = Path(path)
    payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"artifact is not a JSON object: {artifact_path}")
    before_report = av.verify_artifact(artifact_path)
    result = normalize_artifact(
        payload,
        required_principle_fields=("honest_verdict", "inference_substrate"),
    )
    after_report = _verify_normalized_copy(result.normalized)
    return {
        "path": str(artifact_path),
        "source_flagged_adversarial": payload.get("flagged_adversarial") is True,
        "source_corrigendum_kinds": _corrigendum_kinds(payload),
        "before_live_flag_kinds": _flag_kinds(before_report),
        "after_live_flag_kinds": _flag_kinds(after_report),
        "safe_repairs": [dict(row) for row in result.safe_repairs],
        "normalization_rejections": [str(row["kind"]) for row in result.unsafe_rejections],
        "normalizer_ready_for_gates": result.ready_for_gated_consumers,
    }


def classify_representative_artifacts(paths: Iterable[Path | str]) -> dict[str, JsonDict]:
    """Return read-only before/after classifications keyed by artifact filename."""

    return {Path(path).name: classify_artifact_path(path) for path in paths}


def build_artifact(
    *,
    representative_classifications: Mapping[str, Any],
    tests_run: Sequence[Mapping[str, Any]],
    duration_s: float,
) -> JsonDict:
    """Build the Exp 5247 terminal receipt."""

    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "run_date": RUN_DATE,
        "spec_refs": list(SPEC_REFS),
        "result_path": str(RESULT_RELATIVE_PATH),
        "duration_s": float(duration_s),
        "field_principles": dict(FIELD_PRINCIPLES),
        "representative_artifact_classifications": dict(representative_classifications),
        "safe_repair_receipts": list(SAFE_REPAIRS_SUPPORTED),
        "unsafe_rejection_receipts": list(UNSAFE_REPAIRS_REJECTED),
        "honest_verdict": _wrap(
            "honest_verdict",
            "complete: strict artifact schema/receipt normalizer ready for gated "
            "consumers; safe repairs are shape-only and missing evidence remains blocked.",
        ),
        "inference_substrate": _wrap("inference_substrate", INFERENCE_SUBSTRATE),
        "artifact_normalizer_ready": True,
        "artifact_normalizer_ready_principle": FIELD_PRINCIPLES["artifact_normalizer_ready"],
        "safe_repairs_supported": _wrap("safe_repairs_supported", list(SAFE_REPAIRS_SUPPORTED)),
        "unsafe_repairs_rejected": _wrap(
            "unsafe_repairs_rejected", list(UNSAFE_REPAIRS_REJECTED)
        ),
        "duration_policy_preserved": _wrap("duration_policy_preserved", True),
        "conductor_modified": _wrap("conductor_modified", False),
        "tests_run": [dict(row) for row in tests_run],
        "reproducibility_checksum": "",
    }
    artifact["reproducibility_checksum"] = _checksum(artifact)
    validate_artifact(artifact)
    return artifact


def _wrapped_value(artifact: Mapping[str, Any], field: str) -> Any:
    value = artifact.get(field)
    if not _is_wrapper(value):
        raise ValueError(f"{field} must be principle-wrapped")
    if value.get("principle") != FIELD_PRINCIPLES[field]:
        raise ValueError(f"{field} principle mismatch")
    return value.get("value")


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Validate the Exp 5247 receipt shape and honesty gates."""

    missing = [field for field in REQUIRED_SCHEMA_FIELDS if field not in artifact]
    if missing:
        raise ValueError(f"missing required fields: {missing}")
    if artifact.get("field_principles") != FIELD_PRINCIPLES:
        raise ValueError("field_principles mismatch")
    verdict = _wrapped_value(artifact, "honest_verdict")
    if not isinstance(verdict, str) or not verdict.startswith(TERMINAL_PREFIXES):
        raise ValueError("honest_verdict must have terminal prefix")
    if "ready for gated consumers" not in verdict:
        raise ValueError("honest_verdict must state gated-consumer readiness")
    if _wrapped_value(artifact, "inference_substrate") != INFERENCE_SUBSTRATE:
        raise ValueError("inference_substrate must be cached_fixture_replay_no_llm")
    if artifact.get("artifact_normalizer_ready") is not True:
        raise ValueError("artifact_normalizer_ready must be bare true")
    if artifact.get("artifact_normalizer_ready_principle") != FIELD_PRINCIPLES[
        "artifact_normalizer_ready"
    ]:
        raise ValueError("artifact_normalizer_ready_principle mismatch")
    if _wrapped_value(artifact, "duration_policy_preserved") is not True:
        raise ValueError("duration_policy_preserved must be true")
    if _wrapped_value(artifact, "conductor_modified") is not False:
        raise ValueError("conductor_modified must be false")
    for field in ("safe_repairs_supported", "unsafe_repairs_rejected"):
        values = _wrapped_value(artifact, field)
        if not isinstance(values, list) or not values:
            raise ValueError(f"{field} must be a non-empty list")
    tests_run = artifact.get("tests_run")
    if not isinstance(tests_run, list) or not tests_run:
        raise ValueError("tests_run must be a non-empty list")
    for row in tests_run:
        if not isinstance(row, Mapping) or not row.get("command") or not row.get("outcome"):
            raise ValueError("tests_run rows require command and outcome")
    if artifact.get("reproducibility_checksum") != _checksum(artifact):
        raise ValueError("reproducibility_checksum mismatch")


def write_artifact(
    *,
    output_path: Path | str = REPO_ROOT / RESULT_RELATIVE_PATH,
    representative_paths: Sequence[Path | str] = tuple(
        REPO_ROOT / path for path in REPRESENTATIVE_RELATIVE_PATHS
    ),
    tests_run: Sequence[Mapping[str, Any]],
    duration_s: float,
) -> JsonDict:
    """Write the Exp 5247 JSON receipt and return the artifact."""

    classifications = classify_representative_artifacts(representative_paths)
    artifact = build_artifact(
        representative_classifications=classifications,
        tests_run=tests_run,
        duration_s=duration_s,
    )
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return artifact


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - CLI wrapper.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=REPO_ROOT / RESULT_RELATIVE_PATH)
    parser.add_argument("--test-run", action="append", default=[])
    args = parser.parse_args(argv)
    started = time.monotonic()
    tests_run = [
        {"command": item.split("=", 1)[0], "outcome": item.split("=", 1)[1]}
        if "=" in item
        else {"command": item, "outcome": "RECORDED"}
        for item in args.test_run
    ] or [{"command": "not provided", "outcome": "RECORDED"}]
    artifact = write_artifact(
        output_path=args.output,
        tests_run=tests_run,
        duration_s=time.monotonic() - started,
    )
    print(json.dumps({"result_path": str(args.output), "checksum": artifact["reproducibility_checksum"]}))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
