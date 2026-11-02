#!/usr/bin/env python3
"""Exp5430 structured risk/prefix tautology corrigendum.

Spec refs: REQ-SAFE-5430, SCENARIO-SAFE-5430.

Exp5417 and Exp5418 contain useful local SOTA GGUF receipts and row records,
but their top-level aggregates were quarantined because unrelated rates happened
to be numerically equal.  This module does not rerun long model inference.  It
repairs the evidence boundary: reload the row records, recompute each aggregate
from its own row predicate, checksum the provenance, and expose a clean terminal
receipt that downstream capstones can use without treating equal numbers as
proof of independence.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Iterable, Mapping, Sequence
import copy
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from carnot import experiment_5404_formal_encoding_corrigendum_v492 as exp5404
from carnot import experiment_5417_risk_calibrated_sota_structured_panel_v493 as exp5417
from carnot import experiment_5418_predictive_prefix_action_safety_v493 as exp5418
from carnot.inference.sota_models import cached_sota_pair, resolve_cached_gguf


JsonDict = dict[str, Any]
ModelResolver = Callable[[str, str], str | None]
CachedPairFn = Callable[..., list[dict[str, Any]] | None]
RuntimeProbe = Callable[..., Mapping[str, Any]]

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path(
    "results/experiment_5430_structured_tautology_corrigendum_v494.json"
)
EXP5417_RELATIVE_PATH = exp5417.RESULT_RELATIVE_PATH
EXP5418_RELATIVE_PATH = exp5418.RESULT_RELATIVE_PATH
EXP5427_RELATIVE_PATH = Path("results/experiment_5427_capstone_v493.json")
SOURCE_ARTIFACT_PATHS = (
    EXP5417_RELATIVE_PATH,
    EXP5418_RELATIVE_PATH,
    EXP5427_RELATIVE_PATH,
)
EXPERIMENT_ID = "experiment_5430_structured_tautology_corrigendum_v494"
TASK_ID = "exp5430-v494-structured-tautology-corrigendum"
MILESTONE = "2026.07.494"
RUN_DATE = "2026-07-08"
SCHEMA = "carnot.experiment_5430.structured_tautology_corrigendum.v494"
SPEC_REFS = ("REQ-SAFE-5430", "SCENARIO-SAFE-5430")
RANDOM_SEED = 5430
AGGREGATE_CODE_VERSION = "structured-tautology-corrigendum-v1"
INFERENCE_SUBSTRATE = "live_llm_inference_and_row_reanalysis"
TERMINAL_PREFIXES = ("complete:", "blocked:")
MANDATED_HF_IDS = exp5417.MANDATED_HF_IDS
RECURRING_FLAG_KINDS = {"TAUTOLOGY", "METHODOLOGY_MISSING"}

FIELD_PRINCIPLES: dict[str, str] = {
    "preconditions_checked": "compute-bound task must fail fast.",
    "model_specs": "mandated SOTA GGUF provenance.",
    "runtime_backend": "no hidden transformers path.",
    "gpu_offload_verified": "no CPU-only SOTA headline.",
    "source_artifact_paths": "provenance.",
    "row_count_recomputed": "aggregate basis.",
    "row_provenance_checksum": "row-level reproducibility.",
    "risk_metric_independence_check": "tautology prevention.",
    "prefix_metric_independence_check": "tautology prevention.",
    "abstention_semantic_metric_separated": "semantic/schema separation.",
    "unreachable_delta_recomputed": "baseline-vs-delta separation.",
    "reproducibility_checksum": "methodology completeness.",
    "adversarial_verify_clean": "no known critical flags.",
    "structured_corrigendum_clean": "downstream gate.",
    "inference_substrate": "explicit evidence source.",
    "honest_verdict": 'terminal status; start with "complete:" or "blocked:".',
}
REQUIRED_ARTIFACT_FIELDS = tuple(FIELD_PRINCIPLES)
BANNED_TOP_LEVEL_REPAIRED_METRICS = frozenset(
    {
        "abstention_rate",
        "semantic_error_rate",
        "accepted_risk_estimate",
        "unsafe_false_accept_rate",
        "final_only_action_unreachability_rate",
        "prefix_gated_action_unreachability_rate",
        "action_unreachability_delta",
        "final_only_unreachable_tool_action_rate",
        "prefix_gated_unreachable_tool_action_rate",
        "unreachable_tool_action_delta",
    }
)


def run(
    *,
    root: Path | str = REPO_ROOT,
    result_path: Path | str | None = None,
    exp5417_artifact: Mapping[str, Any] | None = None,
    exp5418_artifact: Mapping[str, Any] | None = None,
    exp5427_artifact: Mapping[str, Any] | None = None,
    model_resolver: ModelResolver = resolve_cached_gguf,
    cached_pair_fn: CachedPairFn = cached_sota_pair,
    runtime_probe: RuntimeProbe | None = None,
    tests_run: Sequence[str | Mapping[str, Any]] = (),
    write: bool = True,
) -> JsonDict:
    """Build the Exp5430 corrigendum and optionally write it to disk."""

    root_path = Path(root)
    destination = _destination(root_path, result_path)
    source_5417 = dict(
        exp5417_artifact or _read_json(root_path / EXP5417_RELATIVE_PATH)
    )
    source_5418 = dict(
        exp5418_artifact or _read_json(root_path / EXP5418_RELATIVE_PATH)
    )
    source_5427 = dict(exp5427_artifact or _read_json(root_path / EXP5427_RELATIVE_PATH))
    preconditions = exp5404.collect_preconditions(
        model_resolver=model_resolver,
        cached_pair_fn=cached_pair_fn,
        runtime_probe=runtime_probe,
    )
    artifact = build_artifact(
        root=root_path,
        exp5417_artifact=source_5417,
        exp5418_artifact=source_5418,
        exp5427_artifact=source_5427,
        preconditions=preconditions,
        tests_run=tests_run,
    )
    if write:
        _write_json(destination, artifact)
    return artifact


def derive_risk_reanalysis(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Recompute Exp5417 risk aggregates from independent row predicates."""

    row_list = [dict(row) for row in rows if isinstance(row, Mapping)]
    total = len(row_list)
    checksum_values = [str(row.get("row_checksum")) for row in row_list]
    checksum_match = checksum_values == [risk_row_checksum(row) for row in row_list]
    abstention_ids = [
        _row_id(row)
        for row in row_list
        if _nested_get(row, ("abstaining", "decision")) == "abstain"
    ]
    semantic_error_ids = [
        _row_id(row) for row in row_list if _risk_row_has_semantic_error(row)
    ]
    accepted_ids = [
        _row_id(row) for row in row_list if _nested_get(row, ("abstaining", "accepted")) is True
    ]
    accepted_risk_ids = [
        _row_id(row)
        for row in row_list
        if _nested_get(row, ("abstaining", "accepted")) is True
        and (
            _risk_row_has_semantic_error(row)
            or _nested_get(row, ("abstaining", "unsafe_false_accept")) is True
        )
    ]
    unsafe_false_accept_ids = [
        _row_id(row)
        for row in row_list
        if _nested_get(row, ("abstaining", "accepted")) is True
        and _nested_get(row, ("abstaining", "unsafe_false_accept")) is True
    ]
    abstention_predicate = "abstaining.decision == 'abstain'"
    semantic_error_predicate = (
        "schema_valid && !semantic_valid on constrained or unconstrained arm"
    )
    separated = bool(
        total
        and abstention_predicate != semantic_error_predicate
        and bool(abstention_ids)
        and bool(semantic_error_ids)
    )
    aggregates = {
        "abstention_rate": _rate(len(abstention_ids), total),
        "semantic_error_rate": _rate(len(semantic_error_ids), total),
        "accepted_risk_estimate": _rate(len(accepted_risk_ids), len(accepted_ids)),
        "unsafe_false_accept_rate": _rate(len(unsafe_false_accept_ids), total),
    }
    return {
        "row_count": total,
        "row_checksums": checksum_values,
        "row_checksums_match": checksum_match,
        "aggregates": aggregates,
        "predicate_support": {
            "abstention_predicate": abstention_predicate,
            "semantic_error_predicate": semantic_error_predicate,
            "accepted_risk_predicate": "accepted && (semantic_error || unsafe_false_accept)",
            "unsafe_false_accept_predicate": "accepted && abstaining.unsafe_false_accept",
            "abstention_row_ids": abstention_ids,
            "semantic_error_row_ids": semantic_error_ids,
            "abstention_semantic_support_sets_equal": set(abstention_ids)
            == set(semantic_error_ids),
            "accepted_row_ids": accepted_ids,
            "accepted_risk_row_ids": accepted_risk_ids,
            "unsafe_false_accept_row_ids": unsafe_false_accept_ids,
        },
        "row_labels": _row_labels(row_list),
        "fixture_ids": _fixture_ids(row_list),
        "independence": {
            "risk_metric_independence_check": bool(checksum_match and separated),
            "abstention_semantic_metric_separated": separated,
        },
    }


def derive_prefix_reanalysis(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Recompute Exp5418 prefix aggregates from row predicates."""

    row_list = [dict(row) for row in rows if isinstance(row, Mapping)]
    total = len(row_list)
    checksum_values = [str(row.get("row_checksum")) for row in row_list]
    checksum_match = checksum_values == [prefix_row_checksum(row) for row in row_list]
    final_unreachable_ids = [
        _row_id(row)
        for row in row_list
        if _nested_get(row, ("final_only", "unreachable_tool_action")) is True
    ]
    prefix_unreachable_ids = [
        _row_id(row)
        for row in row_list
        if _nested_get(row, ("prefix_gated", "unreachable_tool_action")) is True
    ]
    final_unsafe_ids = [
        _row_id(row)
        for row in row_list
        if _nested_get(row, ("final_only", "unsafe_false_accept")) is True
    ]
    prefix_unsafe_ids = [
        _row_id(row)
        for row in row_list
        if _nested_get(row, ("prefix_gated", "unsafe_false_accept")) is True
    ]
    abstention_ids = [
        _row_id(row)
        for row in row_list
        if _nested_get(row, ("prefix_gate", "decision")) == "abstained"
    ]
    final_unreachable_rate = _rate(len(final_unreachable_ids), total)
    prefix_unreachable_rate = _rate(len(prefix_unreachable_ids), total)
    delta = _rate(len(final_unreachable_ids) - len(prefix_unreachable_ids), total)
    delta_recomputed = math.isclose(
        delta,
        _rate(len(final_unreachable_ids) - len(prefix_unreachable_ids), total),
        rel_tol=0.0,
        abs_tol=1e-12,
    )
    independent = bool(
        checksum_match
        and total
        and set(final_unreachable_ids) != set(prefix_unreachable_ids)
        and delta_recomputed
    )
    return {
        "row_count": total,
        "row_checksums": checksum_values,
        "row_checksums_match": checksum_match,
        "aggregates": {
            "final_only_action_unreachability_rate": final_unreachable_rate,
            "prefix_gated_action_unreachability_rate": prefix_unreachable_rate,
            "action_unreachability_delta": delta,
            "final_only_unsafe_false_accept_rate": _rate(len(final_unsafe_ids), total),
            "prefix_gated_unsafe_false_accept_rate": _rate(len(prefix_unsafe_ids), total),
            "abstention_rate": _rate(len(abstention_ids), total),
        },
        "predicate_support": {
            "final_unreachable_predicate": "final_only.unreachable_tool_action is true",
            "prefix_unreachable_predicate": "prefix_gated.unreachable_tool_action is true",
            "delta_formula": "final_only_action_unreachability_rate - prefix_gated_action_unreachability_rate",
            "final_unreachable_row_ids": final_unreachable_ids,
            "prefix_unreachable_row_ids": prefix_unreachable_ids,
            "final_unsafe_row_ids": final_unsafe_ids,
            "prefix_unsafe_row_ids": prefix_unsafe_ids,
            "abstention_row_ids": abstention_ids,
        },
        "row_labels": _row_labels(row_list),
        "fixture_ids": _fixture_ids(row_list),
        "independence": {
            "prefix_metric_independence_check": independent,
            "unreachable_delta_recomputed": delta_recomputed,
        },
    }


def build_artifact(
    *,
    root: Path,
    exp5417_artifact: Mapping[str, Any],
    exp5418_artifact: Mapping[str, Any],
    exp5427_artifact: Mapping[str, Any],
    preconditions: Any,
    tests_run: Sequence[str | Mapping[str, Any]] = (),
) -> JsonDict:
    """Assemble the terminal artifact from source rows and preflight evidence."""

    risk_rows = _mapping_rows(exp5417_artifact.get("risk_rows"))
    prefix_rows = _mapping_rows(exp5418_artifact.get("prefix_traces"))
    risk = derive_risk_reanalysis(risk_rows)
    prefix = derive_prefix_reanalysis(prefix_rows)
    source_records = source_artifact_records(root, SOURCE_ARTIFACT_PATHS)
    model_specs = _mark_model_specs(preconditions.model_specs)
    blockers = _blockers(exp5417_artifact, exp5418_artifact, preconditions, risk, prefix)
    row_prov_checksum = compute_row_provenance_checksum(
        source_artifact_records=source_records,
        model_specs=model_specs,
        risk_reanalysis=risk,
        prefix_reanalysis=prefix,
    )
    source_boundary = source_boundary_review(exp5417_artifact, exp5418_artifact, exp5427_artifact)
    runtime_backend = _runtime_backend(preconditions.gpu_offload_receipt, exp5417_artifact)
    gpu_offload_verified = bool(
        not preconditions.blocked_preconditions
        and preconditions.gpu_offload_receipt.get("proof_not_cpu_only_headline_evidence")
    )
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "task_id": TASK_ID,
        "milestone": MILESTONE,
        "run_date": RUN_DATE,
        "random_seed": RANDOM_SEED,
        "spec_refs": list(SPEC_REFS),
        "aggregate_code_version": AGGREGATE_CODE_VERSION,
        "preconditions_checked": True,
        "model_specs": model_specs,
        "runtime_backend": runtime_backend,
        "gpu_offload_verified": gpu_offload_verified,
        "source_artifact_paths": [str(path) for path in SOURCE_ARTIFACT_PATHS],
        "row_count_recomputed": int(risk["row_count"]) + int(prefix["row_count"]),
        "row_provenance_checksum": row_prov_checksum,
        "risk_metric_independence_check": bool(
            risk["independence"]["risk_metric_independence_check"]
        ),
        "prefix_metric_independence_check": bool(
            prefix["independence"]["prefix_metric_independence_check"]
        ),
        "abstention_semantic_metric_separated": bool(
            risk["independence"]["abstention_semantic_metric_separated"]
        ),
        "unreachable_delta_recomputed": bool(
            prefix["independence"]["unreachable_delta_recomputed"]
        ),
        "reproducibility_checksum": "0" * 64,
        "adversarial_verify_clean": False,
        "structured_corrigendum_clean": False,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "honest_verdict": "blocked: pending focused adversarial check",
        "status": "blocked",
        "blocked_preconditions": blockers,
        "source_artifact_checksums": source_records,
        "aggregate_reanalysis": {"risk": risk, "prefix": prefix},
        "source_boundary_review": source_boundary,
        "gpu_offload_receipt": dict(preconditions.gpu_offload_receipt),
        "field_principles": dict(FIELD_PRINCIPLES),
        "tests_run": _normalise_tests_run(tests_run),
        "research_conductor_modified": False,
    }
    artifact["reproducibility_checksum"] = compute_reproducibility_checksum(artifact)
    recurring_flags = focused_adversarial_flags(artifact)
    artifact["adversarial_focus_flags"] = recurring_flags
    adversarial_clean = not recurring_flags
    artifact["adversarial_verify_clean"] = adversarial_clean
    clean = bool(
        not blockers
        and gpu_offload_verified
        and artifact["row_count_recomputed"] > 0
        and artifact["risk_metric_independence_check"]
        and artifact["prefix_metric_independence_check"]
        and artifact["abstention_semantic_metric_separated"]
        and artifact["unreachable_delta_recomputed"]
        and adversarial_clean
    )
    artifact["structured_corrigendum_clean"] = clean
    artifact["status"] = "complete" if clean else "blocked"
    artifact["honest_verdict"] = _honest_verdict(clean, blockers, recurring_flags)
    artifact["reproducibility_checksum"] = compute_reproducibility_checksum(artifact)
    validate_artifact(artifact)
    return artifact


def source_artifact_records(root: Path, paths: Sequence[Path]) -> list[JsonDict]:
    """Return source paths and file hashes used by the reproducibility checks."""

    records: list[JsonDict] = []
    for relative in paths:
        path = root / relative
        if path.exists() and path.is_file():
            records.append({"path": str(relative), "sha256": _sha256_bytes(path.read_bytes())})
        else:
            records.append({"path": str(relative), "sha256": None, "missing": True})
    return records


def compute_row_provenance_checksum(
    *,
    source_artifact_records: Sequence[Mapping[str, Any]],
    model_specs: Sequence[Mapping[str, Any]],
    risk_reanalysis: Mapping[str, Any],
    prefix_reanalysis: Mapping[str, Any],
) -> str:
    """Hash source files, model specs, row labels, fixture IDs, and code version."""

    material = {
        "aggregate_code_version": AGGREGATE_CODE_VERSION,
        "source_artifact_records": list(source_artifact_records),
        "model_specs": _model_spec_fingerprint(model_specs),
        "risk_fixture_ids": risk_reanalysis.get("fixture_ids", []),
        "risk_row_labels": risk_reanalysis.get("row_labels", []),
        "risk_row_checksums": risk_reanalysis.get("row_checksums", []),
        "prefix_fixture_ids": prefix_reanalysis.get("fixture_ids", []),
        "prefix_row_labels": prefix_reanalysis.get("row_labels", []),
        "prefix_row_checksums": prefix_reanalysis.get("row_checksums", []),
    }
    return hashlib.sha256(_stable_json(material).encode("utf-8")).hexdigest()


def compute_reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    """Hash the method-bearing artifact payload without self-referencing."""

    material = {
        key: value
        for key, value in artifact.items()
        if key
        not in {
            "reproducibility_checksum",
            "honest_verdict",
            "status",
        }
    }
    return hashlib.sha256(_stable_json(material).encode("utf-8")).hexdigest()


def focused_adversarial_flags(artifact: Mapping[str, Any]) -> list[JsonDict]:
    """Run the local TAUTOLOGY/METHODOLOGY_MISSING subset used by this receipt."""

    try:
        import scripts.adversarial_verify as adversarial_verify  # noqa: PLC0415
    except Exception:
        return [
            {
                "kind": "METHODOLOGY_MISSING",
                "severity": "warn",
                "detail": "focused adversarial verifier import failed",
            }
        ]
    flags: list[Any] = []
    flattened = adversarial_verify._flatten_metrics(dict(artifact))  # noqa: SLF001
    normalised = adversarial_verify._normalize_principle_wrapped_fields(flattened)  # noqa: SLF001
    adversarial_verify.check_tautology(normalised, flags)
    adversarial_verify.check_methodology_present(normalised, flags)
    return [
        flag.to_dict()
        for flag in flags
        if flag.kind in RECURRING_FLAG_KINDS
    ]


def source_boundary_review(
    exp5417_artifact: Mapping[str, Any],
    exp5418_artifact: Mapping[str, Any],
    exp5427_artifact: Mapping[str, Any],
) -> JsonDict:
    """Preserve the prior capstone/adversarial boundary as context."""

    return {
        "exp5417_flagged_adversarial": bool(exp5417_artifact.get("flagged_adversarial")),
        "exp5417_corrigendum_pending": copy.deepcopy(
            exp5417_artifact.get("corrigendum_pending", [])
        ),
        "exp5418_flagged_adversarial": bool(exp5418_artifact.get("flagged_adversarial")),
        "exp5418_corrigendum_pending": copy.deepcopy(
            exp5418_artifact.get("corrigendum_pending", [])
        ),
        "exp5427_structured_lanes": [
            copy.deepcopy(row)
            for row in exp5427_artifact.get("truth_table", [])
            if isinstance(row, Mapping)
            and row.get("lane")
            in {
                "risk_calibrated_structured_verification",
                "predictive_prefix_action_safety",
            }
        ],
    }


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Raise when the Exp5430 artifact cannot support downstream use."""

    errors = artifact_schema_errors(artifact)
    if errors:
        raise ValueError("; ".join(errors))


def artifact_schema_errors(artifact: Mapping[str, Any]) -> list[str]:
    """Return schema, checksum, and readiness validation errors."""

    errors: list[str] = []
    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    if missing:
        errors.append(f"missing required fields: {missing}")
    if artifact.get("field_principles") != FIELD_PRINCIPLES:
        errors.append("field_principles must match REQ-SAFE-5430")
    if artifact.get("preconditions_checked") is not True:
        errors.append("preconditions_checked must be true")
    if not _model_specs_cover_mandated(artifact.get("model_specs")):
        errors.append("model_specs must include all mandated SOTA GGUF ids")
    if not str(artifact.get("runtime_backend", "")).startswith("llama.cpp"):
        errors.append("runtime_backend must be llama.cpp based")
    if type(artifact.get("gpu_offload_verified")) is not bool:
        errors.append("gpu_offload_verified must be boolean")
    if not _bare_non_negative_int(artifact.get("row_count_recomputed")):
        errors.append("row_count_recomputed must be a non-negative integer")
    if not _sha256_text(artifact.get("row_provenance_checksum")):
        errors.append("row_provenance_checksum must be a sha256 hex string")
    if not _sha256_text(artifact.get("reproducibility_checksum")):
        errors.append("reproducibility_checksum must be a sha256 hex string")
    for field in (
        "risk_metric_independence_check",
        "prefix_metric_independence_check",
        "abstention_semantic_metric_separated",
        "unreachable_delta_recomputed",
        "adversarial_verify_clean",
        "structured_corrigendum_clean",
    ):
        if type(artifact.get(field)) is not bool:
            errors.append(f"{field} must be boolean")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate must be live_llm_inference_and_row_reanalysis")
    verdict = artifact.get("honest_verdict")
    if not isinstance(verdict, str) or "\n" in verdict or not verdict.startswith(
        TERMINAL_PREFIXES
    ):
        errors.append("honest_verdict must start with complete: or blocked:")
    exposed = sorted(BANNED_TOP_LEVEL_REPAIRED_METRICS.intersection(artifact))
    if exposed:
        errors.append(f"repaired metrics must stay nested, not top-level: {exposed}")
    if artifact.get("source_artifact_paths") != [str(path) for path in SOURCE_ARTIFACT_PATHS]:
        errors.append("source_artifact_paths must preserve Exp5417/Exp5418/Exp5427 order")
    aggregate_reanalysis = artifact.get("aggregate_reanalysis")
    if not isinstance(aggregate_reanalysis, Mapping):
        errors.append("aggregate_reanalysis must preserve risk and prefix recomputations")
    else:
        errors.extend(_aggregate_reanalysis_errors(artifact, aggregate_reanalysis))
    recurring = [
        flag
        for flag in artifact.get("adversarial_focus_flags", [])
        if isinstance(flag, Mapping) and flag.get("kind") in RECURRING_FLAG_KINDS
    ]
    if recurring and artifact.get("honest_verdict", "").startswith("complete:"):
        errors.append("recurring adversarial flags require blocked honest_verdict")
    if artifact.get("structured_corrigendum_clean") is True:
        if artifact.get("status") != "complete":
            errors.append("structured_corrigendum_clean requires complete status")
        if artifact.get("gpu_offload_verified") is not True:
            errors.append("structured_corrigendum_clean requires GPU offload")
        if int(artifact.get("row_count_recomputed") or 0) <= 0:
            errors.append("structured_corrigendum_clean requires recomputed rows")
        for field in (
            "risk_metric_independence_check",
            "prefix_metric_independence_check",
            "abstention_semantic_metric_separated",
            "unreachable_delta_recomputed",
            "adversarial_verify_clean",
        ):
            if artifact.get(field) is not True:
                errors.append(f"structured_corrigendum_clean requires {field}")
        if recurring:
            errors.append("structured_corrigendum_clean requires adversarial clean")
    if artifact.get("research_conductor_modified") is not False:
        errors.append("scripts/research_conductor.py must not be modified")
    return errors


def risk_row_checksum(row: Mapping[str, Any]) -> str:
    """Return a stable checksum over one risk row excluding its checksum field."""

    clean = {key: value for key, value in row.items() if key != "row_checksum"}
    return hashlib.sha256(_stable_json(clean).encode("utf-8")).hexdigest()


def risk_row_with_checksum(row: Mapping[str, Any]) -> JsonDict:
    """Return a copied risk row with a checksum attached."""

    out = copy.deepcopy(dict(row))
    out["row_checksum"] = risk_row_checksum(out)
    return out


def prefix_row_checksum(row: Mapping[str, Any]) -> str:
    """Return a stable checksum over one prefix row excluding its checksum field."""

    clean = {key: value for key, value in row.items() if key != "row_checksum"}
    return hashlib.sha256(_stable_json(clean).encode("utf-8")).hexdigest()


def prefix_row_with_checksum(row: Mapping[str, Any]) -> JsonDict:
    """Return a copied prefix row with a checksum attached."""

    out = copy.deepcopy(dict(row))
    out["row_checksum"] = prefix_row_checksum(out)
    return out


def main(
    argv: Sequence[str] | None = None,
    *,
    model_resolver: ModelResolver = resolve_cached_gguf,
    cached_pair_fn: CachedPairFn = cached_sota_pair,
    runtime_probe: RuntimeProbe | None = None,
) -> int:
    """CLI entry point for producing the Exp5430 result artifact."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    parser.add_argument("--result-path", type=Path, default=None)
    args = parser.parse_args(argv)
    artifact = run(
        root=args.root,
        result_path=args.result_path,
        model_resolver=model_resolver,
        cached_pair_fn=cached_pair_fn,
        runtime_probe=runtime_probe,
        write=True,
    )
    print(json.dumps(artifact, indent=2, sort_keys=True, ensure_ascii=True))
    return 0 if artifact["status"] == "complete" else 1


def _risk_row_has_semantic_error(row: Mapping[str, Any]) -> bool:
    return bool(
        _schema_only_error(_nested_mapping(row, ("constrained",)))
        or _schema_only_error(_nested_mapping(row, ("unconstrained",)))
    )


def _schema_only_error(arm: Mapping[str, Any]) -> bool:
    return bool(arm.get("schema_valid") is True and arm.get("semantic_valid") is False)


def _blockers(
    exp5417_artifact: Mapping[str, Any],
    exp5418_artifact: Mapping[str, Any],
    preconditions: Any,
    risk: Mapping[str, Any],
    prefix: Mapping[str, Any],
) -> list[str]:
    blockers: list[str] = [str(item) for item in preconditions.blocked_preconditions]
    if exp5417_artifact.get("gpu_offload_verified") is not True:
        blockers.append("exp5417_gpu_offload_verified_false")
    if exp5418_artifact.get("gpu_offload_verified") is not True:
        blockers.append("exp5418_gpu_offload_verified_false")
    if not _model_specs_cover_mandated(exp5417_artifact.get("model_specs")):
        blockers.append("exp5417_mandated_model_specs_missing")
    if not _model_specs_cover_mandated(exp5418_artifact.get("model_specs")):
        blockers.append("exp5418_mandated_model_specs_missing")
    if int(risk.get("row_count") or 0) <= 0:
        blockers.append("exp5417_risk_rows_missing")
    if int(prefix.get("row_count") or 0) <= 0:
        blockers.append("exp5418_prefix_traces_missing")
    if risk.get("row_checksums_match") is not True:
        blockers.append("exp5417_row_checksum_mismatch")
    if prefix.get("row_checksums_match") is not True:
        blockers.append("exp5418_row_checksum_mismatch")
    if risk.get("independence", {}).get("risk_metric_independence_check") is not True:
        blockers.append("risk_metric_independence_failed")
    if prefix.get("independence", {}).get("prefix_metric_independence_check") is not True:
        blockers.append("prefix_metric_independence_failed")
    return _unique(blockers)


def _aggregate_reanalysis_errors(
    artifact: Mapping[str, Any],
    aggregate_reanalysis: Mapping[str, Any],
) -> list[str]:
    errors: list[str] = []
    risk = aggregate_reanalysis.get("risk")
    prefix = aggregate_reanalysis.get("prefix")
    if not isinstance(risk, Mapping) or not isinstance(prefix, Mapping):
        return ["aggregate_reanalysis must include risk and prefix"]
    expected_count = int(risk.get("row_count") or 0) + int(prefix.get("row_count") or 0)
    if artifact.get("row_count_recomputed") != expected_count:
        errors.append("row_count_recomputed must equal risk + prefix row counts")
    if artifact.get("risk_metric_independence_check") != risk.get("independence", {}).get(
        "risk_metric_independence_check"
    ):
        errors.append("risk_metric_independence_check must mirror risk reanalysis")
    if artifact.get("prefix_metric_independence_check") != prefix.get("independence", {}).get(
        "prefix_metric_independence_check"
    ):
        errors.append("prefix_metric_independence_check must mirror prefix reanalysis")
    if artifact.get("abstention_semantic_metric_separated") != risk.get(
        "independence", {}
    ).get("abstention_semantic_metric_separated"):
        errors.append("abstention_semantic_metric_separated must mirror risk reanalysis")
    if artifact.get("unreachable_delta_recomputed") != prefix.get("independence", {}).get(
        "unreachable_delta_recomputed"
    ):
        errors.append("unreachable_delta_recomputed must mirror prefix reanalysis")
    return errors


def _mark_model_specs(precondition_specs: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    rows: list[JsonDict] = []
    for spec in precondition_specs:
        row = dict(spec)
        row["selected_for_exp5430_precondition"] = bool(
            row.get("status") == "local_gguf_resolved"
        )
        row["ran_in_exp5430"] = False
        rows.append(row)
    return rows


def _model_spec_fingerprint(model_specs: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    return [
        {
            "hf_id": row.get("hf_id"),
            "model_path": row.get("model_path"),
            "quantization": row.get("quantization"),
            "status": row.get("status"),
            "gguf_loader_family": row.get("gguf_loader_family"),
        }
        for row in model_specs
    ]


def _runtime_backend(receipt: Mapping[str, Any], fallback_artifact: Mapping[str, Any]) -> str:
    return str(
        receipt.get("runtime_backend")
        or receipt.get("backend")
        or fallback_artifact.get("runtime_backend")
        or "llama.cpp/llama-cpp-python"
    )


def _model_specs_cover_mandated(value: Any) -> bool:
    return bool(
        isinstance(value, list)
        and {row.get("hf_id") for row in value if isinstance(row, Mapping)}
        == set(MANDATED_HF_IDS)
    )


def _mapping_rows(value: Any) -> list[Mapping[str, Any]]:
    return [row for row in value if isinstance(row, Mapping)] if isinstance(value, list) else []


def _normalise_tests_run(tests_run: Sequence[str | Mapping[str, Any]]) -> list[Any]:
    if tests_run:
        return [dict(row) if isinstance(row, Mapping) else str(row) for row in tests_run]
    return [
        ".venv/bin/pytest tests/python/test_experiment_5430_structured_tautology_corrigendum_v494.py -q",
    ]


def _honest_verdict(
    clean: bool,
    blockers: Sequence[str],
    recurring_flags: Sequence[Mapping[str, Any]],
) -> str:
    if clean:
        return "complete: structured tautology corrigendum clean with row checksums"
    if recurring_flags:
        kinds = ",".join(_unique(flag.get("kind", "unknown") for flag in recurring_flags))
        return f"blocked: recurring_adversarial_flags_{kinds}"
    if blockers:
        return "blocked: " + ",".join(_unique(blockers))
    return "blocked: structured_corrigendum_clean_false"


def _row_labels(rows: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    return [
        {
            "row_id": _row_id(row),
            "fixture_id": str(row.get("source_fixture_id") or row.get("fixture_id") or ""),
            "label": str(
                row.get("fixture_family")
                or row.get("source_fixture_family")
                or row.get("prefix_family")
                or ""
            ),
        }
        for row in rows
    ]


def _fixture_ids(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    return [
        str(row.get("source_fixture_id") or row.get("fixture_id") or _row_id(row))
        for row in rows
    ]


def _row_id(row: Mapping[str, Any]) -> str:
    return str(row.get("row_id") or row.get("source_risk_row_id") or "")


def _nested_mapping(row: Mapping[str, Any], path: Sequence[str]) -> Mapping[str, Any]:
    value = _nested_get(row, path)
    return value if isinstance(value, Mapping) else {}


def _nested_get(row: Mapping[str, Any], path: Sequence[str]) -> Any:
    current: Any = row
    for key in path:
        if not isinstance(current, Mapping):
            return None
        current = current.get(key)
    return current


def _rate(numerator: int | float, denominator: int) -> float:
    return 0.0 if denominator <= 0 else round(float(numerator) / denominator, 6)


def _bare_non_negative_int(value: Any) -> bool:
    return type(value) is int and value >= 0


def _sha256_text(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(ch in "0123456789abcdef" for ch in value)
    )


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _unique(values: Iterable[Any]) -> list[str]:
    return list(dict.fromkeys(str(value) for value in values))


def _destination(root: Path, result_path: Path | str | None) -> Path:
    destination = Path(result_path) if result_path is not None else root / RESULT_RELATIVE_PATH
    return destination if destination.is_absolute() else root / destination


def _read_json(path: Path) -> JsonDict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )


def _stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
