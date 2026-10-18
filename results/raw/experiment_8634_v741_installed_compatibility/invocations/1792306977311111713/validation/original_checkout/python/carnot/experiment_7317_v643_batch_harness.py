"""Qualify the unchanged batch fixture through explicit validation scopes.

The V642 fixture mechanics were valid, but its launcher made an unrelated
repository-wide test run a required check. This module keeps those mechanics
and sends only named files to the shipped scoped runner. Historical failures
remain visible as health evidence and cannot authorize this result.

Spec refs: REQ-VERIFY-7317 and SCENARIO-VERIFY-7317-*.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping, Sequence
from copy import deepcopy
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
import tempfile
import time
from typing import Any

import yaml

from carnot import experiment_7306_v642_batch_fixture as fixture
from carnot.experiment_artifacts import atomic_write_json
from carnot.paths import repo_root as find_repo_root
from carnot.reporting import experiment_7303_validation_scope as scoped


JsonDict = dict[str, Any]
RUN_DATE = "20260915"
MILESTONE = "2026.09.643"
EXPERIMENT_ID = "exp7317-batch-harness"
SCHEMA = "carnot.exp7317.v643_batch_harness.v1"
SCOPED_RUNNER = "carnot.reporting.experiment_7303_validation_scope.run_scoped_validation"
MODEL_SPECS: list[JsonDict] = []
INFERENCE_SUBSTRATE = "cpu_exact_solver_or_simulator"
INFERENCE_SUBSTRATE_CLASS = "cpu_exact_solver_or_simulator"
EXECUTION_VENUE = "host"
ZERO_INVOCATION_COUNTS = deepcopy(fixture.ZERO_INVOCATION_COUNTS)

RESULT_PATH = Path("results/experiment_7317_v643_batch_harness.json")
RAW_DIR = Path("results/raw/experiment_7317_v643_batch_harness")
CANDIDATE_PATH = RAW_DIR / "terminal_candidate.json"
PUBLIC_PATH = RAW_DIR / "public_panel.json"
LABEL_PATH = RAW_DIR / "evaluator_labels.json"
PREDICTION_PATH = RAW_DIR / "public_predictions.json"
ROW_PATH = RAW_DIR / "scored_rows.json"
CALL_PATH = RAW_DIR / "call_rows.json"
PAYLOAD_PATH = RAW_DIR / "injected_outputs.json"
COST_PATH = RAW_DIR / "injected_costs.json"
CONTROL_PATH = RAW_DIR / "batch_controls.json"
REDUCTION_PATH = RAW_DIR / "independent_reduction.json"

V642_PUBLIC_PATH = Path("results/raw/experiment_7306_v642_batch_fixture/public_panel.json")
V642_LABEL_PATH = Path("results/raw/experiment_7306_v642_batch_fixture/evaluator_labels.json")
HISTORICAL_ARTIFACTS = (
    Path("results/experiment_7306_v642_batch_fixture.json"),
    Path("results/experiment_7307_v642_batch_canary.json"),
)
HEALTH_SOURCE = Path("results/experiment_7303_v642_validation_scope.json")

REQUIRED_CURRENT_PATHS = {
    "scoped_runner": Path("python/carnot/reporting/experiment_7303_validation_scope.py"),
    "fixture_module": Path("python/carnot/experiment_7306_v642_batch_fixture.py"),
    "module": Path("python/carnot/experiment_7317_v643_batch_harness.py"),
    "entrypoint": Path("scripts/experiments/experiment_7317_v643_batch_harness.py"),
    "focused_tests": Path("tests/python/test_experiment_7317_v643_batch_harness.py"),
    "verification_spec": Path("openspec/capabilities/verification/spec.md"),
    "exclusion_manifest": Path("ops/exclusion_manifest.yaml"),
    "adversarial_verifier": Path("scripts/adversarial_verify.py"),
    "row_consistency_lint": Path("scripts/verdict_row_consistency_lint.py"),
}

TERMINAL_CHECK_NAMES = (
    "candidate_reload_and_independent_reduce",
    "adversarial_verify",
    "verdict_row_consistency_strict",
)

FIELD_PRINCIPLES: JsonDict = {
    "schema": "Version this artifact and keep ordinary experiment and milestone fields.",
    "status": "Publish a terminal value only after current work and required validation.",
    "run_date": "Use the declared date and preserve actual UTC and monotonic timing.",
    "preconditions_checked": "Record each input identity, availability result, and exact failure.",
    "MODEL_SPECS": "List current executable models only; this CPU harness uses none.",
    "model_invoked": "Record every attempted model load or generation, including failures.",
    "invocation_counts": "Keep attempted, completed, failed, cancelled, and active work separate.",
    "inference_substrate": "Name the actual computation with a recognized substrate value.",
    "inference_substrate_class": "Select the duration class from work that actually ran.",
    "execution_venue": "This CPU exact fixture runs on the host.",
    "duration_s": "Measure real elapsed time without sleeps or padding.",
    "phase_spans": "Preserve disjoint work spans, units, checkpoints, and pending operations.",
    "random_seed": "Seal independent development, evaluation, and bootstrap orders.",
    "reproducibility_checksum": "Bind code, public input, labels, settings, and raw evidence.",
    "source_artifact_hashes": "Authenticate exact producers without treating history as a gate.",
    "rows": "Keep every arm and unit with metrics, costs, errors, abstentions, and censoring.",
    "sample_size_budget": "Record planned and observed denominators with the fixed stopping rule.",
    "acceptance_gate_results": "Retain expected, observed, passed, and purpose for every check.",
    "gate_check_summary": "Preserve the first exact failed check and both compared values.",
    "verifier_is_oracle": "Shared exact authority forbids a positive scientific class.",
    "honest_verdict": "Name the actual terminal outcome with a recognized prefix.",
    "verdict_class": "Use the closed terminal class set without hiding unfinished work.",
    "validation_receipts": "Keep commands, scopes, exits, elapsed time, and exact log hashes.",
    "repository_health": "Keep unrelated dated failures as observations, not required checks.",
    "field_principles": "Explain why each artifact field exists without wrapping its value.",
    "batch_harness_ready_score": "One requires current mechanics and every scoped check.",
    "validation_entrypoint_receipt": "Prove the new runner received explicit files.",
    "sealed_panel_manifest": "Separate public, label, selection, order, and cost hashes.",
    "call_budget_contract": "Keep all three arms at 1,280 tokens per source version.",
    "acceptance_contract": "Freeze the V642 semantic, safety, coverage, and cost bounds.",
    "batch_control_rows": "Use injected attacks without claiming model performance.",
}
REQUIRED_ARTIFACT_FIELDS = frozenset(
    {
        *FIELD_PRINCIPLES,
        "experiment_id",
        "milestone",
        "timestamps",
        "authority_separation",
        "independent_reduction",
        "diagnostic_evidence",
        "methodology_note",
    }
)


def _utc_now() -> str:
    """Record a real UTC boundary without using wall time as scientific evidence."""

    return datetime.now(UTC).isoformat()


def _canonical_json(value: Any) -> str:
    """Use one JSON spelling for hashes and raw-evidence comparisons."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _sha256_bytes(value: bytes) -> str:
    """Label hashes so identifiers cannot be confused with raw values."""

    return "sha256:" + hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    """Hash exact file bytes without normalizing historical logs."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _progress(phase: str, event: str, started: float, detail: str = "") -> None:
    """Flush each boundary with truthful monotonic elapsed time."""

    suffix = f" {detail}" if detail else ""
    print(
        f"[exp7317] phase={phase} event={event} elapsed_s={time.monotonic() - started:.3f}{suffix}",
        flush=True,
    )


def gate_row(
    check: str,
    upstream: str,
    field: str,
    expected: Any,
    observed: Any,
    passed: bool,
    *,
    principle: str = "A failed prerequisite must stop current readiness.",
    readiness_gate: bool = True,
) -> JsonDict:
    """Keep the exact check and values that determine one gate."""

    return {
        "check": check,
        "upstream": upstream,
        "field": field,
        "expected_value": expected,
        "observed_value": observed,
        "passed": passed,
        "principle": principle,
        "readiness_gate": readiness_gate,
    }


def gate_check_summary(checks: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Project the first failed readiness check without changing its values."""

    failed = next(
        (
            row
            for row in checks
            if row.get("readiness_gate", True) and row.get("passed") is not True
        ),
        None,
    )
    if failed is None:
        return {
            "failed_check": None,
            "upstream": None,
            "field": None,
            "expected_value": None,
            "observed_value": None,
        }
    return {
        "failed_check": failed["check"],
        "upstream": failed["upstream"],
        "field": failed["field"],
        "expected_value": failed["expected_value"],
        "observed_value": failed["observed_value"],
    }


def validation_scope(kind: str) -> JsonDict:
    """Return explicit files for this harness or one planned consumer."""

    identities = {
        "harness": "experiment_7317_v643_batch_harness",
        "canary": "experiment_7320_v643_batch_canary",
        "capture": "experiment_7321_v643_batch_measurement",
    }
    if kind not in identities:
        raise ValueError(f"unsupported validation consumer: {kind}")
    identity = identities[kind]
    return {
        "consumer": kind,
        "runner": SCOPED_RUNNER,
        "test_paths": [f"tests/python/test_{identity}.py"],
        "changed_modules": [f"python/carnot/{identity}.py"],
        "static_paths": [f"scripts/experiments/{identity}.py"],
    }


def _scope_errors(target: Mapping[str, Any]) -> list[str]:
    """Reject directory or implicit scope before the shared runner starts."""

    errors = []
    if target.get("runner") != SCOPED_RUNNER:
        errors.append("runner")
    for field in ("test_paths", "changed_modules", "static_paths"):
        paths = target.get(field)
        if (
            not isinstance(paths, list)
            or not paths
            or any(not isinstance(path, str) or not path.endswith(".py") for path in paths)
        ):
            errors.append(field)
    if any(path.rstrip("/") == "tests/python" for path in target.get("test_paths", [])):
        errors.append("repository_wide_target")
    return list(dict.fromkeys(errors))


def run_scoped_checks(
    root: Path,
    target: Mapping[str, Any],
    *,
    historical_failures: Sequence[Mapping[str, Any]] = (),
    extra_env: Mapping[str, str] | None = None,
) -> JsonDict:
    """Call Exp7303 once with explicit files and retain the call contract."""

    errors = _scope_errors(target)
    if errors:
        raise ValueError("invalid scoped validation target: " + ",".join(errors))
    raw_dir = root / RAW_DIR
    basetemp = Path(tempfile.mkdtemp(prefix="carnot-exp7317-"))
    result = scoped.run_scoped_validation(
        root,
        list(target["test_paths"]),
        list(target["changed_modules"]),
        static_paths=list(target["static_paths"]),
        basetemp=basetemp,
        coverage_file=raw_dir / ".coverage",
        log_dir=raw_dir / "validation",
        historical_failures=historical_failures,
        extra_env=extra_env,
    )
    commands = [value for row in result["validation_receipts"] for value in row["command_argv"]]
    result["validation_entrypoint_receipt"] = {
        "runner": SCOPED_RUNNER,
        "called": True,
        "test_paths": list(target["test_paths"]),
        "changed_modules": list(target["changed_modules"]),
        "static_paths": list(target["static_paths"]),
        "legacy_launcher_called": False,
        "repository_wide_target_present": "tests/python" in commands,
    }
    if result["validation_entrypoint_receipt"]["repository_wide_target_present"]:
        result["required_checks_passed"] = False
        result["failed_required_commands"].append("repository_wide_target")
    return result


def qualification_state(validation: Mapping[str, Any], *, mechanics_passed: bool) -> JsonDict:
    """Keep mechanics, current validation, and terminal class consistent."""

    if validation.get("required_checks_passed") is not True:
        return {
            "ready_score": 0,
            "status": "complete",
            "verdict_class": "disqualified",
            "honest_verdict": "complete_disqualified_batch_harness_validation_failed",
        }
    if not mechanics_passed:
        return {
            "ready_score": 0,
            "status": "complete",
            "verdict_class": "null",
            "honest_verdict": "complete_null_batch_harness_mechanics_failed",
        }
    return {
        "ready_score": 1,
        "status": "complete",
        "verdict_class": "circular_positive",
        "honest_verdict": "complete_circular_positive_batch_harness_qualified",
    }


def panel_contract_errors(public: Mapping[str, Any], scorer: Mapping[str, Any]) -> list[str]:
    """Check group, version, claim, and authority boundaries before panel reuse."""

    errors = []
    split_counts = {"development_groups": 8, "evaluation_groups": 16}
    for split, count in split_counts.items():
        groups = public.get(split)
        if not isinstance(groups, list) or len(groups) != count:
            errors.append(split)
            continue
        for group in groups:
            versions = group.get("source_versions", [])
            claims = group.get("claims", [])
            if [row.get("source_version") for row in versions] != [1, 2]:
                errors.append("source_versions")
            for version in (1, 2):
                ids = [row.get("unit_id") for row in claims if row.get("source_version") == version]
                if len(ids) != 4 or len(set(ids)) != 4:
                    errors.append("claims_per_version")
    public_text = _canonical_json(public)
    if "expected_decision" in public_text or "case_type" in public_text:
        errors.append("public_authority_fields")
    labels = scorer.get("labels")
    if not isinstance(labels, list) or len(labels) != 192:
        errors.append("label_denominator")
    elif len({row.get("unit_id") for row in labels}) != 192:
        errors.append("label_identity")
    return list(dict.fromkeys(errors))


def load_or_build_panel(root: Path) -> tuple[JsonDict, JsonDict, JsonDict]:
    """Reuse valid V642 bytes or rebuild only the declared deterministic view."""

    fresh_public, fresh_scorer = fixture.build_fixture()
    public_path = root / V642_PUBLIC_PATH
    label_path = root / V642_LABEL_PATH
    if public_path.is_file() and label_path.is_file():
        try:
            public = json.loads(public_path.read_text(encoding="utf-8"))
            scorer = json.loads(label_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            public, scorer = fresh_public, fresh_scorer
            origin = "rebuilt_declared_view_after_unreadable_history"
        else:
            errors = panel_contract_errors(public, scorer)
            if not errors:
                return (
                    public,
                    scorer,
                    {
                        "origin": "reused_v642_bytes",
                        "public_source_path": V642_PUBLIC_PATH.as_posix(),
                        "label_source_path": V642_LABEL_PATH.as_posix(),
                        "public_source_sha256": _sha256_file(public_path),
                        "label_source_sha256": _sha256_file(label_path),
                        "materialized_difference": None,
                    },
                )
            public, scorer = fresh_public, fresh_scorer
            origin = "rebuilt_declared_view_after_contract_mismatch"
    else:
        public, scorer = fresh_public, fresh_scorer
        origin = "rebuilt_declared_view_after_missing_history"
    return (
        public,
        scorer,
        {
            "origin": origin,
            "public_source_path": V642_PUBLIC_PATH.as_posix(),
            "label_source_path": V642_LABEL_PATH.as_posix(),
            "public_source_sha256": _sha256_file(public_path) if public_path.is_file() else None,
            "label_source_sha256": _sha256_file(label_path) if label_path.is_file() else None,
            "materialized_difference": "deterministic_declared_view_only",
        },
    )


def _extra_controls(public: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Exercise the three fixture cases not named by the V642 control reducer."""

    group = public["evaluation_groups"][0]
    claims_v1 = fixture._claims_for(group, 1)
    claims_v2 = fixture._claims_for(group, 2)
    mixed = fixture.build_batch_request(
        "batched_warm_prefix_direct", group, [claims_v1[0], claims_v2[0]]
    )
    absent_text = "No relation span is available."
    absent_document = fixture.make_document("absent-source", absent_text)
    absent = fixture.VersionedSourceCompiler().compile(
        "absent-source",
        {
            "source_version": 1,
            "source_hash": fixture.sha256_bytes(absent_text.encode("utf-8")),
            "document": absent_document,
        },
    )
    unsupported_units = {
        row["unit_id"] for row in rows if row.get("expected_decision") == "unknown"
    }
    unsupported_rows = [row for row in rows if row["unit_id"] in unsupported_units]
    values = (
        ("mixed_source_versions", "mixed_source_versions", mixed.get("error")),
        ("absent_source_span", "source_compile_failed", absent.get("error")),
        (
            "unsupported_claim",
            {"unit_count": 32, "row_count": 96, "all_unknown": True},
            {
                "unit_count": len(unsupported_units),
                "row_count": len(unsupported_rows),
                "all_unknown": all(row.get("prediction") == "unknown" for row in unsupported_rows),
            },
        ),
    )
    return [
        {
            "control": name,
            "expected": expected,
            "observed": observed,
            "passed": observed == expected,
            "principle": "The injected edge case must reject or preserve its declared semantics.",
        }
        for name, expected, observed in values
    ]


def build_injected_costs(public: Mapping[str, Any]) -> list[JsonDict]:
    """Create a labeled contract witness without claiming measured live cost."""

    costs = {
        "serial_versioned_verifier": 8.0,
        "batched_versioned_verifier": 2.0,
        "batched_warm_prefix_direct": 4.0,
    }
    return [
        {
            "group_id": group["group_id"],
            "source_version": version,
            "arm": arm,
            "injected_full_cost": cost,
            "evidence_kind": "injected_acceptance_contract_witness_not_measured_cost",
        }
        for group in public["evaluation_groups"]
        for version in (1, 2)
        for arm, cost in costs.items()
    ]


def _apply_cost_witness(gates: list[JsonDict], costs: Sequence[Mapping[str, Any]]) -> None:
    """Fill only the two preregistered cost gates from labeled injected values."""

    by_unit = {
        (row["group_id"], row["source_version"], row["arm"]): float(row["injected_full_cost"])
        for row in costs
    }
    units = sorted({(str(row["group_id"]), int(row["source_version"])) for row in costs})
    serial = min(
        by_unit[(*unit, "serial_versioned_verifier")]
        / by_unit[(*unit, "batched_versioned_verifier")]
        for unit in units
    )
    direct = min(
        by_unit[(*unit, "batched_warm_prefix_direct")]
        / by_unit[(*unit, "batched_versioned_verifier")]
        for unit in units
    )
    observed = {
        "full_cost_speedup_lower_vs_serial": serial,
        "full_cost_speedup_lower_vs_direct": direct,
    }
    for row in gates:
        if row["criterion"] in observed:
            row["observed"] = observed[row["criterion"]]
            row["passed"] = row["observed"] >= 1.5
            row["principle"] = (
                "A labeled injected-cost witness proves a nonempty pass region; it is not live cost."
            )
            row["evidence_kind"] = "injected_acceptance_contract_witness"


def reduce_rows(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Independently reduce raw claim rows without reading artifact aggregates."""

    by_key = {(str(row["unit_id"]), str(row["arm"])): row for row in rows}
    units = sorted({str(row["unit_id"]) for row in rows})
    mismatches = sum(
        by_key[(unit, "serial_versioned_verifier")]["prediction"]
        != by_key[(unit, "batched_versioned_verifier")]["prediction"]
        for unit in units
    )
    return {
        "row_count": len(rows),
        "unit_count": len(units),
        "arm_counts": dict(sorted(Counter(str(row["arm"]) for row in rows).items())),
        "semantic_mismatches": mismatches,
        "stale_constraints_served": sum(
            int(bool(row.get("served_stale_constraints"))) for row in rows
        ),
        "false_accepts": {
            arm: sum(int(row.get("false_accept", 0)) for row in rows if row.get("arm") == arm)
            for arm in fixture.ARMS
        },
        "complete_rows": sum(not bool(row.get("censored")) for row in rows),
        "censored_rows": sum(bool(row.get("censored")) for row in rows),
    }


def validate_raw_reduction(
    rows: Sequence[Mapping[str, Any]], expected: Mapping[str, Any]
) -> list[str]:
    """Require a fresh reduction to match the sealed raw-row receipt."""

    return [] if reduce_rows(rows) == expected else ["raw_reduction"]


def _panel_manifest(
    public: Mapping[str, Any],
    scorer: Mapping[str, Any],
    execution: Mapping[str, Any],
    costs: Sequence[Mapping[str, Any]],
    panel_receipt: Mapping[str, Any],
) -> JsonDict:
    """Seal public groups, private labels, costs, and call order separately."""

    groups = []
    for split, split_groups in (
        ("development", public["development_groups"]),
        ("held_out", public["evaluation_groups"]),
    ):
        for group in split_groups:
            groups.append(
                {
                    "split": split,
                    "group_id": group["group_id"],
                    "source_id": group["source_id"],
                    "versions": [
                        {
                            "source_version": source["source_version"],
                            "source_hash": source["source_hash"],
                            "claim_ids": [
                                claim["unit_id"]
                                for claim in group["claims"]
                                if claim["source_version"] == source["source_version"]
                            ],
                        }
                        for source in group["source_versions"]
                    ],
                }
            )
    return {
        "panel_origin": deepcopy(dict(panel_receipt)),
        "development_group_count": 8,
        "held_out_group_count": 16,
        "source_versions_per_group": 2,
        "claims_per_source_version": 4,
        "labels_separate_from_public_input": True,
        "public_panel_path": PUBLIC_PATH.as_posix(),
        "public_panel_sha256": _sha256_bytes(_canonical_json(public).encode("utf-8")),
        "evaluator_labels_path": LABEL_PATH.as_posix(),
        "evaluator_labels_sha256": _sha256_bytes(_canonical_json(scorer).encode("utf-8")),
        "injected_output_path": PAYLOAD_PATH.as_posix(),
        "injected_output_sha256": _sha256_bytes(
            _canonical_json(execution["payloads"]).encode("utf-8")
        ),
        "injected_cost_path": COST_PATH.as_posix(),
        "injected_cost_sha256": _sha256_bytes(_canonical_json(costs).encode("utf-8")),
        "development_selection": [
            row["group_id"] for row in groups if row["split"] == "development"
        ],
        "held_out_order": [row["group_id"] for row in groups if row["split"] == "held_out"],
        "call_order_sha256": _sha256_bytes(
            _canonical_json([row["call_id"] for row in execution["call_rows"]]).encode("utf-8")
        ),
        "groups": groups,
    }


def build_harness_evidence(
    public: Mapping[str, Any] | None = None,
    scorer: Mapping[str, Any] | None = None,
    *,
    draws: int = 10_000,
    panel_receipt: Mapping[str, Any] | None = None,
) -> JsonDict:
    """Run unchanged CPU mechanics and build the injected acceptance witness."""

    if public is None or scorer is None:
        built_public, built_scorer = fixture.build_fixture()
        public = built_public if public is None else public
        scorer = built_scorer if scorer is None else scorer
    panel_errors = panel_contract_errors(public, scorer)
    if panel_errors:
        raise ValueError("panel contract: " + ",".join(panel_errors))
    execution = fixture.execute_public_fixture(public)
    rows = fixture.score_predictions(execution["predictions"], scorer["labels"])
    controls = [*fixture.run_batch_controls(public), *_extra_controls(public, rows)]
    bootstrap = fixture.paired_group_bootstrap(rows, draws=draws, seed=fixture.BOOTSTRAP_SEED)
    gates = fixture.acceptance_gates(rows, controls, bootstrap, execution["call_rows"])
    costs = build_injected_costs(public)
    _apply_cost_witness(gates, costs)
    reduction = reduce_rows(rows)
    receipt = panel_receipt or {
        "origin": "deterministic_v642_builder",
        "materialized_difference": None,
    }
    return {
        "public": deepcopy(dict(public)),
        "scorer": deepcopy(dict(scorer)),
        "execution": execution,
        "rows": rows,
        "controls": controls,
        "bootstrap": bootstrap,
        "gates": gates,
        "costs": costs,
        "independent_reduction": reduction,
        "panel_manifest": _panel_manifest(public, scorer, execution, costs, receipt),
    }


def dependency_check(
    artifact: Mapping[str, Any] | None, upstream: str, score_field: str
) -> list[JsonDict]:
    """Reject absent and failure-class dependencies before trusting a score."""

    if artifact is None:
        return [
            gate_row(
                "dependency_available",
                upstream,
                "artifact",
                "available",
                "missing_artifact",
                False,
            )
        ]
    if artifact.get("flagged_adversarial") is True or artifact.get("quarantined") is True:
        return [
            gate_row(
                "dependency_quarantine",
                upstream,
                "quarantined",
                False,
                True,
                False,
            )
        ]
    terminal = artifact.get("verdict_class")
    if terminal in {"blocked", "partial", "disqualified"}:
        return [
            gate_row(
                "dependency_terminal_class",
                upstream,
                "verdict_class",
                "not_failure_class",
                terminal,
                False,
            )
        ]
    return [
        gate_row(
            "dependency_ready_score",
            upstream,
            score_field,
            1,
            artifact.get(score_field),
            artifact.get(score_field) == 1,
        )
    ]


def _consumer_inputs(artifact: Mapping[str, Any] | None, kind: str) -> JsonDict:
    """Return sealed group identities only after all current dependency gates pass."""

    checks = dependency_check(artifact, EXPERIMENT_ID, "batch_harness_ready_score")
    if any(row["passed"] is not True for row in checks):
        return {"ok": False, "gate_check_summary": gate_check_summary(checks)}
    assert artifact is not None
    manifest = artifact.get("sealed_panel_manifest", {})
    split = "development" if kind == "canary" else "held_out"
    limit = 2 if kind == "canary" else 16
    groups = [row["group_id"] for row in manifest.get("groups", []) if row.get("split") == split]
    if len(groups) < limit:
        failed = gate_row(
            "sealed_group_denominator",
            EXPERIMENT_ID,
            f"{split}_groups",
            limit,
            len(groups),
            False,
        )
        return {"ok": False, "gate_check_summary": gate_check_summary([failed])}
    target = validation_scope(kind)
    return {
        "ok": True,
        "upstream": EXPERIMENT_ID,
        "group_ids": groups[:limit],
        "public_panel_path": manifest["public_panel_path"],
        "public_panel_sha256": manifest["public_panel_sha256"],
        "validation_scope": target,
    }


def build_canary_inputs(artifact: Mapping[str, Any] | None) -> JsonDict:
    """Build the bounded development input contract for the planned canary."""

    return _consumer_inputs(artifact, "canary")


def build_capture_inputs(artifact: Mapping[str, Any] | None) -> JsonDict:
    """Build the held-out input contract for the planned capture."""

    return _consumer_inputs(artifact, "capture")


def authenticate_inputs(root: Path) -> tuple[list[JsonDict], JsonDict]:
    """Authenticate current executable inputs before fixture work starts."""

    checks = []
    hashes: JsonDict = {}
    for name, relative in REQUIRED_CURRENT_PATHS.items():
        path = root / relative
        available = path.is_file()
        checks.append(
            gate_row(
                "required_path_available",
                name,
                "path",
                "file",
                "file" if available else "missing",
                available,
            )
        )
        if available:
            hashes[relative.as_posix()] = {
                "sha256": _sha256_file(path),
                "producer_identity": name,
                "readiness_gate": True,
            }
    manifest_path = root / REQUIRED_CURRENT_PATHS["exclusion_manifest"]
    try:
        manifest = (
            yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
            if manifest_path.is_file()
            else {}
        )
    except yaml.YAMLError:
        manifest = {"malformed": True}
    excluded = fixture.reuse._manifest_lists_experiment(manifest, EXPERIMENT_ID)
    checks.append(
        gate_row(
            "exclusion_manifest",
            EXPERIMENT_ID,
            "retired_or_quarantined",
            False,
            excluded,
            not excluded,
        )
    )
    output = (root / RESULT_PATH).resolve()
    inside_results = output.is_relative_to((root / "results").resolve())
    checks.append(
        gate_row(
            "declared_output_path",
            EXPERIMENT_ID,
            "path_scope",
            "inside_results",
            "inside_results" if inside_results else str(output),
            inside_results,
        )
    )
    return checks, hashes


def historical_diagnostics(root: Path) -> tuple[list[JsonDict], JsonDict]:
    """Authenticate old failed launchers without using them as readiness gates."""

    diagnostics = []
    hashes: JsonDict = {}
    for relative in HISTORICAL_ARTIFACTS:
        path = root / relative
        if not path.is_file():
            diagnostics.append(
                gate_row(
                    "historical_artifact_available",
                    relative.as_posix(),
                    "path",
                    "diagnostic_only",
                    "missing",
                    False,
                    readiness_gate=False,
                )
            )
            continue
        hashes[relative.as_posix()] = {
            "sha256": _sha256_file(path),
            "producer_identity": "historical_diagnostic",
            "readiness_gate": False,
        }
        try:
            artifact = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            diagnostics.append(
                gate_row(
                    "historical_artifact_parse",
                    relative.as_posix(),
                    "json",
                    "diagnostic_only",
                    "malformed",
                    False,
                    readiness_gate=False,
                )
            )
            continue
        receipt = next(
            (
                row
                for row in artifact.get("validation_receipts", [])
                if row.get("name") == "full_python_suite"
            ),
            None,
        )
        log_path = root / str(receipt.get("log_path", "")) if receipt else root / "missing"
        actual_hash = _sha256_file(log_path) if log_path.is_file() else None
        expected_hash = receipt.get("log_sha256") if receipt else None
        diagnostics.append(
            {
                **gate_row(
                    "historical_full_python_suite_failure",
                    str(artifact.get("experiment_id", relative.stem)),
                    "log_sha256",
                    expected_hash,
                    actual_hash,
                    receipt is not None and actual_hash == expected_hash,
                    principle="Authenticate the old failure without making it a current gate.",
                    readiness_gate=False,
                ),
                "terminal_class": artifact.get("verdict_class"),
                "honest_verdict": artifact.get("honest_verdict"),
                "historical_ready_score": artifact.get(
                    "batch_fixture_ready_score", artifact.get("batch_canary_ready_score")
                ),
                "command": receipt.get("command") if receipt else None,
                "exit_code": receipt.get("exit_code") if receipt else None,
                "log_path": receipt.get("log_path") if receipt else None,
            }
        )
        if log_path.is_file():
            hashes[str(receipt["log_path"])] = {
                "sha256": actual_hash,
                "producer_identity": "historical_full_python_suite_failure",
                "readiness_gate": False,
            }
    return diagnostics, hashes


def repository_health(root: Path) -> JsonDict:
    """Reuse dated Exp7303 observations without turning them into current checks."""

    path = root / HEALTH_SOURCE
    if not path.is_file():
        return scoped.build_repository_health(
            [
                {
                    "experiment_id": "historical_repository_health",
                    "observed_at_utc": None,
                    "collection_errors": [],
                    "resolved": False,
                    "observation": "health source unavailable",
                }
            ]
        )
    try:
        artifact = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return scoped.build_repository_health(
            [
                {
                    "experiment_id": "historical_repository_health",
                    "observed_at_utc": None,
                    "collection_errors": [],
                    "resolved": False,
                    "observation": "health source malformed",
                }
            ]
        )
    health = deepcopy(artifact.get("repository_health", {}))
    if not health:
        return scoped.build_repository_health([])
    health["affects_required_checks"] = False
    return health


def artifact_checksum(artifact: Mapping[str, Any]) -> str:
    """Bind durable evidence while excluding process-local timing fields."""

    stable = {
        key: value
        for key, value in artifact.items()
        if key not in {"duration_s", "timestamps", "phase_spans", "reproducibility_checksum"}
    }
    return _sha256_bytes(_canonical_json(stable).encode("utf-8"))


def _validation_failure_checks(validation: Mapping[str, Any]) -> list[JsonDict]:
    """Convert missing and failed scoped commands to exact gate comparisons."""

    checks = []
    for name in validation.get("missing_required_commands", []):
        checks.append(
            gate_row(
                "required_validation_present", str(name), "receipt", "present", "missing", False
            )
        )
    failed = set(validation.get("failed_required_commands", []))
    for row in validation.get("validation_receipts", []):
        if row.get("name") in failed:
            checks.append(
                gate_row(
                    "required_validation_passed",
                    str(row["name"]),
                    "exit_code",
                    0,
                    row.get("exit_code"),
                    False,
                )
            )
    return checks


def assemble_artifact(
    evidence: Mapping[str, Any],
    validation: Mapping[str, Any],
    *,
    preconditions: Sequence[Mapping[str, Any]],
    diagnostics: Sequence[Mapping[str, Any]],
    duration_s: float,
    phase_spans: Sequence[Mapping[str, Any]],
    timestamps: Mapping[str, Any],
    source_hashes: Mapping[str, Any] | None = None,
) -> JsonDict:
    """Assemble one candidate while forcing failed validation out of readiness."""

    mechanics_passed = (
        all(row.get("passed") is True for row in evidence["gates"])
        and all(row.get("passed") is True for row in evidence["controls"])
        and not validate_raw_reduction(evidence["rows"], evidence["independent_reduction"])
    )
    state = qualification_state(validation, mechanics_passed=mechanics_passed)
    acceptance = [deepcopy(dict(row)) for row in evidence["gates"]]
    acceptance.append(
        {
            "criterion": "required_scoped_validation",
            "expected": True,
            "observed": validation.get("required_checks_passed"),
            "passed": validation.get("required_checks_passed") is True,
            "principle": "A current affected failure disqualifies fixture readiness.",
        }
    )
    failure_checks = _validation_failure_checks(validation)
    if not mechanics_passed and not failure_checks:
        failed_gate = next(
            (
                row
                for row in [*evidence["gates"], *evidence["controls"]]
                if row.get("passed") is not True
            ),
            {"criterion": "raw_reduction", "expected": "match", "observed": "mismatch"},
        )
        failure_checks.append(
            gate_row(
                "fixture_mechanics",
                EXPERIMENT_ID,
                str(failed_gate.get("criterion", failed_gate.get("control"))),
                failed_gate.get("expected"),
                failed_gate.get("observed"),
                False,
            )
        )
    all_hashes = deepcopy(dict(source_hashes or {}))
    all_hashes.update(
        {
            PUBLIC_PATH.as_posix(): {
                "sha256": evidence["panel_manifest"]["public_panel_sha256"],
                "producer_identity": EXPERIMENT_ID,
            },
            LABEL_PATH.as_posix(): {
                "sha256": evidence["panel_manifest"]["evaluator_labels_sha256"],
                "producer_identity": EXPERIMENT_ID,
            },
            PAYLOAD_PATH.as_posix(): {
                "sha256": evidence["panel_manifest"]["injected_output_sha256"],
                "producer_identity": EXPERIMENT_ID,
            },
            COST_PATH.as_posix(): {
                "sha256": evidence["panel_manifest"]["injected_cost_sha256"],
                "producer_identity": EXPERIMENT_ID,
            },
        }
    )
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "status": state["status"],
        "run_date": RUN_DATE,
        "field_principles": deepcopy(FIELD_PRINCIPLES),
        "preconditions_checked": [deepcopy(dict(row)) for row in preconditions],
        "MODEL_SPECS": [],
        "model_invoked": False,
        "invocation_counts": deepcopy(ZERO_INVOCATION_COUNTS),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "inference_substrate_class": INFERENCE_SUBSTRATE_CLASS,
        "execution_venue": EXECUTION_VENUE,
        "duration_s": duration_s,
        "timestamps": deepcopy(dict(timestamps)),
        "phase_spans": [deepcopy(dict(row)) for row in phase_spans],
        "random_seed": {
            "fixture": fixture.RANDOM_SEED,
            "development": fixture.DEVELOPMENT_SEED,
            "evaluation": fixture.EVALUATION_SEED,
            "bootstrap": fixture.BOOTSTRAP_SEED,
        },
        "source_artifact_hashes": all_hashes,
        "rows": [deepcopy(dict(row)) for row in evidence["rows"]],
        "sample_size_budget": {
            "planned_development_groups": 8,
            "planned_held_out_groups": 16,
            "planned_units_per_arm": 128,
            "planned_arm_rows": 384,
            "attempted_units_per_arm": {
                arm: sum(row["arm"] == arm for row in evidence["rows"]) for arm in fixture.ARMS
            },
            "complete_arm_rows": evidence["independent_reduction"]["complete_rows"],
            "censored_arm_rows": evidence["independent_reduction"]["censored_rows"],
            "stopping_rule": "fixed_16_groups_128_units_per_arm_no_optional_stopping",
        },
        "acceptance_gate_results": acceptance,
        "gate_check_summary": gate_check_summary(failure_checks),
        "verifier_is_oracle": True,
        "honest_verdict": state["honest_verdict"],
        "verdict_class": state["verdict_class"],
        "validation_receipts": [
            deepcopy(dict(row)) for row in validation.get("validation_receipts", [])
        ],
        "repository_health": deepcopy(dict(validation["repository_health"])),
        "batch_harness_ready_score": state["ready_score"],
        "validation_entrypoint_receipt": deepcopy(
            dict(validation["validation_entrypoint_receipt"])
        ),
        "sealed_panel_manifest": deepcopy(dict(evidence["panel_manifest"])),
        "call_budget_contract": fixture.call_budget_contract(),
        "acceptance_contract": {
            **fixture.acceptance_contract(),
            "pass_region_witness": "separately_labeled_injected_outputs_and_costs",
            "injected_witness_claims_live_performance": False,
        },
        "batch_control_rows": [deepcopy(dict(row)) for row in evidence["controls"]],
        "independent_reduction": deepcopy(dict(evidence["independent_reduction"])),
        "authority_separation": {
            "prediction_callable_inputs": ["public_panel", "injected_transport"],
            "scoring_callable_inputs": ["public_predictions", "evaluator_labels"],
            "evaluation_labels_read_during_prediction": evidence["execution"][
                "evaluation_labels_read"
            ],
            "labels_present_in_injected_outputs": False,
            "costs_separately_labeled": True,
        },
        "diagnostic_evidence": [deepcopy(dict(row)) for row in diagnostics],
        "methodology_note": (
            "This CPU exact harness qualifies fixture and validation mechanics. Injected outputs and "
            "costs prove a nonempty contract pass region. They are not held-out model output, live "
            "latency, learned accuracy, or a scientific value result."
        ),
    }
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    return artifact


def blocked_artifact(
    run_date: str,
    checks: Sequence[Mapping[str, Any]],
    *,
    duration_s: float,
    timestamps: Mapping[str, Any],
) -> JsonDict:
    """Finish an external failure without creating success-shaped fixture evidence."""

    summary = gate_check_summary(checks)
    artifact: JsonDict = {
        **{field: None for field in REQUIRED_ARTIFACT_FIELDS},
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "status": "blocked",
        "run_date": run_date,
        "field_principles": deepcopy(FIELD_PRINCIPLES),
        "preconditions_checked": [deepcopy(dict(row)) for row in checks],
        "MODEL_SPECS": [],
        "model_invoked": False,
        "invocation_counts": deepcopy(ZERO_INVOCATION_COUNTS),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "inference_substrate_class": INFERENCE_SUBSTRATE_CLASS,
        "execution_venue": EXECUTION_VENUE,
        "duration_s": duration_s,
        "timestamps": deepcopy(dict(timestamps)),
        "phase_spans": [
            {
                "phase": "preconditions",
                "duration_s": duration_s,
                "units": len(checks),
                "checkpoint_boundary": None,
                "pending_operations": [],
            }
        ],
        "random_seed": {
            "fixture": fixture.RANDOM_SEED,
            "development": fixture.DEVELOPMENT_SEED,
            "evaluation": fixture.EVALUATION_SEED,
            "bootstrap": fixture.BOOTSTRAP_SEED,
        },
        "source_artifact_hashes": {},
        "rows": [],
        "sample_size_budget": {
            "planned_arm_rows": 384,
            "attempted_arm_rows": 0,
            "complete_arm_rows": 0,
            "censored_arm_rows": 384,
            "stopping_rule": "terminal_block_on_failed_external_precondition",
        },
        "acceptance_gate_results": [],
        "gate_check_summary": summary,
        "verifier_is_oracle": True,
        "honest_verdict": f"blocked_{summary['upstream']}_{summary['field']}",
        "verdict_class": "blocked",
        "validation_receipts": [],
        "repository_health": {
            "status": "unknown_before_scoped_validation",
            "affects_required_checks": False,
        },
        "batch_harness_ready_score": 0,
        "validation_entrypoint_receipt": {
            "runner": SCOPED_RUNNER,
            "called": False,
            "test_paths": validation_scope("harness")["test_paths"],
            "changed_modules": validation_scope("harness")["changed_modules"],
            "static_paths": validation_scope("harness")["static_paths"],
            "legacy_launcher_called": False,
            "repository_wide_target_present": False,
        },
        "sealed_panel_manifest": None,
        "call_budget_contract": fixture.call_budget_contract(),
        "acceptance_contract": fixture.acceptance_contract(),
        "batch_control_rows": [],
        "independent_reduction": None,
        "authority_separation": None,
        "diagnostic_evidence": [],
        "methodology_note": "Current fixture work did not start because an external input failed.",
    }
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    return artifact


def validate_artifact(
    artifact: Mapping[str, Any], *, require_terminal_checks: bool = True
) -> list[str]:
    """Cold-check identity, denominators, controls, scope, and terminal semantics."""

    errors = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    expected = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "run_date": RUN_DATE,
        "MODEL_SPECS": [],
        "model_invoked": False,
        "invocation_counts": ZERO_INVOCATION_COUNTS,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "inference_substrate_class": INFERENCE_SUBSTRATE_CLASS,
        "execution_venue": EXECUTION_VENUE,
    }
    for field, value in expected.items():
        if artifact.get(field) != value:
            errors.append(field)
    if artifact.get("verdict_class") not in {
        "positive",
        "circular_positive",
        "null",
        "blocked",
        "disqualified",
        "partial",
    }:
        errors.append("verdict_class")
    status = artifact.get("status")
    if status == "blocked":
        if artifact.get("batch_harness_ready_score") != 0 or artifact.get("rows") != []:
            errors.append("blocked_readiness")
        if not str(artifact.get("honest_verdict", "")).startswith("blocked_"):
            errors.append("honest_verdict")
    elif status == "complete":
        rows = artifact.get("rows", [])
        if len(rows) != 384:
            errors.append("rows")
        elif Counter(row.get("arm") for row in rows) != {arm: 128 for arm in fixture.ARMS}:
            errors.append("row_arm_denominator")
        try:
            reduction_errors = validate_raw_reduction(
                rows, artifact.get("independent_reduction", {})
            )
        except (KeyError, TypeError, ZeroDivisionError):
            reduction_errors = ["raw_reduction"]
        if reduction_errors:
            errors.append("independent_reduction")
        controls = artifact.get("batch_control_rows", [])
        if len(controls) != 10 or any(row.get("passed") is not True for row in controls):
            errors.append("batch_control_rows")
        if artifact.get("batch_harness_ready_score") == 1 and any(
            row.get("passed") is not True for row in artifact.get("acceptance_gate_results", [])
        ):
            errors.append("acceptance_gate_results")
        entrypoint = artifact.get("validation_entrypoint_receipt", {})
        if (
            entrypoint.get("runner") != SCOPED_RUNNER
            or entrypoint.get("legacy_launcher_called") is not False
            or entrypoint.get("repository_wide_target_present") is not False
        ):
            errors.append("validation_entrypoint_receipt")
        receipts = artifact.get("validation_receipts", [])
        names = {row.get("name") for row in receipts if row.get("passed") is True}
        if artifact.get("batch_harness_ready_score") == 1:
            if not set(scoped.REQUIRED_CHECK_NAMES).issubset(names):
                errors.append("required_scoped_validation")
            if require_terminal_checks and not set(TERMINAL_CHECK_NAMES).issubset(names):
                errors.append("terminal_checks")
            if artifact.get("verdict_class") != "circular_positive":
                errors.append("ready_verdict_class")
        if not str(artifact.get("honest_verdict", "")).startswith("complete_"):
            errors.append("honest_verdict")
    else:
        errors.append("status")
    if artifact.get("reproducibility_checksum") != artifact_checksum(artifact):
        errors.append("reproducibility_checksum")
    return list(dict.fromkeys(errors))


def write_json(root: Path, path: Path, value: Mapping[str, Any]) -> Path:
    """Atomically write one task-owned JSON file under the result root."""

    return atomic_write_json(path, value, root=root, allow_override=False, sort_keys=True)


def _write_sidecars(root: Path, evidence: Mapping[str, Any]) -> JsonDict:
    """Write public, private, injected, and reduced evidence to distinct files."""

    values = {
        PUBLIC_PATH: evidence["public"],
        LABEL_PATH: evidence["scorer"],
        PREDICTION_PATH: {"predictions": evidence["execution"]["predictions"]},
        ROW_PATH: {"rows": evidence["rows"]},
        CALL_PATH: {"call_rows": evidence["execution"]["call_rows"]},
        PAYLOAD_PATH: {"payloads": evidence["execution"]["payloads"]},
        COST_PATH: {"costs": evidence["costs"]},
        CONTROL_PATH: {"controls": evidence["controls"]},
        REDUCTION_PATH: evidence["independent_reduction"],
    }
    hashes = {}
    for relative, value in values.items():
        target = write_json(root, root / relative, value)
        hashes[relative.as_posix()] = {
            "sha256": _sha256_file(target),
            "producer_identity": EXPERIMENT_ID,
        }
    return hashes


def _terminal_commands(root: Path, candidate: Path) -> list[scoped.CommandSpec]:
    """Build artifact-only checks without adding any pytest target."""

    python = str(root / ".venv/bin/python")
    script = "scripts/experiments/experiment_7317_v643_batch_harness.py"
    return [
        scoped.CommandSpec(
            TERMINAL_CHECK_NAMES[0],
            (
                python,
                "-u",
                script,
                "--date",
                RUN_DATE,
                "--check-artifact",
                str(candidate),
                "--raw-rows",
                str(root / ROW_PATH),
            ),
            "terminal_candidate_and_raw_rows",
        ),
        scoped.CommandSpec(
            TERMINAL_CHECK_NAMES[1],
            (python, "-u", "scripts/adversarial_verify.py", str(candidate)),
            "terminal_candidate",
        ),
        scoped.CommandSpec(
            TERMINAL_CHECK_NAMES[2],
            (
                python,
                "-u",
                "scripts/verdict_row_consistency_lint.py",
                "--strict",
                str(candidate),
            ),
            "terminal_candidate",
        ),
    ]


def run_terminal_checks(
    root: Path,
    candidate: Path,
    *,
    extra_env: Mapping[str, str] | None = None,
) -> list[JsonDict]:
    """Stream candidate reload, adversarial, and strict consistency checks."""

    return scoped.run_commands(
        root,
        _terminal_commands(root, candidate),
        log_dir=root / RAW_DIR / "terminal-validation",
        extra_env=extra_env,
    )


def _phase_row(phase: str, started: float, units: int) -> JsonDict:
    """Close one disjoint monotonic phase with explicit empty pending work."""

    return {
        "phase": phase,
        "duration_s": time.monotonic() - started,
        "units": units,
        "checkpoint_boundary": None,
        "pending_operations": [],
    }


def run_experiment(
    root: Path,
    run_date: str,
    *,
    extra_env: Mapping[str, str] | None = None,
) -> JsonDict:
    """Run the finite CPU harness, scoped checks, and atomic publication."""

    if run_date != RUN_DATE:
        raise ValueError(f"--date must be {RUN_DATE}")
    overall_started = time.monotonic()
    started_at = _utc_now()
    spans = []

    phase_started = time.monotonic()
    _progress("preconditions", "start", overall_started)
    checks, current_hashes = authenticate_inputs(root)
    diagnostics, historical_hashes = historical_diagnostics(root)
    checks.extend(diagnostics)
    spans.append(_phase_row("preconditions", phase_started, len(checks)))
    _progress("preconditions", "complete", overall_started, f"units={len(checks)}")
    if any(row["readiness_gate"] and row["passed"] is not True for row in checks):
        artifact = blocked_artifact(
            run_date,
            checks,
            duration_s=time.monotonic() - overall_started,
            timestamps={"started_at_utc": started_at, "completed_at_utc": _utc_now()},
        )
        errors = validate_artifact(artifact, require_terminal_checks=False)
        if errors:
            raise ValueError("blocked artifact invalid: " + ",".join(errors))
        write_json(root, root / RESULT_PATH, artifact)
        return artifact

    phase_started = time.monotonic()
    _progress("fixture", "start", overall_started, "units=384")
    public, scorer, panel_receipt = load_or_build_panel(root)
    evidence = build_harness_evidence(public, scorer, panel_receipt=panel_receipt)
    spans.append(_phase_row("fixture", phase_started, len(evidence["rows"])))
    _progress("fixture", "complete", overall_started, f"units={len(evidence['rows'])}")

    phase_started = time.monotonic()
    _progress("sidecars", "start", overall_started, "units=9")
    raw_hashes = _write_sidecars(root, evidence)
    spans.append(_phase_row("sidecars", phase_started, len(raw_hashes)))
    _progress("sidecars", "complete", overall_started, f"units={len(raw_hashes)}")

    phase_started = time.monotonic()
    _progress("scoped_validation", "before_subprocesses", overall_started, "units=8")
    history = repository_health(root).get("historical_failures", [])
    validation = run_scoped_checks(
        root,
        validation_scope("harness"),
        historical_failures=history,
        extra_env=extra_env,
    )
    spans.append(
        _phase_row("scoped_validation", phase_started, len(validation["validation_receipts"]))
    )
    _progress("scoped_validation", "after_subprocesses", overall_started)

    source_hashes = {**current_hashes, **historical_hashes, **raw_hashes}
    candidate = assemble_artifact(
        evidence,
        validation,
        preconditions=checks,
        diagnostics=diagnostics,
        duration_s=time.monotonic() - overall_started,
        phase_spans=spans,
        timestamps={"started_at_utc": started_at, "completed_at_utc": _utc_now()},
        source_hashes=source_hashes,
    )
    candidate_errors = validate_artifact(candidate, require_terminal_checks=False)
    if candidate_errors:
        raise ValueError("terminal candidate invalid: " + ",".join(candidate_errors))
    candidate_path = write_json(root, root / CANDIDATE_PATH, candidate)

    phase_started = time.monotonic()
    _progress("terminal_validation", "before_subprocesses", overall_started, "units=3")
    terminal_receipts = run_terminal_checks(root, candidate_path, extra_env=extra_env)
    spans.append(_phase_row("terminal_validation", phase_started, len(terminal_receipts)))
    _progress("terminal_validation", "after_subprocesses", overall_started)
    validation["validation_receipts"] = [
        *validation["validation_receipts"],
        *terminal_receipts,
    ]
    failed_terminal = [row["name"] for row in terminal_receipts if row.get("passed") is not True]
    if failed_terminal:
        validation["required_checks_passed"] = False
        validation["failed_required_commands"] = [
            *validation.get("failed_required_commands", []),
            *failed_terminal,
        ]

    artifact = assemble_artifact(
        evidence,
        validation,
        preconditions=checks,
        diagnostics=diagnostics,
        duration_s=time.monotonic() - overall_started,
        phase_spans=spans,
        timestamps={"started_at_utc": started_at, "completed_at_utc": _utc_now()},
        source_hashes=source_hashes,
    )
    errors = validate_artifact(artifact)
    if errors:
        raise ValueError("terminal artifact invalid: " + ",".join(errors))
    _progress("publish", "start", overall_started)
    write_json(root, root / RESULT_PATH, artifact)
    _progress("publish", "complete", overall_started)
    return artifact


def _date_argument(value: str) -> str:
    """Accept only the execution date fixed by the V643 contract."""

    if value != RUN_DATE:
        raise argparse.ArgumentTypeError(f"--date must be {RUN_DATE}")
    return value


def main(argv: Sequence[str] | None = None) -> int:
    """Run the harness or cold-check one task-owned candidate."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", type=_date_argument, required=True)
    parser.add_argument("--check-artifact", type=Path)
    parser.add_argument("--raw-rows", type=Path)
    arguments = parser.parse_args(argv)
    print("[exp7317] startup", flush=True)
    if arguments.check_artifact:
        artifact = json.loads(arguments.check_artifact.read_text(encoding="utf-8"))
        errors = validate_artifact(artifact, require_terminal_checks=False)
        if arguments.raw_rows:
            raw = json.loads(arguments.raw_rows.read_text(encoding="utf-8"))
            errors.extend(validate_raw_reduction(raw["rows"], artifact["independent_reduction"]))
        if errors:
            raise ValueError("artifact check failed: " + ",".join(dict.fromkeys(errors)))
        print("[exp7317] artifact_check=pass", flush=True)
        return 0
    root = find_repo_root(start=__file__)
    artifact = run_experiment(root, arguments.date)
    print(f"[exp7317] verdict={artifact['honest_verdict']}", flush=True)
    return 0


if __name__ == "__main__":  # pragma: no cover - the repository wrapper owns CLI execution.
    raise SystemExit(main())
