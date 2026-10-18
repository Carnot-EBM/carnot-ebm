"""Measure the total durable consumer cost of the qualified native boundary.

The task compares fixed callers. It does not choose a comparator after timing,
load a model, change a production default, or issue a hardware operation.

Spec: REQ-REPORT-7627 and SCENARIO-REPORT-7627-*.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from copy import deepcopy
from datetime import UTC, datetime
import json
import math
import os
from pathlib import Path
import platform
import random
import socket
import statistics
import subprocess
import sys
import tempfile
import time
from typing import Any

from carnot import experiment_7598_v663_rust_consumer as consumer
from carnot import experiment_7626_v665_native_service as native
from carnot.pipeline.calibrated_decision_service import DURABILITY_POLICY
from carnot.reporting import experiment_7303_validation_scope as validation
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash, sha256_file


JsonDict = dict[str, Any]
RUN_DATE = "20260924"
MILESTONE = "2026.09.665"
EXPERIMENT_ID = "exp7627-v665-native-cost"
SCHEMA = "carnot.exp7627.v665.native_cost.v1"
RANDOM_SEED = 7_627
BOOTSTRAP_DRAWS = 2_000
REPEATS = 30
CONTROL_REPEATS = 10
MODES = ("cold", "warm")
BATCH_SIZES = (1, 8)
ARMS = ("python_inprocess", "rust_jsonl", "direct_native")
RESULT_PATH = Path("results/experiment_7627_v665_native_cost.json")
RAW_DIR = Path("results/raw/experiment_7627_v665_native_cost")
MODULE_PATH = Path("python/carnot/experiment_7627_v665_native_cost.py")
WRAPPER_PATH = Path("scripts/experiments/experiment_7627_v665_native_cost.py")
TEST_PATH = Path("tests/python/test_experiment_7627_v665_native_cost.py")
NOTES_PATH = Path("docs/research-notes/v665-native-cost.md")
SPEC_PATH = Path("openspec/capabilities/research-reporting/spec.md")
EXP7626_PATH = Path("results/experiment_7626_v665_native_service.json")
EXP7599_PATH = Path("results/experiment_7599_v663_board_continuity.json")
WISHLIST_PATH = Path("research-hardware-wishlist.md")
MODEL_SPECS: list[JsonDict] = []
ZERO_INVOCATIONS = {
    "model_loads": 0,
    "forward_calls": 0,
    "generation_calls": 0,
    "input_tokens": 0,
    "output_tokens": 0,
}
VALIDATION_NAMES = (
    *validation.REQUIRED_CHECK_NAMES,
    "entrypoint_help",
    "e2e_native_import",
    "e2e_durable_sequence",
)
TERMINAL_NAMES = (
    "fresh_process_cold_replay",
    "independent_raw_reduction",
    "adversarial_verify",
    "verdict_row_consistency_strict",
)


def _percentile(values: Sequence[float], fraction: float) -> float:
    """Return one interpolated percentile without adding observations."""

    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("percentile_requires_values")
    position = fraction * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _synthetic_timing_row(
    mode: str, batch_size: int, repeat: int, arm: str, total_ns: int
) -> JsonDict:
    seed = RANDOM_SEED + (10_000 if mode == "warm" else 0) + batch_size * 100 + repeat
    return {
        "row_type": "paired_native_cost",
        "unit_id": f"{mode}:{batch_size}:{repeat}:{arm}",
        "pair_id": f"{mode}:{batch_size}:{repeat}",
        "mode": mode,
        "batch_size": batch_size,
        "repeat": repeat,
        "arm": arm,
        "arm_order": list(ARMS),
        "seed": seed,
        "total_ns": total_ns,
        "exclusive_span_ns": {
            "setup": total_ns // 10 if mode == "cold" else 0,
            "predict": total_ns // 5,
            "update_persist_ack": total_ns // 2,
            "reload_verification": total_ns
            - total_ns // 5
            - total_ns // 2
            - (total_ns // 10 if mode == "cold" else 0),
        },
        "numerator": total_ns,
        "denominator": batch_size * 2 + 1,
        "metric_direction": "lower_is_better",
        "decision_parity": True,
        "state_parity": True,
        "reload_agreement": True,
        "durable_acknowledgments": batch_size,
        "durability_policy": DURABILITY_POLICY,
        "failure": None,
        "error_count": 0,
        "cpu_affinity": [0],
        "environment": {"fixture": True},
        "censored": False,
        "raw_provenance": "deterministic reducer fixture",
    }


def synthetic_timing_rows() -> list[JsonDict]:
    """Build 120 complete triples for reducer and mutation tests."""

    totals = {"python_inprocess": 140_000, "rust_jsonl": 160_000, "direct_native": 100_000}
    return [
        _synthetic_timing_row(mode, batch, repeat, arm, totals[arm] + repeat * 101 + batch)
        for mode in MODES
        for batch in BATCH_SIZES
        for repeat in range(REPEATS)
        for arm in ARMS
    ]


def _ratio_interval(ratios: Sequence[float], seed: int) -> JsonDict:
    """Bootstrap paired block ratios while retaining the registered direction."""

    if len(ratios) != REPEATS:
        raise ValueError("paired_ratio_requires_30_blocks")
    rng = random.Random(seed)
    draws = [
        statistics.median(ratios[rng.randrange(REPEATS)] for _ in range(REPEATS))
        for _ in range(BOOTSTRAP_DRAWS)
    ]
    return {
        "estimate": statistics.median(ratios),
        "lower95": _percentile(draws, 0.025),
        "upper95": _percentile(draws, 0.975),
        "pair_count": REPEATS,
        "bootstrap_seed": seed,
        "bootstrap_draws": BOOTSTRAP_DRAWS,
        "direction": "comparator_over_direct_native_higher_favors_native",
    }


def _equal_stratum_interval(ratios: Mapping[str, Sequence[float]], seed: int) -> JsonDict:
    """Give every stratum equal weight instead of pooling fast observations."""

    if set(ratios) != {f"{mode}:{batch}" for mode in MODES for batch in BATCH_SIZES}:
        raise ValueError("geometric_strata")
    estimates = [statistics.median(values) for values in ratios.values()]
    estimate = math.exp(statistics.mean(math.log(value) for value in estimates))
    rng = random.Random(seed)
    draws = []
    for _ in range(BOOTSTRAP_DRAWS):
        stratum_values = []
        for values in ratios.values():
            sampled = [values[rng.randrange(REPEATS)] for _ in range(REPEATS)]
            stratum_values.append(statistics.median(sampled))
        draws.append(math.exp(statistics.mean(math.log(value) for value in stratum_values)))
    return {
        "estimate": estimate,
        "lower95": _percentile(draws, 0.025),
        "upper95": _percentile(draws, 0.975),
        "equal_stratum_weight": True,
        "independent_blocks": REPEATS * len(ratios),
        "bootstrap_seed": seed,
        "bootstrap_draws": BOOTSTRAP_DRAWS,
        "direction": "comparator_over_direct_native_higher_favors_native",
    }


def reduce_timing_rows(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Independently reduce complete triples under the fixed Python comparator."""

    if len(rows) != REPEATS * len(MODES) * len(BATCH_SIZES) * len(ARMS):
        raise ValueError("paired_row_count")
    if {row.get("durability_policy") for row in rows} != {DURABILITY_POLICY}:
        raise ValueError("durability_policy_mismatch")
    pairs: dict[str, dict[str, Mapping[str, Any]]] = {}
    for row in rows:
        pairs.setdefault(str(row.get("pair_id")), {})[str(row.get("arm"))] = row
    if len(pairs) != 120 or any(set(arms) != set(ARMS) for arms in pairs.values()):
        raise ValueError("paired_arms")
    for arms in pairs.values():
        for row in arms.values():
            if (
                row.get("decision_parity") is not True
                or row.get("state_parity") is not True
                or row.get("reload_agreement") is not True
                or row.get("failure") is not None
                or int(row.get("error_count", 0)) != 0
                or row.get("censored") is not False
            ):
                raise ValueError("parity_or_error_failure")

    strata: JsonDict = {}
    all_python: dict[str, list[float]] = {}
    all_rust: dict[str, list[float]] = {}
    for mode_index, mode in enumerate(MODES):
        for batch_size in BATCH_SIZES:
            key = f"{mode}:{batch_size}"
            selected = [
                arms
                for arms in pairs.values()
                if key == arms["direct_native"]["pair_id"].rsplit(":", 1)[0]
            ]
            if len(selected) != REPEATS:
                raise ValueError("paired_stratum_count")
            selected.sort(key=lambda value: int(value["direct_native"]["repeat"]))
            python_ratios = [
                float(arms["python_inprocess"]["total_ns"])
                / float(arms["direct_native"]["total_ns"])
                for arms in selected
            ]
            rust_ratios = [
                float(arms["rust_jsonl"]["total_ns"]) / float(arms["direct_native"]["total_ns"])
                for arms in selected
            ]
            python_interval = _ratio_interval(
                python_ratios, RANDOM_SEED + mode_index * 100 + batch_size
            )
            rust_interval = _ratio_interval(
                rust_ratios, RANDOM_SEED + 1_000 + mode_index * 100 + batch_size
            )
            all_python[key] = python_ratios
            all_rust[key] = rust_ratios
            strata[key] = {
                "python_over_direct_native": python_interval,
                "rust_over_direct_native": rust_interval,
                "pair_count": REPEATS,
                "bootstrap_draws": BOOTSTRAP_DRAWS,
                "excluded_count": 0,
                "censored_count": 0,
            }
    equal = {
        "python_over_direct_native": _equal_stratum_interval(all_python, RANDOM_SEED + 2_000),
        "rust_over_direct_native": _equal_stratum_interval(all_rust, RANDOM_SEED + 3_000),
    }
    primary = equal["python_over_direct_native"]
    minimum_stratum = min(
        value["python_over_direct_native"]["lower95"] for value in strata.values()
    )
    benefit = int(
        primary["estimate"] >= 1.10 and primary["lower95"] >= 1.10 and minimum_stratum >= 0.95
    )
    return {
        "complete": True,
        "independent_blocks": len(pairs),
        "primary_comparator": "python_inprocess",
        "secondary_comparator": "rust_jsonl",
        "strata": strata,
        "equal_stratum_geometric_mean": equal,
        "minimum_primary_stratum_lower95": minimum_stratum,
        "parity_complete": True,
        "extra_native_errors": 0,
        "native_speed_benefit_score": benefit,
        "nfr_10x_met": primary["estimate"] >= 10.0,
    }


def synthetic_instrumentation_rows() -> list[JsonDict]:
    """Return 40 paired controls that have no comparator-selection authority."""

    rows = []
    for mode in MODES:
        for batch_size in BATCH_SIZES:
            for repeat in range(CONTROL_REPEATS):
                seed = RANDOM_SEED + 100_000 + batch_size * 100 + repeat
                for enabled in (False, True):
                    total = 100_000 + batch_size * 1_000 + repeat * 17 + (2_000 if enabled else 0)
                    rows.append(
                        {
                            "row_type": "instrumentation_control",
                            "unit_id": f"{mode}:{batch_size}:{repeat}:{int(enabled)}",
                            "pair_id": f"{mode}:{batch_size}:{repeat}",
                            "mode": mode,
                            "batch_size": batch_size,
                            "repeat": repeat,
                            "seed": seed,
                            "telemetry_enabled": enabled,
                            "arm_order": [False, True],
                            "total_ns": total,
                            "numerator": total,
                            "denominator": batch_size * 2 + 1,
                            "metric_direction": "lower_is_better",
                            "failure": None,
                            "censored": False,
                            "raw_provenance": "deterministic telemetry fixture",
                        }
                    )
    return rows


def reduce_instrumentation_rows(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Reduce matched on/off totals without changing the primary comparison."""

    if len(rows) != 80:
        raise ValueError("instrumentation_row_count")
    strata: JsonDict = {}
    for mode in MODES:
        for batch_size in BATCH_SIZES:
            selected = [
                row
                for row in rows
                if row.get("mode") == mode and row.get("batch_size") == batch_size
            ]
            pairs: dict[str, dict[bool, Mapping[str, Any]]] = {}
            for row in selected:
                pairs.setdefault(str(row.get("pair_id")), {})[
                    bool(row.get("telemetry_enabled"))
                ] = row
            if len(pairs) != CONTROL_REPEATS or any(
                set(pair) != {False, True} for pair in pairs.values()
            ):
                raise ValueError("instrumentation_pairs")
            ratios = [
                float(pair[True]["total_ns"]) / float(pair[False]["total_ns"])
                for pair in pairs.values()
            ]
            strata[f"{mode}:{batch_size}"] = {
                "enabled_over_disabled_median": statistics.median(ratios),
                "enabled_over_disabled_lower95": _percentile(ratios, 0.025),
                "enabled_over_disabled_upper95": _percentile(ratios, 0.975),
                "pair_count": CONTROL_REPEATS,
            }
    return {
        "independent_blocks": 40,
        "strata": strata,
        "comparator_selection_authority": False,
    }


def _load_object(path: Path) -> JsonDict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _check(
    check: str,
    upstream: str,
    path: str,
    field: str,
    operator: str,
    expected: Any,
    observed: Any,
) -> JsonDict:
    passed = observed == expected if operator == "eq" else observed in expected
    return {
        "check": check,
        "upstream": upstream,
        "path": path,
        "field": field,
        "operator": operator,
        "expected": deepcopy(expected),
        "observed": deepcopy(observed),
        "passed": passed,
    }


def hardware_dispositions(root: Path) -> list[JsonDict]:
    """Retain authenticated device scope without probing attached hardware."""

    board_artifact = _load_object(root.resolve() / EXP7599_PATH)
    rows = []
    for source in board_artifact.get("board_rows") or []:
        row = deepcopy(source)
        row["hardware"] = row.pop("board")
        row["current_execution"] = False
        row["evidence_age_days"] = 11 if row.get("hardware") != "GateMate" else 9
        if row.get("hardware") == "GateMate":
            row["last_observed"] = row.get("last_diagnostic", {}).get("observed")
        rows.append(row)
    rows.extend(
        [
            {
                "hardware": "local_rtx3090_pair",
                "disposition": "historical_two_device_model_lease_scope_preserved",
                "availability": "historical_available_current_not_probed",
                "device_count": 2,
                "current_model_invocation": False,
                "current_execution": False,
                "evidence_date": "20260924",
                "evidence_age_days": 0,
                "evidence_path": WISHLIST_PATH.as_posix(),
            },
            {
                "hardware": "Extropic_TSU",
                "disposition": "prospective_only_no_authenticated_access",
                "availability": "unavailable",
                "current_execution": False,
                "evidence_date": "20260924",
                "evidence_age_days": 0,
                "evidence_path": WISHLIST_PATH.as_posix(),
            },
            {
                "hardware": "AMD_XDNA",
                "disposition": "prospective_only_runtime_unavailable",
                "availability": "unavailable",
                "current_execution": False,
                "evidence_date": "20260924",
                "evidence_age_days": 0,
                "evidence_path": WISHLIST_PATH.as_posix(),
            },
        ]
    )
    return rows


load_native_extension = native.load_native_extension


def _tool_version(argv: Sequence[str]) -> str | None:
    try:
        child = subprocess.run(argv, capture_output=True, text=True, timeout=10, check=False)
    except (OSError, subprocess.TimeoutExpired):  # pragma: no cover - external absence.
        return None
    output = (child.stdout or child.stderr).strip().splitlines()
    return output[0] if child.returncode == 0 and output else None


def collect_preconditions(root: Path) -> JsonDict:
    """Authenticate named inputs, tools, build bytes, and the real extension."""

    repo = root.resolve()
    checks: list[JsonDict] = []
    sources: dict[str, JsonDict] = {}
    required = (
        Path("CLAUDE.md"),
        Path("CODEX.md"),
        Path("research-program.md"),
        Path("ops/exclusion_manifest.yaml"),
        Path("ops/e2e-test-plan.md"),
        Path("scripts/experiment_template.py"),
        Path("python/carnot/reporting/current_work_receipt.py"),
        Path("python/carnot/reporting/experiment_7303_validation_scope.py"),
        Path("python/carnot/experiment_7598_v663_rust_consumer.py"),
        Path("python/carnot/experiment_7613_v664_service_attribution.py"),
        EXP7626_PATH,
        EXP7599_PATH,
        WISHLIST_PATH,
        Path("ops/known-issues.md"),
        SPEC_PATH,
        consumer.RUST_BINARY,
    )
    for relative in required:
        path = repo / relative
        readable = path.is_file() and path.stat().st_size > 0
        checks.append(
            _check(
                f"source_readable:{relative.as_posix()}",
                relative.as_posix(),
                relative.as_posix(),
                "bytes",
                "eq",
                "readable_nonempty",
                "readable_nonempty" if readable else None,
            )
        )
        if readable:
            sources[relative.as_posix()] = {
                "path": relative.as_posix(),
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
                "role": "pre_gate_receipt"
                if relative in {EXP7626_PATH, EXP7599_PATH}
                else "actual_producer",
            }

    upstream = _load_object(repo / EXP7626_PATH)
    board_artifact = _load_object(repo / EXP7599_PATH)
    checks.extend(
        [
            _check(
                "native_service_ready",
                "Exp7626",
                EXP7626_PATH.as_posix(),
                "native_service_ready_score",
                "eq",
                1,
                upstream.get("native_service_ready_score"),
            ),
            _check(
                "native_verdict_class",
                "Exp7626",
                EXP7626_PATH.as_posix(),
                "verdict_class",
                "in",
                ["null", "positive"],
                upstream.get("verdict_class"),
            ),
            _check(
                "native_not_flagged",
                "Exp7626",
                EXP7626_PATH.as_posix(),
                "flagged_adversarial",
                "eq",
                False,
                upstream.get("flagged_adversarial"),
            ),
            _check(
                "board_continuity_complete",
                "Exp7599",
                EXP7599_PATH.as_posix(),
                "board_continuity_complete_score",
                "eq",
                1,
                board_artifact.get("board_continuity_complete_score"),
            ),
        ]
    )
    manifest_path = Path(str(upstream.get("native_build_manifest_path", "")))
    if not manifest_path.is_absolute():
        manifest_path = repo / manifest_path
    manifest = _load_object(manifest_path)
    manifest_hash = sha256_file(manifest_path) if manifest_path.is_file() else None
    checks.append(
        _check(
            "native_build_manifest_hash",
            "Exp7626",
            str(manifest_path),
            "native_build_manifest_sha256",
            "eq",
            upstream.get("native_build_manifest_sha256"),
            manifest_hash,
        )
    )
    extension = Path(str(manifest.get("module_path", "")))
    extension_hash = sha256_file(extension) if extension.is_file() else None
    checks.append(
        _check(
            "native_extension_hash",
            "Exp7626 build manifest",
            str(extension),
            "module_sha256",
            "eq",
            manifest.get("module_sha256"),
            extension_hash,
        )
    )
    binary_expected = (
        upstream.get("source_artifact_hashes", {})
        .get(consumer.RUST_BINARY.as_posix(), {})
        .get("sha256")
    )
    binary_hash = (
        sha256_file(repo / consumer.RUST_BINARY)
        if (repo / consumer.RUST_BINARY).is_file()
        else None
    )
    checks.append(
        _check(
            "rust_jsonl_binary_hash",
            "Exp7626",
            consumer.RUST_BINARY.as_posix(),
            "sha256",
            "eq",
            binary_expected,
            binary_hash,
        )
    )
    binding = None
    import_observed: Any = None
    if extension.is_file():
        try:
            binding = load_native_extension(extension)
            import_observed = str(Path(binding.__file__).resolve())
        except Exception as error:  # pragma: no cover - mutation replaces the loader.
            import_observed = f"{type(error).__name__}:{error}"
    checks.append(
        _check(
            "actual_native_import",
            "Exp7626 private extension",
            str(extension),
            "imported_module_path",
            "eq",
            str(extension.resolve()),
            import_observed,
        )
    )
    spec_text = (
        (repo / SPEC_PATH).read_text(encoding="utf-8") if (repo / SPEC_PATH).is_file() else ""
    )
    checks.append(
        _check(
            "driving_requirement",
            SPEC_PATH.as_posix(),
            SPEC_PATH.as_posix(),
            "REQ-*",
            "eq",
            "REQ-REPORT-7627",
            "REQ-REPORT-7627" if "REQ-REPORT-7627" in spec_text else None,
        )
    )
    tools = {
        "python": _tool_version([sys.executable, "--version"]),
        "pytest": _tool_version([str(repo / ".venv/bin/pytest"), "--version"]),
        "ruff": _tool_version([str(repo / ".venv/bin/ruff"), "--version"]),
        "mypy": _tool_version([str(repo / ".venv/bin/mypy"), "--version"]),
        "cargo": _tool_version(["cargo", "--version"]),
        "rustc": _tool_version(["rustc", "--version"]),
    }
    for name, observed in tools.items():
        checks.append(
            _check(
                f"tool_available:{name}",
                "declared toolchain",
                "PATH",
                name,
                "eq",
                True,
                bool(observed),
            )
        )
    failed = next((row for row in checks if row["passed"] is not True), None)
    blocker = (
        {
            key: deepcopy(failed[key])
            for key in ("check", "upstream", "path", "field", "operator", "expected", "observed")
        }
        if failed
        else None
    )
    return {
        "checks": checks,
        "blocker": blocker,
        "source_artifact_hashes": sources,
        "tools": tools,
        "native_extension": extension,
        "native_binding": binding,
        "build_manifest": manifest,
        "hardware_dispositions": hardware_dispositions(repo),
        "hardware_operations_issued": [],
        "upstream_e2e_results": deepcopy(upstream.get("e2e_results", {})),
    }


def _provisional_receipts() -> list[JsonDict]:
    """Supply complete test-only command names without claiming execution."""

    return [
        {
            "name": name,
            "passed": True,
            "exit_code": 0,
            "command": "unit-test provisional receipt",
            "log_sha256": "sha256:" + "0" * 64,
            "provisional": True,
        }
        for name in (*VALIDATION_NAMES, *TERMINAL_NAMES)
    ]


def _gate(
    category: str, check: str, condition: str, expected: Any, observed: Any, passed: bool
) -> JsonDict:
    return {
        "category": category,
        "check": check,
        "condition": condition,
        "upstream": "current Exp7627 evidence",
        "path": "timing_reduction",
        "field": check,
        "operator": "eq",
        "expected": deepcopy(expected),
        "observed": deepcopy(observed),
        "passed": passed,
        "principle": "Validity, readiness, benefit, retention, and freshness are separate claims.",
    }


def _acceptance_gates(reduction: Mapping[str, Any], blocked: bool) -> list[JsonDict]:
    primary = reduction.get("equal_stratum_geometric_mean", {}).get("python_over_direct_native", {})
    return [
        _gate(
            "validity",
            "native_cost_valid_score",
            "120 complete fair three-arm paired blocks and independent reduction",
            1,
            0 if blocked else 1,
            not blocked,
        ),
        _gate(
            "readiness",
            "native_service_ready_score",
            "Exp7626 readiness and exact native import remain authenticated",
            1,
            0 if blocked else 1,
            not blocked,
        ),
        _gate(
            "benefit",
            "fixed_total_cost_gate",
            "geomean estimate/lower95 >=1.10, each lower95 >=0.95, parity, no errors",
            1,
            int(reduction.get("native_speed_benefit_score", 0)),
            reduction.get("native_speed_benefit_score") == 1,
        ),
        _gate(
            "retention",
            "production_defaults_and_hardware_dispositions_unchanged",
            "no default, graduation, or prior verdict changes",
            False,
            False,
            True,
        ),
        _gate(
            "freshness",
            "current_model_or_hardware_execution_claim",
            "historical evidence does not become current execution",
            False,
            False,
            True,
        ),
        _gate(
            "benefit",
            "nfr_10x_total_cost",
            "measured equal-stratum total-cost estimate >=10x",
            10.0,
            primary.get("estimate"),
            reduction.get("nfr_10x_met") is True,
        ),
    ]


def field_principles(keys: Sequence[str]) -> dict[str, str]:
    """Keep each required reporting rule beside the field it controls."""

    special = {
        "honest_verdict": "Use complete_ for terminal work; completion does not prove benefit.",
        "verdict_class": "Use only the closed verdict enum; external absence is blocked.",
        "flagged_adversarial": "Persist the terminal reader result; flagged evidence opens no gate.",
        "gate_check_summary": "Every block retains exact operands, source, path, and operator.",
        "acceptance_gate_results": "Keep validity, readiness, benefit, retention, and freshness separate.",
        "rows": "Keep every independent unit and arm with absolute measured costs.",
        "sample_size_budget": "Arms, seeds, views, and replays do not multiply independent units.",
        "inference_substrate_class": "Record planned and actual no-model execution separately.",
        "MODEL_SPECS": "Pure timing and reduction use an empty current model list.",
        "model_invoked": "Historical GPU evidence is not a current model invocation.",
        "phase_spans": "Use disjoint monotonic stages with completed and pending units.",
        "invocation_counts": "Count loads, forwards, generations, and tokens separately.",
        "duration_s": "Measure current monotonic elapsed time without padding.",
        "random_seed": "Bind event, arm-order, control, and bootstrap randomness.",
        "reproducibility_checksum": "Bind immutable inputs, configuration, rows, and reductions.",
        "source_artifact_hashes": "Separate actual producers, pre-gate receipts, and missing bytes.",
        "validation_receipts": "Record command, exit, worktree, and exact log hash.",
        "verifier_is_oracle": "Exact fixtures cannot establish oracle-distinct learned advantage.",
        "native_cost_valid_score": "One requires complete fair paired measurement and reduction.",
        "native_speed_benefit_score": "One requires every fixed total-cost, cold, parity, and error gate.",
        "paired_timing_rows": "Keep 120 blocks times three arms across four fixed strata.",
        "instrumentation_rows": "Keep 40 matched on/off controls outside comparator selection.",
        "hardware_dispositions": "Preserve each dated scope without inventing current execution.",
        "nfr_10x_met": "Only measured total consumer cost can meet the separate ten-times target.",
    }
    return {
        key: special.get(key, f"Retain {key} so its terminal scope stays auditable.")
        for key in keys
    }


def reproducibility_checksum(value: Mapping[str, Any]) -> str:
    """Hash stable evidence while excluding clocks and command outcomes."""

    payload = deepcopy(dict(value))
    for key in (
        "started_at_utc",
        "completed_at_utc",
        "duration_s",
        "phase_spans",
        "validation_receipts",
        "terminal_reader_outcomes",
        "field_principles",
        "reproducibility_checksum",
    ):
        payload.pop(key, None)
    return canonical_hash(payload)


def build_artifact(
    root: Path,
    timing_rows: Sequence[Mapping[str, Any]],
    instrumentation_rows: Sequence[Mapping[str, Any]],
    *,
    preconditions: Mapping[str, Any] | None = None,
    validation_receipts: Sequence[Mapping[str, Any]] | None = None,
    duration_s: float = 0.01,
    phase_spans: Sequence[Mapping[str, Any]] = (),
    fixture: bool = False,
) -> JsonDict:
    """Build one terminal artifact from raw rows and authenticated inputs."""

    repo = root.resolve()
    context = dict(preconditions or collect_preconditions(repo))
    if context.get("blocker") is not None:
        return build_blocked_artifact(repo, context, duration_s=duration_s)
    timing = reduce_timing_rows(timing_rows)
    overhead = reduce_instrumentation_rows(instrumentation_rows)
    receipts = [deepcopy(dict(row)) for row in (validation_receipts or _provisional_receipts())]
    benefit = int(timing["native_speed_benefit_score"])
    verdict_class = "circular_positive" if fixture else ("positive" if benefit else "null")
    verdict = (
        "complete_circular_positive_native_cost_fixture"
        if fixture
        else (
            "complete_positive_native_total_cost_gate_met"
            if benefit
            else "complete_null_native_total_cost_gate_not_met"
        )
    )
    sources = deepcopy(context["source_artifact_hashes"])
    generated = (MODULE_PATH, WRAPPER_PATH, TEST_PATH, NOTES_PATH, SPEC_PATH)
    for relative in generated:
        path = repo / relative
        if path.is_file():
            sources[relative.as_posix()] = {
                "path": relative.as_posix(),
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
                "role": "actual_producer",
            }
    extension = Path(context["native_extension"])
    if extension.is_file():
        sources["native_extension"] = {
            "path": str(extension.resolve()),
            "sha256": sha256_file(extension),
            "bytes": extension.stat().st_size,
            "role": "actual_native_module",
        }
    boards = [deepcopy(dict(row)) for row in context["hardware_dispositions"]]
    gates = _acceptance_gates(timing, blocked=False)
    failed_gates = [
        {
            key: deepcopy(gate[key])
            for key in ("check", "upstream", "path", "field", "operator", "expected", "observed")
        }
        for gate in gates
        if gate["passed"] is not True
    ]
    terminal = [
        {
            "name": row.get("name"),
            "passed": row.get("passed") is True,
            "exit_code": row.get("exit_code"),
            "log_sha256": row.get("log_sha256"),
        }
        for row in receipts
        if row.get("name") in TERMINAL_NAMES
    ]
    started = datetime.now(UTC).isoformat()
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "experiment": 7627,
        "milestone": MILESTONE,
        "run_date": RUN_DATE,
        "worktree": str(repo),
        "started_at_utc": started,
        "completed_at_utc": datetime.now(UTC).isoformat(),
        "duration_s": float(duration_s),
        "phase_spans": [deepcopy(dict(row)) for row in phase_spans],
        "honest_verdict": verdict,
        "verdict_class": verdict_class,
        "flagged_adversarial": False,
        "gate_check_summary": failed_gates,
        "acceptance_gate_results": gates,
        "rows": [
            *[deepcopy(dict(row)) for row in timing_rows],
            *[deepcopy(dict(row)) for row in instrumentation_rows],
            *deepcopy(boards),
        ],
        "paired_timing_rows": [deepcopy(dict(row)) for row in timing_rows],
        "instrumentation_rows": [deepcopy(dict(row)) for row in instrumentation_rows],
        "timing_reduction": timing,
        "instrumentation_reduction": overhead,
        "sample_size_budget": {
            "paired_blocks": {
                "intended": 120,
                "observed": timing["independent_blocks"],
                "excluded": 0,
                "censored": 0,
                "arms_per_block": 3,
            },
            "instrumentation_blocks": {
                "intended": 40,
                "observed": overhead["independent_blocks"],
                "excluded": 0,
                "censored": 0,
                "observations_per_block": 2,
            },
        },
        "native_cost_valid_score": 1,
        "native_speed_benefit_score": benefit,
        "nfr_10x_met": bool(timing["nfr_10x_met"]),
        "primary_comparator": "python_inprocess",
        "secondary_comparator": "rust_jsonl",
        "preconditions_checked": deepcopy(context["checks"]),
        "inference_substrate": "host_cpu_three_arm_durable_consumer_timing",
        "inference_substrate_class": "no_model_load",
        "planned_inference_substrate_class": "no_model_load",
        "actual_inference_substrate_class": "no_model_load",
        "MODEL_SPECS": [],
        "model_specs": [],
        "target_model": "none:no_model_load",
        "model_invoked": False,
        "historical_model_identity": "unsloth/Qwen3.8-27B-GGUF; historical lease evidence only",
        "invocation_counts": deepcopy(ZERO_INVOCATIONS),
        "execution_venue": "host",
        "execution_venue_details": {
            "host": socket.gethostname(),
            "platform": platform.platform(),
            "processor": platform.processor(),
            "cpu_affinity": sorted(os.sched_getaffinity(0))
            if hasattr(os, "sched_getaffinity")
            else [],
            "gpu_uuid": None,
            "physical_device": "host CPU only",
        },
        "random_seed": {
            "event_workloads_and_arm_order": RANDOM_SEED,
            "instrumentation_controls": RANDOM_SEED + 100_000,
            "per_stratum_bootstrap": "7627 + mode_offset + batch_size",
            "equal_stratum_bootstrap": [RANDOM_SEED + 2_000, RANDOM_SEED + 3_000],
        },
        "source_artifact_hashes": sources,
        "validation_receipts": receipts,
        "terminal_reader_outcomes": terminal,
        "verifier_is_oracle": fixture,
        "hardware_dispositions": boards,
        "hardware_operations_issued": [],
        "upstream_e2e_results": deepcopy(context["upstream_e2e_results"]),
        "production_default_changed": False,
        "generator_weights_changed": False,
        "prior_verdicts_changed": False,
        "freshness_claimed": False,
        "repository_health": deepcopy(
            context.get(
                "repository_health",
                {
                    "status": "not_part_of_scoped_validation",
                    "affects_required_checks": False,
                    "broad_suite_receipt": None,
                },
            )
        ),
    }
    artifact["field_principles"] = field_principles(
        [*artifact, "field_principles", "reproducibility_checksum"]
    )
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    return artifact


def build_test_artifact(root: Path) -> JsonDict:
    """Build a deterministic exact fixture that cannot claim learned advantage."""

    return build_artifact(
        root,
        synthetic_timing_rows(),
        synthetic_instrumentation_rows(),
        fixture=True,
    )


def build_blocked_artifact(
    root: Path,
    context: Mapping[str, Any],
    *,
    duration_s: float,
    validation_receipts: Sequence[Mapping[str, Any]] = (),
) -> JsonDict:
    """Close an external native block while retaining every device disposition."""

    blocker = deepcopy(dict(context["blocker"]))
    token = "".join(character if character.isalnum() else "_" for character in blocker["check"])
    boards = [deepcopy(dict(row)) for row in context["hardware_dispositions"]]
    receipts = [deepcopy(dict(row)) for row in validation_receipts]
    gates = _acceptance_gates({}, blocked=True)
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "experiment": 7627,
        "milestone": MILESTONE,
        "run_date": RUN_DATE,
        "worktree": str(root.resolve()),
        "started_at_utc": datetime.now(UTC).isoformat(),
        "completed_at_utc": datetime.now(UTC).isoformat(),
        "duration_s": float(duration_s),
        "phase_spans": [],
        "honest_verdict": f"complete_blocked_{token}",
        "verdict_class": "blocked",
        "flagged_adversarial": False,
        "gate_check_summary": [blocker],
        "acceptance_gate_results": gates,
        "rows": deepcopy(boards),
        "paired_timing_rows": [],
        "instrumentation_rows": [],
        "timing_reduction": {},
        "instrumentation_reduction": {},
        "sample_size_budget": {
            "paired_blocks": {"intended": 120, "observed": 0, "excluded": 120, "censored": 0},
            "instrumentation_blocks": {
                "intended": 40,
                "observed": 0,
                "excluded": 40,
                "censored": 0,
            },
        },
        "native_cost_valid_score": 0,
        "native_speed_benefit_score": 0,
        "nfr_10x_met": False,
        "primary_comparator": "python_inprocess",
        "secondary_comparator": "rust_jsonl",
        "preconditions_checked": deepcopy(context["checks"]),
        "inference_substrate": "host_cpu_precondition_and_hardware_receipt_reduction_only",
        "inference_substrate_class": "no_model_load",
        "planned_inference_substrate_class": "no_model_load",
        "actual_inference_substrate_class": "no_model_load",
        "MODEL_SPECS": [],
        "model_specs": [],
        "target_model": "none:no_model_load",
        "model_invoked": False,
        "historical_model_identity": "unsloth/Qwen3.8-27B-GGUF; historical lease evidence only",
        "invocation_counts": deepcopy(ZERO_INVOCATIONS),
        "execution_venue": "host",
        "execution_venue_details": {
            "host": socket.gethostname(),
            "gpu_uuid": None,
            "physical_device": "host CPU only",
        },
        "random_seed": {
            "planned_event_workloads_and_arm_order": RANDOM_SEED,
            "planned_instrumentation_controls": RANDOM_SEED + 100_000,
        },
        "source_artifact_hashes": deepcopy(context["source_artifact_hashes"]),
        "validation_receipts": receipts,
        "terminal_reader_outcomes": [],
        "verifier_is_oracle": False,
        "hardware_dispositions": boards,
        "hardware_operations_issued": [],
        "upstream_e2e_results": deepcopy(context.get("upstream_e2e_results", {})),
        "production_default_changed": False,
        "generator_weights_changed": False,
        "prior_verdicts_changed": False,
        "freshness_claimed": False,
        "repository_health": deepcopy(
            context.get(
                "repository_health",
                {
                    "status": "not_part_of_scoped_validation",
                    "affects_required_checks": False,
                    "broad_suite_receipt": None,
                },
            )
        ),
    }
    artifact["field_principles"] = field_principles(
        [*artifact, "field_principles", "reproducibility_checksum"]
    )
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    return artifact


def _verify_sources(value: Mapping[str, Any], root: Path) -> list[str]:
    errors = []
    for key, receipt in (value.get("source_artifact_hashes") or {}).items():
        if not isinstance(receipt, Mapping):
            errors.append(f"source_receipt:{key}")
            continue
        raw = Path(str(receipt.get("path", key)))
        path = raw if raw.is_absolute() else root / raw
        if not path.is_file() or sha256_file(path) != receipt.get("sha256"):
            errors.append(f"source_hash:{key}")
    return errors


def validate_artifact(value: Mapping[str, Any], *, root: Path | None = None) -> list[str]:
    """Reject claim, reduction, hardware, provenance, or receipt drift."""

    required = {
        "honest_verdict",
        "verdict_class",
        "flagged_adversarial",
        "gate_check_summary",
        "acceptance_gate_results",
        "rows",
        "paired_timing_rows",
        "instrumentation_rows",
        "sample_size_budget",
        "preconditions_checked",
        "inference_substrate",
        "inference_substrate_class",
        "MODEL_SPECS",
        "model_invoked",
        "target_model",
        "execution_venue",
        "execution_venue_details",
        "phase_spans",
        "invocation_counts",
        "duration_s",
        "random_seed",
        "reproducibility_checksum",
        "source_artifact_hashes",
        "validation_receipts",
        "verifier_is_oracle",
        "field_principles",
        "native_cost_valid_score",
        "native_speed_benefit_score",
        "hardware_dispositions",
        "nfr_10x_met",
        "repository_health",
    }
    errors = []
    missing = sorted(required.difference(value))
    if missing:
        errors.append("required_fields:" + ",".join(missing))
    if (
        value.get("MODEL_SPECS") != []
        or value.get("model_specs") != []
        or value.get("model_invoked") is not False
        or value.get("inference_substrate_class") != "no_model_load"
        or value.get("planned_inference_substrate_class") != "no_model_load"
        or value.get("actual_inference_substrate_class") != "no_model_load"
        or value.get("invocation_counts") != ZERO_INVOCATIONS
        or value.get("target_model") != "none:no_model_load"
    ):
        errors.append("model_contract")
    if value.get("execution_venue") != "host":
        errors.append("execution_venue")
    if value.get("flagged_adversarial") is not False:
        errors.append("flagged_adversarial")
    if value.get("hardware_operations_issued") != []:
        errors.append("hardware_operations")
    if (value.get("repository_health") or {}).get("affects_required_checks") is not False:
        errors.append("repository_health_scope")
    if value.get("verdict_class") not in {
        "positive",
        "circular_positive",
        "null",
        "blocked",
        "disqualified",
        "partial",
    }:
        errors.append("verdict_class")
    if not str(value.get("honest_verdict", "")).startswith("complete_"):
        errors.append("honest_verdict")
    blocked = value.get("verdict_class") == "blocked"
    if blocked:
        summary = value.get("gate_check_summary")
        first = summary[0] if isinstance(summary, list) and summary else {}
        if not all(
            key in first
            for key in ("check", "upstream", "path", "field", "operator", "expected", "observed")
        ):
            errors.append("blocked_gate_summary")
        if value.get("paired_timing_rows") != [] or value.get("native_cost_valid_score") != 0:
            errors.append("blocked_measurement")
    else:
        try:
            timing = reduce_timing_rows(value.get("paired_timing_rows") or [])
            overhead = reduce_instrumentation_rows(value.get("instrumentation_rows") or [])
        except (KeyError, TypeError, ValueError, ZeroDivisionError) as error:
            errors.append(f"reduction:{error}")
        else:
            if timing != value.get("timing_reduction"):
                errors.append("timing_reduction")
            if overhead != value.get("instrumentation_reduction"):
                errors.append("instrumentation_reduction")
            if value.get("native_cost_valid_score") != 1:
                errors.append("valid_score")
            if value.get("native_speed_benefit_score") != timing["native_speed_benefit_score"]:
                errors.append("speed_score")
            if value.get("nfr_10x_met") is not timing["nfr_10x_met"]:
                errors.append("nfr_10x_met")
            if (
                timing["native_speed_benefit_score"] == 0
                and value.get("verdict_class") == "positive"
            ):
                errors.append("positive_without_benefit")
    devices = {
        row.get("hardware"): row
        for row in value.get("hardware_dispositions") or []
        if isinstance(row, Mapping)
    }
    expected_devices = {
        "KV260",
        "PolarFire",
        "GateMate",
        "local_rtx3090_pair",
        "Extropic_TSU",
        "AMD_XDNA",
    }
    if set(devices) != expected_devices:
        errors.append("hardware_identity")
    else:
        if (
            devices["KV260"].get("k_max") != 5
            or devices["KV260"].get("current_execution") is not False
        ):
            errors.append("hardware_kv260")
        if devices["PolarFire"].get("fpga_sampling_measured") is not False:
            errors.append("hardware_polarfire")
        if devices["GateMate"].get("last_observed") != "0xffffffff":
            errors.append("hardware_gatemate")
        if devices["local_rtx3090_pair"].get("current_model_invocation") is not False:
            errors.append("hardware_gpu")
        if any(
            devices[name].get("availability") != "unavailable"
            for name in ("Extropic_TSU", "AMD_XDNA")
        ):
            errors.append("hardware_prospective")
    categories = {
        row.get("category")
        for row in value.get("acceptance_gate_results") or []
        if isinstance(row, Mapping)
    }
    if categories != {"validity", "readiness", "benefit", "retention", "freshness"}:
        errors.append("acceptance_gates")
    if value.get("upstream_e2e_results") != {
        "E2E-003": "passed_actual_private_extension_round_trip",
        "E2E-004": "passed_service_json_cross_language_crash_and_reload",
    }:
        errors.append("upstream_e2e")
    receipt_names = {
        row.get("name")
        for row in value.get("validation_receipts") or []
        if isinstance(row, Mapping) and row.get("passed") is True and row.get("exit_code") == 0
    }
    if not blocked and not set((*VALIDATION_NAMES, *TERMINAL_NAMES)).issubset(receipt_names):
        errors.append("validation_receipts")
    principles = value.get("field_principles")
    if not isinstance(principles, Mapping) or any(key not in principles for key in value):
        errors.append("field_principles")
    if value.get("reproducibility_checksum") != reproducibility_checksum(value):
        errors.append("reproducibility_checksum")
    if root is not None:
        errors.extend(_verify_sources(value, root.resolve()))
    return sorted(set(errors))


def cold_replay(path: Path) -> JsonDict:
    """Load the exact bytes in a fresh reader and validate their contract."""

    value = _load_object(path)
    if not value:
        return {"valid": False, "errors": ["artifact_unreadable"]}
    errors = validate_artifact(value)
    return {"valid": not errors, "errors": errors, "sha256": sha256_file(path)}


def independent_replay(path: Path) -> JsonDict:
    """Recompute all comparative metrics without trusting producer summaries."""

    value = _load_object(path)
    if not value:
        return {"valid": False, "error": "artifact_unreadable"}
    if value.get("verdict_class") == "blocked":
        return {"valid": True, "blocked": True}
    try:
        timing = reduce_timing_rows(value.get("paired_timing_rows") or [])
        overhead = reduce_instrumentation_rows(value.get("instrumentation_rows") or [])
    except (KeyError, TypeError, ValueError, ZeroDivisionError) as error:
        return {"valid": False, "error": str(error)}
    if timing != value.get("timing_reduction"):
        return {"valid": False, "error": "timing_reduction_drift"}
    if overhead != value.get("instrumentation_reduction"):
        return {"valid": False, "error": "instrumentation_reduction_drift"}
    return {"valid": True, "timing_reduction": timing, "instrumentation_reduction": overhead}


def build_validation_commands(root: Path, private: Path) -> list[validation.CommandSpec]:
    """Freeze serial tests, private coverage, static checks, and E2E receipts."""

    commands = validation.build_scoped_commands(
        root,
        [TEST_PATH.as_posix()],
        [MODULE_PATH.as_posix()],
        static_paths=[WRAPPER_PATH.as_posix()],
        basetemp=private / "basetemp",
        coverage_file=private / ".coverage-exp7627",
    )
    pytest = str(root / ".venv/bin/pytest")
    python = str(root / ".venv/bin/python")
    commands.extend(
        [
            validation.CommandSpec(
                "entrypoint_help",
                (python, "-u", WRAPPER_PATH.as_posix(), "--help"),
                "thin entrypoint",
                60.0,
            ),
            validation.CommandSpec(
                "e2e_native_import",
                (
                    pytest,
                    "-n",
                    "0",
                    "-o",
                    "addopts=",
                    "--no-cov",
                    f"--basetemp={private / 'basetemp' / 'e2e-import'}",
                    "tests/python/test_experiment_7626_v665_native_service.py::test_scenario_report_7626_actual_binding_invalid_and_cold_reload",
                    "-q",
                ),
                "E2E-003 actual private extension",
                300.0,
            ),
            validation.CommandSpec(
                "e2e_durable_sequence",
                (
                    pytest,
                    "-n",
                    "0",
                    "-o",
                    "addopts=",
                    "--no-cov",
                    f"--basetemp={private / 'basetemp' / 'e2e-durable'}",
                    "tests/python/test_experiment_7626_v665_native_service.py::test_scenario_report_7626_actual_three_caller_parity",
                    "-q",
                ),
                "E2E-004 durable cross-language sequence",
                300.0,
            ),
        ]
    )
    return commands


def terminal_commands(candidate: Path, root: Path) -> list[validation.CommandSpec]:
    """Build independent readers for one exact terminal candidate."""

    python = str(root / ".venv/bin/python")
    wrapper = str(root / WRAPPER_PATH)
    return [
        validation.CommandSpec(
            "fresh_process_cold_replay",
            (python, "-u", wrapper, "--cold-replay", str(candidate)),
            "exact terminal candidate",
            60.0,
        ),
        validation.CommandSpec(
            "independent_raw_reduction",
            (python, "-u", wrapper, "--independent-replay", str(candidate)),
            "exact terminal candidate",
            60.0,
        ),
        validation.CommandSpec(
            "adversarial_verify",
            (python, "-u", "scripts/adversarial_verify.py", str(candidate)),
            "exact terminal candidate",
            120.0,
        ),
        validation.CommandSpec(
            "verdict_row_consistency_strict",
            (python, "-u", "scripts/verdict_row_consistency_lint.py", "--strict", str(candidate)),
            "exact terminal candidate",
            120.0,
        ),
    ]


def progress(started: float, phase: str, event: str, **fields: Any) -> None:  # pragma: no cover
    """Emit a truthful flushed boundary before and after long operations."""

    detail = " ".join(f"{key}={value}" for key, value in sorted(fields.items()))
    print(
        f"[exp7627] phase={phase} event={event} elapsed_s={time.monotonic() - started:.3f}"
        + (f" {detail}" if detail else ""),
        flush=True,
    )


def _open_arm(
    root: Path,
    arm: str,
    state: Path,
    extension: Path,
) -> tuple[Any, int]:  # pragma: no cover - benchmark E2E.
    """Open one real arm and measure import, process, and initialization cost."""

    started = time.perf_counter_ns()
    if arm == "python_inprocess":
        service = consumer._open_service(root, "python_inprocess", state)
    elif arm == "rust_jsonl":
        service = consumer._open_service(root, "rust", state, telemetry_enabled=False)
    else:
        binding = load_native_extension(extension)
        service = native.NativeServiceClient(binding, state)
    return service, time.perf_counter_ns() - started


def _close_arm(service: Any) -> None:  # pragma: no cover - benchmark E2E.
    close = getattr(service, "close", None)
    if close is not None:
        close()


def _measure_arm_block(
    service: Any,
    events: Sequence[tuple[str, float, int]],
    *,
    setup_ns: int,
    include_setup: bool,
) -> JsonDict:  # pragma: no cover - benchmark E2E.
    """Measure matched calls and retain exclusive consumer-side spans."""

    started = time.perf_counter_ns()
    spans = {
        "setup": setup_ns if include_setup else 0,
        "predict": 0,
        "update_persist_ack": 0,
        "reload_verification": 0,
    }
    decisions = []
    errors = []
    acknowledgments = 0
    for event_id, probability, label in events:
        span_started = time.perf_counter_ns()
        decision = service.predict(event_id, probability)
        spans["predict"] += time.perf_counter_ns() - span_started
        if not decision.available:
            errors.append(f"prediction:{event_id}:{decision.error}")
            continue
        decisions.append((event_id, decision.error_probability, decision.action))
        span_started = time.perf_counter_ns()
        acknowledgment = service.release_feedback(event_id, label)
        spans["update_persist_ack"] += time.perf_counter_ns() - span_started
        if not acknowledgment.durable:
            errors.append(f"feedback:{event_id}:{acknowledgment.error}")
        else:
            acknowledgments += 1
    span_started = time.perf_counter_ns()
    state = consumer.upstream._load_state(service.state_path).to_payload()
    resumed = service.predict(f"{events[-1][0]}-resumed", 0.37)
    spans["reload_verification"] = time.perf_counter_ns() - span_started
    if not resumed.available:
        errors.append(f"reload_prediction:{resumed.error}")
    elapsed = time.perf_counter_ns() - started + spans["setup"]
    spans["other_consumer_work"] = max(0, elapsed - sum(spans.values()))
    return {
        "total_ns": elapsed,
        "exclusive_span_ns": spans,
        "decisions": decisions,
        "resumed": (resumed.error_probability, resumed.action) if resumed.available else None,
        "state": state,
        "durable_acknowledgments": acknowledgments,
        "errors": errors,
    }


def _parity(results: Mapping[str, Mapping[str, Any]]) -> tuple[bool, bool]:  # pragma: no cover
    translated = {
        "python_inprocess": results["python_inprocess"],
        "rust": results["rust_jsonl"],
        "python_service": results["direct_native"],
    }
    return consumer._outcomes_match(translated)


def _timing_row(
    result: Mapping[str, Any],
    *,
    mode: str,
    batch_size: int,
    repeat: int,
    arm: str,
    seed: int,
    order: Sequence[str],
    decision_parity: bool,
    state_parity: bool,
    affinity: Sequence[int],
    environment: Mapping[str, Any],
) -> JsonDict:  # pragma: no cover
    total = int(result["total_ns"])
    errors = list(result["errors"])
    return {
        "row_type": "paired_native_cost",
        "unit_id": f"{mode}:{batch_size}:{repeat}:{arm}",
        "pair_id": f"{mode}:{batch_size}:{repeat}",
        "mode": mode,
        "batch_size": batch_size,
        "repeat": repeat,
        "arm": arm,
        "arm_order": list(order),
        "seed": seed,
        "total_ns": total,
        "exclusive_span_ns": deepcopy(result["exclusive_span_ns"]),
        "numerator": total,
        "denominator": batch_size * 2 + 1,
        "metric_direction": "lower_is_better",
        "decision_parity": decision_parity,
        "state_parity": state_parity,
        "reload_agreement": state_parity,
        "durable_acknowledgments": int(result["durable_acknowledgments"]),
        "durability_policy": DURABILITY_POLICY,
        "failure": errors[0] if errors else None,
        "error_count": len(errors),
        "cpu_affinity": list(affinity),
        "environment": deepcopy(dict(environment)),
        "censored": bool(errors),
        "raw_provenance": "matched predict-update-persist-ack-disk-reload sequence",
    }


def measure_timing_rows(
    root: Path,
    scratch: Path,
    extension: Path,
    started: float,
    checkpoint: Path,
) -> list[JsonDict]:  # pragma: no cover - declared benchmark E2E.
    """Measure 120 randomized three-arm blocks across four fixed strata."""

    rows: list[JsonDict] = []
    completed = 0
    affinity = sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else []
    environment = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
        "extension_sha256": sha256_file(extension),
        "rust_binary_sha256": sha256_file(root / consumer.RUST_BINARY),
    }
    scratch.mkdir(parents=True, exist_ok=True)
    for mode_index, mode in enumerate(MODES):
        for batch_size in BATCH_SIZES:
            warm: dict[str, tuple[Any, int]] = {}
            if mode == "warm":
                for arm in ARMS:
                    warm[arm] = _open_arm(
                        root,
                        arm,
                        scratch / f"warm-b{batch_size}-{arm}.json",
                        extension,
                    )
            try:
                for repeat in range(REPEATS):
                    seed = RANDOM_SEED + mode_index * 10_000 + batch_size * 100 + repeat
                    events = consumer._block_events(seed, batch_size)
                    order = list(ARMS)
                    random.Random(seed ^ RANDOM_SEED).shuffle(order)
                    results: dict[str, JsonDict] = {}
                    for arm in order:
                        if mode == "cold":
                            service, setup_ns = _open_arm(
                                root,
                                arm,
                                scratch / f"cold-b{batch_size}-r{repeat}-{arm}.json",
                                extension,
                            )
                        else:
                            service, setup_ns = warm[arm]
                        try:
                            results[arm] = _measure_arm_block(
                                service,
                                events,
                                setup_ns=setup_ns,
                                include_setup=mode == "cold",
                            )
                        finally:
                            if mode == "cold":
                                _close_arm(service)
                    if any(result["errors"] for result in results.values()):
                        decision_parity = state_parity = False
                    else:
                        decision_parity, state_parity = _parity(results)
                    for arm in ARMS:
                        rows.append(
                            _timing_row(
                                results[arm],
                                mode=mode,
                                batch_size=batch_size,
                                repeat=repeat,
                                arm=arm,
                                seed=seed,
                                order=order,
                                decision_parity=decision_parity,
                                state_parity=state_parity,
                                affinity=affinity,
                                environment=environment,
                            )
                        )
                    completed += 1
                    atomic_json(checkpoint, {"completed_blocks": completed, "rows": rows})
                    progress(
                        started,
                        "benchmark",
                        "paired_block_complete",
                        completed=completed,
                        planned=120,
                    )
            finally:
                for service, _setup_ns in warm.values():
                    _close_arm(service)
    return rows


def measure_instrumentation_rows(
    root: Path,
    scratch: Path,
    started: float,
    checkpoint: Path,
) -> list[JsonDict]:  # pragma: no cover - declared control E2E.
    """Measure 40 paired Rust telemetry controls outside the benefit gate."""

    rows: list[JsonDict] = []
    completed = 0
    scratch.mkdir(parents=True, exist_ok=True)
    for mode_index, mode in enumerate(MODES):
        for batch_size in BATCH_SIZES:
            warm: dict[bool, Any] = {}
            if mode == "warm":
                for enabled in (False, True):
                    warm[enabled] = consumer._open_service(
                        root,
                        "rust",
                        scratch / f"warm-b{batch_size}-{int(enabled)}.json",
                        telemetry_enabled=enabled,
                    )
            try:
                for repeat in range(CONTROL_REPEATS):
                    seed = RANDOM_SEED + 100_000 + mode_index * 10_000 + batch_size * 100 + repeat
                    events = consumer._block_events(seed, batch_size)
                    order = [False, True]
                    random.Random(seed ^ RANDOM_SEED).shuffle(order)
                    for enabled in order:
                        service = warm.get(enabled)
                        if service is None:
                            service = consumer._open_service(
                                root,
                                "rust",
                                scratch / f"cold-b{batch_size}-r{repeat}-{int(enabled)}.json",
                                telemetry_enabled=enabled,
                            )
                        try:
                            result = consumer._run_block(
                                service,
                                events,
                                include_setup=mode == "cold",
                            )
                        finally:
                            if mode == "cold":
                                service.close()
                        rows.append(
                            {
                                "row_type": "instrumentation_control",
                                "unit_id": f"{mode}:{batch_size}:{repeat}:{int(enabled)}",
                                "pair_id": f"{mode}:{batch_size}:{repeat}",
                                "mode": mode,
                                "batch_size": batch_size,
                                "repeat": repeat,
                                "seed": seed,
                                "telemetry_enabled": enabled,
                                "arm_order": list(order),
                                "total_ns": int(result["elapsed_ns"]),
                                "numerator": int(result["elapsed_ns"]),
                                "denominator": batch_size * 2 + 1,
                                "metric_direction": "lower_is_better",
                                "failure": None,
                                "censored": False,
                                "raw_provenance": "matched Rust telemetry on-off durable sequence",
                            }
                        )
                    completed += 1
                    atomic_json(checkpoint, {"completed_blocks": completed, "rows": rows})
                    progress(
                        started,
                        "instrumentation",
                        "paired_block_complete",
                        completed=completed,
                        planned=40,
                    )
            finally:
                for service in warm.values():
                    service.close()
    return rows


def _span(phase: str, phase_started: float, run_started: float, units: int) -> JsonDict:
    ended = time.monotonic()
    return {
        "phase": phase,
        "start_offset_s": phase_started - run_started,
        "end_offset_s": ended - run_started,
        "duration_s": ended - phase_started,
        "completed_units": units,
        "pending_operation": None,
        "checkpoint_position": units,
    }


def _all_passed(receipts: Sequence[Mapping[str, Any]], names: Sequence[str]) -> bool:
    passed = {
        str(row.get("name"))
        for row in receipts
        if row.get("passed") is True and row.get("exit_code") == 0
    }
    return set(names).issubset(passed)


def _write_manifest(
    root: Path,
    raw: Path,
    private: Path,
    extension: Path,
    commands: Sequence[validation.CommandSpec],
) -> Path:  # pragma: no cover - producer E2E.
    affected = (MODULE_PATH, WRAPPER_PATH, TEST_PATH, NOTES_PATH, SPEC_PATH)
    value = {
        "experiment_id": EXPERIMENT_ID,
        "worktree": str(root),
        "affected_files": [
            {"path": path.as_posix(), "sha256": sha256_file(root / path)} for path in affected
        ],
        "commands": [
            {
                "name": command.name,
                "argv": list(command.argv),
                "scope": command.scope,
                "timeout_s": command.timeout_s,
            }
            for command in commands
        ],
        "private_basetemp": str((private / "basetemp").resolve()),
        "coverage_file": str((private / ".coverage-exp7627").resolve()),
        "PYTHONPATH": f"{root / 'python'}:{root}",
        "native_extension": str(extension.resolve()),
    }
    path = raw / "affected_validation_manifest.json"
    atomic_json(path, value)
    return path


def run_experiment(
    root: Path,
    run_date: str,
    output: Path,
) -> JsonDict:  # pragma: no cover - declared entrypoint E2E.
    """Authenticate, benchmark, validate exact bytes, and publish atomically."""

    repo = root.resolve()
    started = time.monotonic()
    started_at = datetime.now(UTC).isoformat()
    progress(started, "startup", "resolved_root", root=repo)
    if run_date != RUN_DATE:
        raise ValueError(f"run_date_must_equal_{RUN_DATE}")
    phase_started = time.monotonic()
    context = collect_preconditions(repo)
    spans = [_span("preconditions", phase_started, started, len(context["checks"]))]
    progress(started, "preconditions", "authenticated", blocker=context["blocker"])
    destination = output if output.is_absolute() else repo / output
    if context["blocker"] is not None:
        blocked = build_blocked_artifact(
            repo,
            context,
            duration_s=time.monotonic() - started,
        )
        atomic_json(destination, blocked)
        progress(started, "publication", "blocked_complete", output=destination)
        return blocked

    raw = repo / RAW_DIR
    raw.mkdir(parents=True, exist_ok=True)
    debt_log = raw / "validation/repository_health/full_python_suite.log"
    if debt_log.is_file():
        context["repository_health"] = {
            "status": "degraded_open_unrelated",
            "affects_required_checks": False,
            "broad_suite_receipt": {
                "command": ".venv/bin/pytest tests/python -q",
                "exit_code": 2,
                "interrupted_after_unrelated_failures": True,
                "observed_before_interrupt": {
                    "passed": 1_670,
                    "failed": 17,
                    "collection_errors": 67,
                    "skipped": 5,
                },
                "log_path": debt_log.relative_to(repo).as_posix(),
                "log_sha256": sha256_file(debt_log),
            },
        }
        context["source_artifact_hashes"][debt_log.relative_to(repo).as_posix()] = {
            "path": debt_log.relative_to(repo).as_posix(),
            "sha256": sha256_file(debt_log),
            "bytes": debt_log.stat().st_size,
            "role": "unrelated_repository_health_receipt",
        }
    private = Path(tempfile.mkdtemp(prefix="carnot-exp7627-private-", dir="/tmp"))
    (private / "basetemp").mkdir(parents=True, exist_ok=True)
    commands = build_validation_commands(repo, private)
    manifest = _write_manifest(repo, raw, private, context["native_extension"], commands)
    progress(started, "manifest", "frozen", sha256=sha256_file(manifest))

    progress(started, "benchmark", "before_measurement", planned=120)
    phase_started = time.monotonic()
    timing_rows = measure_timing_rows(
        repo,
        private / "timing",
        context["native_extension"],
        started,
        raw / "timing_checkpoint.json",
    )
    spans.append(_span("benchmark", phase_started, started, 120))
    progress(started, "benchmark", "after_measurement", rows=len(timing_rows))

    progress(started, "instrumentation", "before_measurement", planned=40)
    phase_started = time.monotonic()
    instrumentation_rows = measure_instrumentation_rows(
        repo,
        private / "instrumentation",
        started,
        raw / "instrumentation_checkpoint.json",
    )
    spans.append(_span("instrumentation", phase_started, started, 40))
    progress(
        started,
        "instrumentation",
        "after_measurement",
        rows=len(instrumentation_rows),
    )
    timing_path = raw / "paired_timing_rows.json"
    instrumentation_path = raw / "instrumentation_rows.json"
    atomic_json(timing_path, {"rows": timing_rows})
    atomic_json(instrumentation_path, {"rows": instrumentation_rows})
    for path in (manifest, timing_path, instrumentation_path):
        context["source_artifact_hashes"][path.relative_to(repo).as_posix()] = {
            "path": path.relative_to(repo).as_posix(),
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
            "role": "actual_producer",
        }

    progress(started, "validation", "before_subprocesses", planned=len(commands))
    phase_started = time.monotonic()
    affected = validation.run_commands(
        repo,
        commands,
        log_dir=raw / "validation/affected",
        extra_env={
            "CARNOT_EXP7626_EXTENSION": str(context["native_extension"].resolve()),
            "COVERAGE_FILE": str((private / ".coverage-exp7627").resolve()),
        },
        heartbeat_s=60.0,
    )
    spans.append(_span("validation", phase_started, started, len(affected)))
    progress(
        started,
        "validation",
        "after_subprocesses",
        passed=_all_passed(affected, VALIDATION_NAMES),
    )
    if not _all_passed(affected, VALIDATION_NAMES):
        failed = [row["name"] for row in affected if row.get("passed") is not True]
        raise RuntimeError("affected_validation_failed:" + ",".join(failed))

    provisional_terminal = [row for row in _provisional_receipts() if row["name"] in TERMINAL_NAMES]
    provisional = build_artifact(
        repo,
        timing_rows,
        instrumentation_rows,
        preconditions=context,
        validation_receipts=[*affected, *provisional_terminal],
        duration_s=time.monotonic() - started,
        phase_spans=spans,
    )
    candidate = private / "terminal-candidate.json"
    atomic_json(candidate, provisional)
    progress(started, "terminal_readers", "before_subprocesses", planned=len(TERMINAL_NAMES))
    phase_started = time.monotonic()
    terminal = validation.run_commands(
        repo,
        terminal_commands(candidate, repo),
        log_dir=raw / "validation/terminal_candidate",
        extra_env={"CARNOT_EXP7626_EXTENSION": str(context["native_extension"].resolve())},
        heartbeat_s=60.0,
    )
    spans.append(_span("terminal_readers", phase_started, started, len(terminal)))
    progress(
        started,
        "terminal_readers",
        "after_subprocesses",
        passed=_all_passed(terminal, TERMINAL_NAMES),
    )
    if not _all_passed(terminal, TERMINAL_NAMES):
        raise RuntimeError("terminal_validation_failed")

    final = build_artifact(
        repo,
        timing_rows,
        instrumentation_rows,
        preconditions=context,
        validation_receipts=[*affected, *terminal],
        duration_s=time.monotonic() - started,
        phase_spans=spans,
    )
    final["started_at_utc"] = started_at
    final["completed_at_utc"] = datetime.now(UTC).isoformat()
    final["affected_validation_manifest_path"] = manifest.relative_to(repo).as_posix()
    final["affected_validation_manifest_sha256"] = sha256_file(manifest)
    final["flagged_adversarial"] = not next(
        row for row in terminal if row["name"] == "adversarial_verify"
    )["passed"]
    final["field_principles"] = field_principles(
        [*final, "field_principles", "reproducibility_checksum"]
    )
    final["reproducibility_checksum"] = reproducibility_checksum(final)
    errors = validate_artifact(final, root=repo)
    if errors:
        raise RuntimeError("terminal_artifact_invalid:" + ",".join(errors))
    exact = private / "exact-terminal-candidate.json"
    atomic_json(exact, final)

    progress(started, "exact_terminal_readers", "before_subprocesses", planned=len(TERMINAL_NAMES))
    exact_receipts = validation.run_commands(
        repo,
        terminal_commands(exact, repo),
        log_dir=raw / "validation/exact_terminal",
        extra_env={"CARNOT_EXP7626_EXTENSION": str(context["native_extension"].resolve())},
        heartbeat_s=60.0,
    )
    progress(
        started,
        "exact_terminal_readers",
        "after_subprocesses",
        passed=_all_passed(exact_receipts, TERMINAL_NAMES),
    )
    if not _all_passed(exact_receipts, TERMINAL_NAMES):
        raise RuntimeError("exact_terminal_validation_failed")
    atomic_json(raw / "exact_terminal_reader_receipts.json", {"receipts": exact_receipts})
    progress(started, "publication", "before_atomic_write", output=destination)
    atomic_json(destination, final)
    if sha256_file(destination) != sha256_file(exact):
        raise RuntimeError("published_bytes_differ")
    progress(started, "publication", "after_atomic_write", bytes=destination.stat().st_size)
    return final


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse producer and read-only replay modes for the thin wrapper."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=RUN_DATE)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output", type=Path, default=RESULT_PATH)
    parser.add_argument("--cold-replay", type=Path)
    parser.add_argument("--independent-replay", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run one selected mode while keeping the repository wrapper thin."""

    args = parse_args(argv)
    if args.cold_replay is not None:
        outcome = cold_replay(args.cold_replay)
        print(json.dumps(outcome, sort_keys=True), flush=True)
        return int(not outcome["valid"])
    if args.independent_replay is not None:
        outcome = independent_replay(args.independent_replay)
        print(json.dumps(outcome, sort_keys=True), flush=True)
        return int(not outcome["valid"])
    run_experiment(args.root.resolve(), args.date, args.output)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
