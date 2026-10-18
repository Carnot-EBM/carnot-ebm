"""Audit rare-event observability in the authenticated Exp7216 traces.

This module only aggregates durable upstream evidence. It does not run a new
sampler or recompute the exact law. The distinction matters because a zero-hit
Markov chain is not proof that the exact event probability is zero.

Spec: REQ-SAMPLER-7229 and SCENARIO-SAMPLER-7229-*.
"""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import tempfile
import time
from typing import Any, Mapping, Sequence

import numpy as np
import yaml

from carnot import experiment_7216_v635_down_up_quality as exp7216
from carnot.samplers import experiment_7215_down_up as down_up


JsonDict = dict[str, Any]

RUN_DATE = "20260912"
TASK_ID = "exp7229-rare-event-audit"
MILESTONE = "2026.09.636"
RESULT_PATH = Path("results/experiment_7229_v636_rare_event_audit.json")
UPSTREAM_PATH = Path("results/experiment_7216_v635_down_up_quality.json")
TRACE_PATH = Path("results/checkpoints/experiment_7216_v635_down_up_quality_traces.jsonl.gz")
SPEC_PATH = Path("openspec/capabilities/samplers/spec.md")
ROADMAP_PATH = Path("research-roadmap.yaml")
EXCLUSION_PATH = Path("ops/exclusion_manifest.yaml")
CHECKPOINT_DIR = Path("results/checkpoints")
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"
INFERENCE_SUBSTRATE_CLASS = "aggregation"
MODEL_SPECS: list[JsonDict] = []
IID_REFERENCE_ONLY = True
NORMAL_95_SQUARED = 1.959963984540054**2
MAX_RETAINED_PER_CHAIN = 10_000_000
RELATIVE_HALF_WIDTH = 0.25
EXPECTED_OBSERVABLE_ROWS = len(exp7216.CELLS) * len(exp7216.GRAPH_SEEDS) * len(exp7216.ARMS) * 4

EXPECTED_TASK_CONTRACT = {
    "id": TASK_ID,
    "milestone": MILESTONE,
    "deliverable": str(RESULT_PATH),
    "prior_failures": [
        {
            "experiment_id": "exp7216-down-up-quality",
            "verdict": (
                "complete_null: all prespecified panels were measured, but the joint "
                "exact-fidelity and cost-adjusted down-up quality gate did not pass."
            ),
            "addressed_by": (
                "Use a read-only exact-probability/observability diagnosis of the null; do not "
                "rerun down-up throughput or change its failed acceptance criteria."
            ),
            "retire_if_same_verdict": True,
        }
    ],
}

REQUIRED_SOURCE_PATHS = (
    Path("AGENTS.md"),
    Path("CLAUDE.md"),
    Path("CODEX.md"),
    Path("research-program.md"),
    Path("research-references.md"),
    ROADMAP_PATH,
    EXCLUSION_PATH,
    Path("ops/e2e-test-plan.md"),
    Path("python/carnot/experiment_7216_v635_down_up_quality.py"),
    Path("python/carnot/samplers/experiment_7215_down_up.py"),
    Path("results/experiment_7215_v635_down_up_prototype.json"),
    UPSTREAM_PATH,
    TRACE_PATH,
    SPEC_PATH,
    Path("python/carnot/experiment_7229_v636_rare_event_audit.py"),
    Path("scripts/experiments/experiment_7229_v636_rare_event_audit.py"),
    Path("tests/python/test_experiment_7229_v636_rare_event_audit.py"),
)

FIELD_PRINCIPLES = {
    "field_principles": (
        "Annotate actual values in this map; do not wrap arbitrary dictionaries as "
        "principle/value records."
    ),
    "status": (
        "Write a terminal artifact only when done or externally blocked; running checkpoints "
        "use a different path."
    ),
    "run_date": "Use 20260912 and record actual UTC timestamps, never copy an upstream run date.",
    "preconditions_checked": "Actual code, resource, identity and gate observations before expensive work.",
    "inference_substrate": (
        "Use the recognized literal for the work actually executed; custom free text caused "
        "the Exp7208 quarantine."
    ),
    "inference_substrate_class": (
        "Match actual generation, load-only, CPU or aggregation work and its duration floor."
    ),
    "execution_venue": (
        "Exactly host, kv260, gatemate or polarfire; the top-level orchestration here is host."
    ),
    "execution_host": "Actual hostname separate from venue.",
    "duration_s": "Measured monotonic work time; no padding or reclassification to evade a floor.",
    "source_artifact_hashes": "Bind code, source documents, manifests and raw evidence to claims.",
    "rows": (
        "Per unit/arm/seed metric, error and abstention for every comparison; retain full "
        "denominators."
    ),
    "sample_size_budget": (
        "Planned, attempted, completed, censored and independent units; no silent removal."
    ),
    "random_seed": "Freeze random choices before reading held-out outcomes.",
    "reproducibility_checksum": "Hash exact source, inputs, settings and raw unit rows.",
    "gate_check_summary": (
        "Every blocked_* verdict names failed check, upstream, field, expected and observed value."
    ),
    "verifier_is_oracle": (
        "True when correctness authority is reused as the verifier; independent code alone is "
        "not distinct authority."
    ),
    "verdict_class": (
        "Closed enum positive | circular_positive | null | blocked | disqualified | partial. "
        "partial means unfinished own work only."
    ),
    "honest_verdict": (
        "Use complete_ or complete: for completed findings; blocked_* for external absence. "
        "A failed acceptance gate forbids positive."
    ),
    "MODEL_SPECS": (
        "Only models actually invoked; [] for CPU/aggregation, mandated Qwen3.8 for every model task."
    ),
    "model_invoked": "True only for actual model execution; upstream model outputs are cached evidence.",
    "rare_event_audit_complete_score": (
        "Read-only cause classification complete, not a new mixing result."
    ),
    "observable_rows": "Per graph/arm/probe probabilities, visits, variance, ESS and failure cause.",
    "iid_reference_only": "Flags any IID diagnostic as non-certifying for correlated chains.",
    "original_gate_preserved": "The prior null and all failed criteria remain.",
    "next_measurement_envelope": "Fixed observable targets, resources and stop condition.",
}

REQUIRED_ARTIFACT_FIELDS = frozenset(FIELD_PRINCIPLES) | {
    "task_id",
    "milestone",
    "root",
    "started_at_utc",
    "completed_at_utc",
    "evidence_checks",
    "down_up_value_score",
    "paper_replication_claimed",
    "sampler_rerun_performed",
    "throughput_sweep_performed",
    "hardware_claimed",
    "spec_refs",
}


def _progress(phase: int, boundary: str, operation: str) -> None:
    """Flush each phase boundary so long experiment orchestration stays observable."""

    print(f"[phase {phase} {boundary}] {operation}", flush=True)


def canonical_json(value: Any) -> str:
    """Encode finite JSON deterministically so evidence hashes are stable."""

    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except ValueError as exc:
        raise ValueError("nonfinite value in canonical JSON") from exc


def sha256_json(value: Any) -> str:
    """Hash canonical JSON instead of interpreter-specific object formatting."""

    return "sha256:" + hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    """Hash every byte of one source or evidence file."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def artifact_checksum(payload: Mapping[str, Any]) -> str:
    """Bind every terminal field except the checksum that stores this digest."""

    material = dict(payload)
    material.pop("reproducibility_checksum", None)
    return sha256_json(material)


def without_row_hash(row: Mapping[str, Any]) -> JsonDict:
    """Return the exact row material covered by its content hash."""

    return {key: value for key, value in row.items() if key != "row_sha256"}


def _finish_row(row: JsonDict) -> JsonDict:
    """Add the common evidence contract before hashing the complete row."""

    row.setdefault("error", None)
    row.setdefault("abstention", False)
    row["row_sha256"] = sha256_json(row)
    return row


def unwrap_principled_value(value: Any) -> Any:
    """Unwrap only the exact principle/value record defined by the artifact contract."""

    if (
        isinstance(value, Mapping)
        and set(value) == {"principle", "value"}
        and isinstance(value.get("principle"), str)
    ):
        return value["value"]
    return value


def upstream_quarantine_observation(
    upstream: Mapping[str, Any], *, manifest_match: bool
) -> JsonDict:
    """Combine artifact and exclusion-manifest quarantine evidence before gate reads."""

    names = (
        "flagged_adversarial",
        "quarantined",
        "quarantine",
        "quarantine_flags",
        "disqualified",
        "invalidated",
    )
    observed = {name: upstream.get(name) for name in names}
    active = [name for name, value in observed.items() if value not in (None, False, "", [], {})]
    if manifest_match:
        active.append("exclusion_manifest")
    return {
        **observed,
        "exclusion_manifest_match": manifest_match,
        "active_flags": active,
        "quarantined": bool(active),
    }


def gated_upstream_value(
    upstream: Mapping[str, Any], quarantine: Mapping[str, Any], field_name: str
) -> Any:
    """Reject quarantined input before narrowly unwrapping a requested field."""

    if quarantine.get("quarantined") is True:
        return "not_consumed_due_to_quarantine"
    return unwrap_principled_value(upstream.get(field_name))


def _read_json_object(path: Path) -> Mapping[str, Any]:
    """Return one decoded object or an empty object for an unreadable prerequisite."""

    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, Mapping) else {}


def _task_contract(root: Path) -> JsonDict | None:
    """Read only the roadmap fields that authorize this audit."""

    try:
        document = yaml.safe_load((root / ROADMAP_PATH).read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError):
        return None
    if not isinstance(document, Mapping) or not isinstance(document.get("tasks"), list):
        return None
    task = next(
        (
            item
            for item in document["tasks"]
            if isinstance(item, Mapping) and item.get("id") == TASK_ID
        ),
        None,
    )
    if task is None:
        return None
    return {
        "id": task.get("id"),
        "milestone": task.get("milestone"),
        "deliverable": task.get("deliverable"),
        "prior_failures": task.get("prior_failures"),
    }


def _check_row(
    check: str,
    upstream: str,
    field: str,
    expected: Any,
    observed: Any,
    passed: bool,
) -> JsonDict:
    """Use one explicit shape for successful and failed prerequisite checks."""

    return {
        "check": check,
        "upstream": upstream,
        "field": field,
        "expected_value": expected,
        "observed_value": observed,
        "passed": passed,
    }


def _directory_writable(directory: Path) -> bool:
    """Test actual create, flush, and unlink access with a private temporary file."""

    try:
        directory.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(prefix=".exp7229-preflight-", dir=directory) as handle:
            handle.write(b"writable")
            handle.flush()
            os.fsync(handle.fileno())
    except OSError:
        return False
    return True


def _trace_receipt(root: Path, upstream: Mapping[str, Any]) -> JsonDict:
    """Authenticate archive bytes and count decoded records without retaining traces."""

    receipt = upstream.get("trace_archive", {})
    if not isinstance(receipt, Mapping):
        return {"valid": False, "record_count": None}
    relative = receipt.get("path")
    if not isinstance(relative, str):
        return {"valid": False, "record_count": None}
    path = root / relative
    try:
        byte_count = path.stat().st_size
        digest = sha256_file(path)
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            record_count = sum(1 for line in handle if line.strip())
    except (OSError, gzip.BadGzipFile):
        return {"valid": False, "record_count": None}
    valid = (
        path.resolve() == (root / TRACE_PATH).resolve()
        and byte_count == receipt.get("bytes")
        and digest == receipt.get("sha256")
        and record_count == receipt.get("record_count")
        and record_count == 720
    )
    return {
        "valid": valid,
        "path": str(relative),
        "bytes": byte_count,
        "sha256": digest,
        "record_count": record_count,
    }


def collect_preconditions(root: Path, *, output: Path) -> tuple[list[JsonDict], dict[str, str]]:
    """Record code, resource, identity, gate, and output checks before aggregation."""

    print("[phase 0 check start] required source bytes", flush=True)
    sizes = {
        str(path): (root / path).stat().st_size if (root / path).is_file() else None
        for path in REQUIRED_SOURCE_PATHS
    }
    hashes = {
        str(path): sha256_file(root / path)
        for path in REQUIRED_SOURCE_PATHS
        if sizes[str(path)] not in (None, 0)
    }
    checks = [
        _check_row(
            "required_source_bytes",
            str(root),
            "REQUIRED_SOURCE_PATHS",
            "all files exist and are nonempty",
            sizes,
            all(size is not None and size > 0 for size in sizes.values()),
        )
    ]

    print("[phase 0 check start] driving capability specification", flush=True)
    spec_file = root / SPEC_PATH
    spec_text = spec_file.read_text(encoding="utf-8") if spec_file.is_file() else ""
    spec_observed = {
        "exists": spec_file.is_file(),
        "req_present": "### REQ-SAMPLER-7229" in spec_text,
        "scenarios_present": "#### SCENARIO-SAMPLER-7229-" in spec_text,
    }
    checks.append(
        _check_row(
            "driving_capability_spec",
            str(SPEC_PATH),
            "REQ-* and SCENARIO-*",
            {"exists": True, "req_present": True, "scenarios_present": True},
            spec_observed,
            all(spec_observed.values()),
        )
    )

    print("[phase 0 check start] roadmap task contract", flush=True)
    contract = _task_contract(root)
    checks.append(
        _check_row(
            "roadmap_task_contract",
            str(ROADMAP_PATH),
            "id,milestone,deliverable,prior_failures",
            EXPECTED_TASK_CONTRACT,
            contract,
            contract == EXPECTED_TASK_CONTRACT,
        )
    )

    print("[phase 0 check start] upstream quarantine and producer validation", flush=True)
    upstream = _read_json_object(root / UPSTREAM_PATH)
    exclusion_text = (
        (root / EXCLUSION_PATH).read_text(encoding="utf-8")
        if (root / EXCLUSION_PATH).is_file()
        else ""
    )
    manifest_match = any(
        token in exclusion_text
        for token in ("exp7216-down-up-quality", "experiment_7216_v635_down_up_quality")
    )
    quarantine = upstream_quarantine_observation(upstream, manifest_match=manifest_match)
    checks.append(
        _check_row(
            "upstream_quarantine",
            f"{UPSTREAM_PATH} and {EXCLUSION_PATH}",
            "artifact and manifest quarantine signals",
            {"quarantined": False},
            quarantine,
            bool(upstream) and quarantine["quarantined"] is False,
        )
    )
    producer_errors = exp7216.validate_artifact(upstream) if upstream else ["unreadable_artifact"]
    authentication = {
        "producer_errors": producer_errors,
        "artifact_checksum_matches": bool(upstream)
        and upstream.get("reproducibility_checksum") == exp7216.artifact_checksum(upstream),
        "row_hashes_valid": bool(upstream)
        and exp7216._row_hashes_valid(exp7216.combined_rows(upstream)),
    }
    checks.append(
        _check_row(
            "upstream_authentication",
            str(UPSTREAM_PATH),
            "producer validator, checksum, and row hashes",
            {"producer_errors": [], "artifact_checksum_matches": True, "row_hashes_valid": True},
            authentication,
            quarantine["quarantined"] is False
            and authentication
            == {
                "producer_errors": [],
                "artifact_checksum_matches": True,
                "row_hashes_valid": True,
            },
        )
    )

    print("[phase 0 check start] exact upstream gate fields", flush=True)
    value_score = gated_upstream_value(upstream, quarantine, "down_up_value_score")
    primary = gated_upstream_value(upstream, quarantine, "primary_gate")
    primary_passed = primary.get("passed") if isinstance(primary, Mapping) else primary
    checks.extend(
        [
            _check_row(
                "upstream_value_gate",
                str(UPSTREAM_PATH),
                "down_up_value_score",
                0,
                value_score,
                value_score == 0,
            ),
            _check_row(
                "upstream_primary_gate",
                str(UPSTREAM_PATH),
                "primary_gate.passed",
                False,
                primary_passed,
                primary_passed is False,
            ),
        ]
    )

    print("[phase 0 check start] trace archive authentication", flush=True)
    trace_observed = _trace_receipt(root, upstream)
    checks.append(
        _check_row(
            "trace_archive_authentication",
            str(TRACE_PATH),
            "path,bytes,sha256,record_count",
            {"valid": True, "record_count": 720},
            trace_observed,
            trace_observed.get("valid") is True,
        )
    )

    print("[phase 0 check start] imports, exact fields, and writable outputs", flush=True)
    rows_observed = {
        "exact_authority_rows": len(upstream.get("exact_authority_rows", [])),
        "matched_budget_rows": len(upstream.get("matched_budget_rows", [])),
        "quality_rows": len(upstream.get("quality_rows", [])),
        "quality_summary_rows": len(upstream.get("quality_summary_rows", [])),
    }
    tool_and_output = {
        "numpy": bool(np.__version__),
        "pyyaml": bool(yaml.__version__),
        "results_output_writable": _directory_writable(output.parent),
        "checkpoint_output_writable": _directory_writable(root / CHECKPOINT_DIR),
        "upstream_rows": rows_observed,
    }
    expected_rows = {
        "exact_authority_rows": 30,
        "matched_budget_rows": 480,
        "quality_rows": 240,
        "quality_summary_rows": 60,
    }
    checks.append(
        _check_row(
            "imports_fields_and_outputs",
            "host and Exp7216",
            "imports,row rosters,results/checkpoints access",
            {"tools_and_outputs": True, "upstream_rows": expected_rows},
            tool_and_output,
            all(
                tool_and_output[name] is True
                for name in (
                    "numpy",
                    "pyyaml",
                    "results_output_writable",
                    "checkpoint_output_writable",
                )
            )
            and rows_observed == expected_rows,
        )
    )
    return checks, hashes


def read_trace_archive(root: Path, upstream: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Authenticate each trace and retain only arrays needed for quality reduction."""

    receipt = _trace_receipt(root, upstream)
    if receipt.get("valid") is not True:
        raise ValueError("trace archive receipt is not authenticated")
    source_rows = [
        *upstream.get("matched_budget_rows", []),
        *upstream.get("quality_rows", []),
    ]
    expected = {str(row["unit_id"]): row for row in source_rows}
    if len(expected) != 720:
        raise ValueError("source trace roster is incomplete or duplicated")
    records: dict[str, dict[str, Any]] = {}
    started = time.monotonic()
    last_report = started
    with gzip.open(root / TRACE_PATH, "rt", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"trace record {line_number} is invalid JSON") from exc
            if not isinstance(record, Mapping):
                raise ValueError(f"trace record {line_number} is not an object")
            unit_id = record.get("unit_id")
            if not isinstance(unit_id, str) or unit_id not in expected or unit_id in records:
                raise ValueError(f"trace record {line_number} has an unknown or duplicate unit_id")
            trace = record.get("trace_indices")
            if not isinstance(trace, list) or not all(type(value) is int for value in trace):
                raise ValueError(f"trace record {unit_id} has invalid indices")
            source = expected[unit_id]
            state_count = math.comb(int(source["n"]), int(source["k"]))
            array = np.asarray(trace, dtype=np.int32)
            if array.size and (int(array.min()) < 0 or int(array.max()) >= state_count):
                raise ValueError(f"trace record {unit_id} has an out-of-range index")
            expected_panel = (
                "quality" if source.get("row_type") == "quality_chain" else "matched_budget"
            )
            if (
                record.get("panel") != expected_panel
                or record.get("rng_seed") != source.get("seed")
                or (
                    expected_panel == "quality"
                    and record.get("initial_state_index") != source.get("initial_state_index")
                )
                or sha256_json(trace) != source.get("trace_sha256")
            ):
                raise ValueError(f"trace record {unit_id} does not match its source row")
            records[unit_id] = {
                "unit_id": unit_id,
                "panel": expected_panel,
                "trace_count": len(trace),
                "trace_indices": array if expected_panel == "quality" else None,
            }
            now = time.monotonic()
            if now - last_report >= 60.0:
                print(
                    f"[phase 3 progress] completed={len(records)}/720 "
                    f"elapsed_s={now - started:.3f}",
                    flush=True,
                )
                last_report = now
    if set(records) != set(expected):
        raise ValueError("trace archive does not match the complete source roster")
    return records


def iid_zero_hit_probability(probability: float, sample_count: int) -> float:
    """Calculate an IID reference without treating it as an MCMC guarantee."""

    if not 0.0 <= probability <= 1.0:
        raise ValueError("probability must be in [0, 1]")
    if sample_count < 0:
        raise ValueError("sample_count must be nonnegative")
    if probability == 1.0:
        return 0.0 if sample_count else 1.0
    return math.exp(sample_count * math.log1p(-probability))


def iid_effective_sample_lower_bound(probability: float, *, relative_half_width: float) -> int:
    """Return the IID effective-draw lower bound for a relative normal interval."""

    if not 0.0 < probability < 1.0:
        raise ValueError("probability must be strictly between zero and one")
    if not 0.0 < relative_half_width < 1.0:
        raise ValueError("relative_half_width must be strictly between zero and one")
    return math.ceil(
        NORMAL_95_SQUARED * (1.0 - probability) / (probability * relative_half_width**2)
    )


def classify_failures(
    summary: Mapping[str, Any], *, observable_kind: str, visits: int | None
) -> JsonDict:
    """Classify each original criterion without turning uncertainty into success."""

    classifications: JsonDict = {}
    for output, source in (
        ("mean_tolerance", "mean_tolerance_passed"),
        ("ess", "chain_ess_passed"),
        ("rhat", "split_rhat_passed"),
    ):
        passed = summary.get(source)
        if passed is True:
            classifications[output] = None
        elif passed is not False:
            classifications[output] = "unresolved"
        elif output == "mean_tolerance":
            classifications[output] = "actual_bias"
        elif observable_kind == "occupancy" and visits == 0:
            classifications[output] = "unobserved_rare_probe"
        else:
            classifications[output] = "insufficient_transitions"
    return classifications


def _quality_groups(
    upstream: Mapping[str, Any],
) -> dict[tuple[int, float, int, str], list[JsonDict]]:
    """Group four independent source chains without changing their stored order."""

    groups: dict[tuple[int, float, int, str], list[JsonDict]] = {}
    for row in upstream.get("quality_rows", []):
        key = (int(row["n"]), float(row["beta"]), int(row["graph_seed"]), str(row["arm"]))
        groups.setdefault(key, []).append(dict(row))
    return groups


def build_observable_rows(
    upstream: Mapping[str, Any], trace_records: Mapping[str, Mapping[str, Any]]
) -> list[JsonDict]:
    """Reconstruct every energy and occupancy group from authenticated source rows."""

    authorities = {
        (int(row["n"]), float(row["beta"]), int(row["graph_seed"])): row
        for row in upstream.get("exact_authority_rows", [])
    }
    groups = _quality_groups(upstream)
    summaries = sorted(
        upstream.get("quality_summary_rows", []),
        key=lambda row: (
            int(row["n"]),
            float(row["beta"]),
            int(row["graph_seed"]),
            str(row["arm"]),
        ),
    )
    lookup_cache: dict[tuple[int, int, int], np.ndarray[Any, np.dtype[np.bool_]]] = {}
    rows: list[JsonDict] = []
    for summary_row in summaries:
        key = (
            int(summary_row["n"]),
            float(summary_row["beta"]),
            int(summary_row["graph_seed"]),
            str(summary_row["arm"]),
        )
        authority = authorities[key[:3]]
        chains = sorted(groups[key], key=lambda row: int(row["chain_id"]))
        probe_names = ["energy", *[f"occupancy_{site}" for site in authority["probe_indices"]]]
        for probe_name in probe_names:
            source_summary = summary_row["probe_summaries"][probe_name]
            sample_count = sum(int(chain["retained"]) for chain in chains)
            visit_count: int | None = None
            occupancy_change_count: int | None = None
            exact_probability: float | None = None
            expected_visits: float | None = None
            iid_zero_hit: float | None = None
            observable_kind = "energy" if probe_name == "energy" else "occupancy"
            if observable_kind == "occupancy":
                site = int(probe_name.removeprefix("occupancy_"))
                lookup_key = (key[0], int(authority["k"]), site)
                if lookup_key not in lookup_cache:
                    states = down_up.enumerate_subsets(key[0], int(authority["k"]))
                    lookup_cache[lookup_key] = np.fromiter(
                        (site in state for state in states), dtype=np.bool_, count=len(states)
                    )
                lookup = lookup_cache[lookup_key]
                visit_count = 0
                occupancy_change_count = 0
                for chain in chains:
                    record = trace_records[str(chain["unit_id"])]
                    trace = record["trace_indices"]
                    if not isinstance(trace, np.ndarray):
                        raise ValueError(f"missing quality trace bytes for {chain['unit_id']}")
                    burn_in = int(chain["burn_in"])
                    retained = int(chain["retained"])
                    values = lookup[trace[burn_in : burn_in + retained]]
                    if len(values) != retained:
                        raise ValueError(f"incomplete retained trace for {chain['unit_id']}")
                    visit_count += int(values.sum())
                    occupancy_change_count += int(np.count_nonzero(values[1:] != values[:-1]))
                exact_probability = float(authority["occupancy_means"][site])
                expected_visits = sample_count * exact_probability
                iid_zero_hit = iid_zero_hit_probability(exact_probability, sample_count)
                reconstructed_mean = visit_count / sample_count
                if reconstructed_mean != source_summary["observed_mean"]:
                    raise ValueError(
                        f"reconstructed mean differs for {summary_row['unit_id']}:{probe_name}"
                    )
            chain_ess = list(source_summary["chain_ess"])
            ess_computable = [value is not None for value in chain_ess]
            structurally_degenerate = (
                float(source_summary["exact_variance"]) == 0.0
                if observable_kind == "energy"
                else exact_probability in (0.0, 1.0)
            )
            audit_criteria = {
                "mean_tolerance_passed": source_summary["mean_tolerance_passed"],
                "chain_ess_passed": structurally_degenerate
                or all(value is not None and value >= exp7216.ESS_MINIMUM for value in chain_ess),
                "split_rhat_passed": structurally_degenerate
                or (
                    source_summary["split_rhat"] is not None
                    and source_summary["split_rhat"] <= exp7216.SPLIT_RHAT_MAXIMUM
                ),
            }
            classifications = classify_failures(
                audit_criteria, observable_kind=observable_kind, visits=visit_count
            )
            failure_causes = list(
                dict.fromkeys(value for value in classifications.values() if value is not None)
            )
            rows.append(
                _finish_row(
                    {
                        "row_type": "rare_event_observable",
                        "unit_id": f"{summary_row['unit_id']}:{probe_name}",
                        "arm": key[3],
                        "seed": key[2],
                        "graph_seed": key[2],
                        "n": key[0],
                        "k": int(authority["k"]),
                        "beta": key[1],
                        "probe": probe_name,
                        "observable_kind": observable_kind,
                        "metric": "authenticated_trace_observability",
                        "sample_count": sample_count,
                        "chain_count": len(chains),
                        "chain_ids": [int(chain["chain_id"]) for chain in chains],
                        "chain_seeds": [int(chain["seed"]) for chain in chains],
                        "source_unit_ids": [str(chain["unit_id"]) for chain in chains],
                        "exact_mean": float(source_summary["exact_mean"]),
                        "exact_probability": exact_probability,
                        "exact_variance": float(source_summary["exact_variance"]),
                        "observed_mean": float(source_summary["observed_mean"]),
                        "visit_count": visit_count,
                        "occupancy_change_count": occupancy_change_count,
                        "expected_visits": expected_visits,
                        "expected_visits_assumes_independence": False,
                        "chain_ess_values": chain_ess,
                        "ess_computable_by_chain": ess_computable,
                        "all_chain_ess_computable": all(ess_computable),
                        "split_rhat": source_summary["split_rhat"],
                        "rhat_computable": source_summary["split_rhat"] is not None,
                        "structurally_degenerate": structurally_degenerate,
                        "source_structurally_degenerate": source_summary["structurally_degenerate"],
                        "original_criteria": {
                            "mean_tolerance_passed": source_summary["mean_tolerance_passed"],
                            "chain_ess_passed": source_summary["chain_ess_passed"],
                            "split_rhat_passed": source_summary["split_rhat_passed"],
                        },
                        "audit_criteria": audit_criteria,
                        "failure_classification": classifications,
                        "failure_causes": failure_causes,
                        "iid_zero_hit_probability": iid_zero_hit,
                        "iid_reference_only": observable_kind == "occupancy",
                        "iid_calculation_is_mcmc_confidence_guarantee": False,
                        "exact_authority_sha256": authority["authority_sha256"],
                    }
                )
            )
    return rows


def build_evidence_checks(
    upstream: Mapping[str, Any],
    trace_records: Mapping[str, Mapping[str, Any]],
    observable_rows: Sequence[Mapping[str, Any]],
) -> JsonDict:
    """Check moments, probe meanings, charged work, construction, and causes."""

    authorities = {
        (int(row["n"]), float(row["beta"]), int(row["graph_seed"])): row
        for row in upstream.get("exact_authority_rows", [])
    }
    energy_failures = 0
    definition_failures = 0
    for summary in upstream.get("quality_summary_rows", []):
        key = (int(summary["n"]), float(summary["beta"]), int(summary["graph_seed"]))
        authority = authorities[key]
        energy = summary["probe_summaries"]["energy"]
        if (
            energy["exact_mean"] != authority["energy_mean"]
            or energy["exact_variance"] != authority["energy_variance"]
        ):
            energy_failures += 1
        expected_probes = {"energy", *[f"occupancy_{site}" for site in authority["probe_indices"]]}
        if set(summary["probe_summaries"]) != expected_probes:
            definition_failures += 1
    for row in observable_rows:
        if row["observable_kind"] == "occupancy" and (
            row["observed_mean"] != row["visit_count"] / row["sample_count"]
            or not math.isclose(
                row["exact_variance"],
                row["exact_probability"] * (1.0 - row["exact_probability"]),
                rel_tol=0.0,
                abs_tol=1.0e-15,
            )
        ):
            definition_failures += 1

    cost_failures: list[str] = []
    for row in [*upstream.get("matched_budget_rows", []), *upstream.get("quality_rows", [])]:
        unit_id = str(row["unit_id"])
        trace_count = int(trace_records[unit_id]["trace_count"])
        n = int(row["n"])
        k = int(row["k"])
        if row["row_type"] == "quality_chain":
            transitions = int(row["attempted_transitions"])
            expected_trace_count = transitions
            unfinished = 0
        else:
            transitions = int(row["completed_transitions"])
            expected_trace_count = transitions
            unfinished = int(row["unfinished_conditional_energy_evaluations"])
        if row["arm"] == "down_up":
            expected_energy = 1 + transitions * (n - k + 1) + unfinished
            expected_normalizations = transitions
        else:
            expected_energy = 1 + transitions
            expected_normalizations = 0
        valid = (
            trace_count == expected_trace_count
            and row["energy_evaluations"] == expected_energy
            and row["normalizations"] == expected_normalizations
        )
        if row["row_type"] == "matched_budget_chain" and row["protocol"] == "equal_work":
            valid = valid and row["energy_evaluations"] == exp7216.ENERGY_EVALUATION_BUDGET
        if row["row_type"] == "matched_budget_chain" and row["protocol"] == "equal_wall":
            valid = (
                valid
                and row["wall_budget_s"] == exp7216.WALL_BUDGET_S
                and row["wall_budget_overshoot_s"] >= 0.0
            )
        if not valid:
            cost_failures.append(unit_id)

    primary_rows = [
        row
        for row in upstream.get("exact_authority_rows", [])
        if row["n"] == 32 and row["k"] == 2 and row["beta"] == 1.0
    ]
    construction_passed = len(primary_rows) == len(exp7216.GRAPH_SEEDS) and all(
        row["state_count"] == math.comb(32, 2)
        and row["probe_indices"] == [0, 10, 21]
        and math.isclose(math.fsum(row["occupancy_means"]), 2.0, abs_tol=1.0e-12)
        and all(
            math.isclose(variance, probability * (1.0 - probability), abs_tol=1.0e-15)
            for probability, variance in zip(
                row["occupancy_means"], row["occupancy_variances"], strict=True
            )
        )
        and row["finite_law_passed"] is True
        and row["normalization_error"] <= exp7216.EXACT_TOLERANCE
        and row["energy_parity_error"] <= exp7216.EXACT_TOLERANCE
        for row in primary_rows
    )

    failed_criteria = 0
    classified_criteria = 0
    cause_counts: dict[str, int] = {}
    for row in observable_rows:
        for criterion, passed in row["audit_criteria"].items():
            if passed is False:
                failed_criteria += 1
                name = (
                    criterion.removesuffix("_passed")
                    .replace("chain_ess", "ess")
                    .replace("split_rhat", "rhat")
                )
                cause = row["failure_classification"].get(name)
                if cause in {
                    "actual_bias",
                    "insufficient_transitions",
                    "unobserved_rare_probe",
                    "missing_bytes",
                    "unresolved",
                }:
                    classified_criteria += 1
                    cause_counts[cause] = cause_counts.get(cause, 0) + 1
    checks = {
        "energy_moments": {
            "checked_summary_rows": len(upstream.get("quality_summary_rows", [])),
            "mismatches": energy_failures,
            "passed": energy_failures == 0,
        },
        "observable_definitions": {
            "checked_observable_rows": len(observable_rows),
            "mismatches": definition_failures,
            "passed": definition_failures == 0,
        },
        "transition_cost_accounting": {
            "checked_chain_rows": len(upstream.get("matched_budget_rows", []))
            + len(upstream.get("quality_rows", [])),
            "failed_unit_ids": cost_failures,
            "passed": not cost_failures,
        },
        "primary_graph_construction": {
            "cell": {"n": 32, "k": 2, "beta": 1.0},
            "expected_state_count": 496,
            "observed_state_counts": sorted({int(row["state_count"]) for row in primary_rows}),
            "checked_graphs": len(primary_rows),
            "passed": construction_passed,
        },
        "failure_classification": {
            "failed_criteria": failed_criteria,
            "classified_failed_criteria": classified_criteria,
            "unclassified_failed_criteria": failed_criteria - classified_criteria,
            "cause_counts": cause_counts,
            "passed": failed_criteria == classified_criteria,
        },
    }
    checks["all_passed"] = all(check["passed"] is True for check in checks.values())
    return checks


def build_next_measurement_envelope(upstream: Mapping[str, Any]) -> JsonDict:
    """Freeze a rare-event target and stop rule before any later chain exists."""

    fixed: list[JsonDict] = []
    for authority in upstream.get("exact_authority_rows", []):
        if authority["n"] != 32 or authority["k"] != 2 or authority["beta"] != 1.0:
            continue
        for site in authority["probe_indices"]:
            probability = float(authority["occupancy_means"][site])
            fixed.append(
                {
                    "graph_seed": int(authority["graph_seed"]),
                    "probe": f"occupancy_{site}",
                    "exact_probability": probability,
                    "exact_variance": float(authority["occupancy_variances"][site]),
                    "iid_effective_sample_lower_bound": iid_effective_sample_lower_bound(
                        probability, relative_half_width=RELATIVE_HALF_WIDTH
                    ),
                    "authority_sha256": authority["authority_sha256"],
                }
            )
    pooled_budget = MAX_RETAINED_PER_CHAIN * len(exp7216.CHAINS)
    required = max(row["iid_effective_sample_lower_bound"] for row in fixed)
    feasible = required <= pooled_budget
    return {
        "observables_fixed_before_new_chain": True,
        "scope": "Exp7216 primary n=32,k=2,beta=1 graph and occupancy probes",
        "fixed_observables": fixed,
        "accuracy_target": {
            "confidence_level": 0.95,
            "relative_half_width": RELATIVE_HALF_WIDTH,
            "all_four_chains_require_nonzero_occupancy_changes": True,
            "all_chain_ess_at_least": exp7216.ESS_MINIMUM,
            "split_rhat_at_most": exp7216.SPLIT_RHAT_MAXIMUM,
        },
        "budget": {
            "max_retained_transitions_per_chain": MAX_RETAINED_PER_CHAIN,
            "chains_per_graph_arm": len(exp7216.CHAINS),
            "graph_count": len(exp7216.GRAPH_SEEDS),
            "arm_count": len(exp7216.ARMS),
            "max_pooled_transitions_per_graph_arm": pooled_budget,
            "max_total_retained_transitions": pooled_budget
            * len(exp7216.GRAPH_SEEDS)
            * len(exp7216.ARMS),
            "gpu_or_llm_budget": 0,
        },
        "worst_probe_iid_effective_sample_lower_bound": required,
        "iid_lower_bound_only": True,
        "iid_lower_bound_is_mcmc_guarantee": False,
        "feasible_at_budget": feasible,
        "decision": "measure_with_fixed_envelope"
        if feasible
        else "retire_quality_claim_at_fixed_budget",
        "stop_condition": (
            "Stop and retain a null if any fixed probe exceeds the budget, has no occupancy "
            "changes in any chain, has null ESS or R-hat, or misses the fixed accuracy target."
        ),
    }


def _base_artifact(
    root: Path, checks: list[JsonDict], hashes: dict[str, str], started_at: str
) -> JsonDict:
    """Create all required fields before selecting a terminal result."""

    return {
        "field_principles": dict(FIELD_PRINCIPLES),
        "status": "running",
        "run_date": RUN_DATE,
        "started_at_utc": started_at,
        "completed_at_utc": None,
        "preconditions_checked": checks,
        "inference_substrate": "not_started",
        "inference_substrate_class": "blocked_no_run",
        "execution_venue": "host",
        "execution_host": platform.node() or "unknown-host",
        "duration_s": 0.0,
        "source_artifact_hashes": hashes,
        "rows": [],
        "observable_rows": [],
        "sample_size_budget": {
            "planned_trace_records": 720,
            "attempted_trace_records": 0,
            "completed_trace_records": 0,
            "censored_trace_records": 0,
            "planned_observable_rows": EXPECTED_OBSERVABLE_ROWS,
            "attempted_observable_rows": 0,
            "completed_observable_rows": 0,
            "censored_observable_rows": 0,
            "independent_graph_count": len(exp7216.GRAPH_SEEDS),
            "independent_chain_runs": (
                len(exp7216.CELLS)
                * len(exp7216.GRAPH_SEEDS)
                * len(exp7216.ARMS)
                * len(exp7216.CHAINS)
            ),
            "independent_draw_assumption_made": False,
            "unique_retained_mcmc_draws": 240 * exp7216.QUALITY_RETAINED,
        },
        "random_seed": {
            "value": None,
            "reason": "Deterministic read-only aggregation makes no random choices.",
        },
        "reproducibility_checksum": "",
        "gate_check_summary": {},
        "verifier_is_oracle": True,
        "verdict_class": "partial",
        "honest_verdict": "running",
        "MODEL_SPECS": [],
        "model_invoked": False,
        "rare_event_audit_complete_score": 0,
        "iid_reference_only": True,
        "original_gate_preserved": {},
        "next_measurement_envelope": {},
        "evidence_checks": {},
        "down_up_value_score": 0,
        "sampler_rerun_performed": False,
        "throughput_sweep_performed": False,
        "paper_replication_claimed": False,
        "hardware_claimed": False,
        "task_id": TASK_ID,
        "milestone": MILESTONE,
        "root": str(root.resolve()),
        "spec_refs": [
            "REQ-SAMPLER-7229",
            "SCENARIO-SAMPLER-7229-ZERO-HIT",
            "SCENARIO-SAMPLER-7229-TRACE",
            "SCENARIO-SAMPLER-7229-GATE",
            "SCENARIO-SAMPLER-7229-ARTIFACT",
        ],
    }


def _blocked_artifact(artifact: JsonDict, failed: Mapping[str, Any], started: float) -> JsonDict:
    """Publish a terminal external block without fabricated audit rows."""

    artifact.update(
        {
            "status": "blocked_external_precondition",
            "completed_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            "inference_substrate": "blocked_no_run",
            "inference_substrate_class": "blocked_no_run",
            "duration_s": time.monotonic() - started,
            "gate_check_summary": {
                "passed": False,
                "failed_check": failed.get("check"),
                "upstream": failed.get("upstream"),
                "field": failed.get("field"),
                "expected_value": failed.get("expected_value"),
                "observed_value": failed.get("observed_value"),
            },
            "verdict_class": "blocked",
            "honest_verdict": (
                f"blocked_external_precondition: {failed.get('check')} failed before aggregation"
            ),
        }
    )
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    return artifact


def build_artifact(
    root: Path,
    *,
    output: Path,
    preconditions: list[JsonDict] | None = None,
    source_hashes: dict[str, str] | None = None,
) -> JsonDict:
    """Build the complete read-only audit or one diagnosed external block."""

    started = time.monotonic()
    started_at = datetime.now(UTC).isoformat().replace("+00:00", "Z")
    _progress(0, "start", "precondition checks")
    if preconditions is None or source_hashes is None:
        measured_checks, measured_hashes = collect_preconditions(root, output=output)
        checks = measured_checks if preconditions is None else preconditions
        hashes = measured_hashes if source_hashes is None else source_hashes
    else:
        checks, hashes = preconditions, source_hashes
    artifact = _base_artifact(root, checks, hashes, started_at)
    failed = next((row for row in checks if row.get("passed") is not True), None)
    _progress(0, "end", "precondition checks")
    if failed is not None:
        return _blocked_artifact(artifact, failed, started)

    _progress(1, "start", "progress and timeout contract active")
    _progress(1, "end", "all aggregation loops have truthful elapsed heartbeats")
    _progress(2, "start", "inference declaration")
    artifact["inference_substrate"] = INFERENCE_SUBSTRATE
    artifact["inference_substrate_class"] = INFERENCE_SUBSTRATE_CLASS
    _progress(2, "end", "MODEL_SPECS=[] model_invoked=false")

    upstream = _read_json_object(root / UPSTREAM_PATH)
    _progress(3, "start", "authenticated trace reduction")
    trace_records = read_trace_archive(root, upstream)
    observable_rows = build_observable_rows(upstream, trace_records)
    artifact["observable_rows"] = observable_rows
    artifact["rows"] = observable_rows
    artifact["sample_size_budget"].update(
        {
            "attempted_trace_records": len(trace_records),
            "completed_trace_records": len(trace_records),
            "attempted_observable_rows": len(observable_rows),
            "completed_observable_rows": len(observable_rows),
        }
    )
    _progress(
        3, "end", f"trace_records={len(trace_records)} observable_rows={len(observable_rows)}"
    )

    _progress(4, "start", "IID reference diagnostics")
    iid_rows = sum(row["observable_kind"] == "occupancy" for row in observable_rows)
    artifact["iid_reference_only"] = True
    _progress(4, "end", f"iid_reference_rows={iid_rows} mcmc_guarantees=0")

    _progress(5, "start", "moment, observable, cost, and construction checks")
    evidence_checks = build_evidence_checks(upstream, trace_records, observable_rows)
    artifact["evidence_checks"] = evidence_checks
    _progress(5, "end", f"all_passed={str(evidence_checks['all_passed']).lower()}")

    _progress(6, "start", "preserved gate and next measurement envelope")
    original_primary = upstream["primary_gate"]
    artifact["original_gate_preserved"] = {
        "artifact": str(UPSTREAM_PATH),
        "upstream_honest_verdict": upstream["honest_verdict"],
        "upstream_verdict_class": upstream["verdict_class"],
        "down_up_value_score": upstream["down_up_value_score"],
        "primary_gate": original_primary,
        "failed_criteria": {
            key: value
            for key, value in original_primary.items()
            if value is False or (key == "ess_per_second_ratio_ci95" and value is None)
        },
        "unchanged": upstream["down_up_value_score"] == 0
        and original_primary.get("passed") is False,
    }
    artifact["next_measurement_envelope"] = build_next_measurement_envelope(upstream)
    complete = (
        len(observable_rows) == EXPECTED_OBSERVABLE_ROWS
        and evidence_checks["all_passed"] is True
        and artifact["original_gate_preserved"]["unchanged"] is True
    )
    artifact.update(
        {
            "status": "complete",
            "completed_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            "duration_s": time.monotonic() - started,
            "gate_check_summary": {
                "passed": True,
                "failed_check": None,
                "upstream": str(UPSTREAM_PATH),
                "field": "down_up_value_score and primary_gate.passed",
                "expected_value": {"down_up_value_score": 0, "primary_gate.passed": False},
                "observed_value": {"down_up_value_score": 0, "primary_gate.passed": False},
            },
            "verdict_class": "null",
            "honest_verdict": (
                "complete_null: the authenticated read-only audit classified every original "
                "criterion, preserved the failed Exp7216 gate, and found the fixed rare-event "
                "accuracy target infeasible at the stated budget."
            ),
            "rare_event_audit_complete_score": int(complete),
            "down_up_value_score": 0,
        }
    )
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    _progress(6, "end", f"audit_complete_score={int(complete)} down_up_value_score=0")
    return artifact


def _expected_observable_ids() -> set[str]:
    """Construct the complete fixed roster without reading held-out values."""

    return {
        f"n{n}:k{exp7216.CARDINALITY}:beta{beta:g}:graph{seed}:{arm}:summary:{probe}"
        for n, beta in exp7216.CELLS
        for seed in exp7216.GRAPH_SEEDS
        for arm in exp7216.ARMS
        for probe in ("energy", f"occupancy_0", f"occupancy_{n // 3}", f"occupancy_{2 * n // 3}")
    }


def _observable_rows_valid(rows: Any) -> bool:
    """Reject missing probes, changed hashes, and favorable zero-hit diagnostics."""

    if not isinstance(rows, list) or len(rows) != EXPECTED_OBSERVABLE_ROWS:
        return False
    if {
        row.get("unit_id") for row in rows if isinstance(row, Mapping)
    } != _expected_observable_ids():
        return False
    for row in rows:
        if not isinstance(row, Mapping) or row.get("row_sha256") != sha256_json(
            without_row_hash(row)
        ):
            return False
        if not all(
            key in row for key in ("unit_id", "arm", "seed", "metric", "error", "abstention")
        ):
            return False
        classification = classify_failures(
            row.get("audit_criteria", {}),
            observable_kind=str(row.get("observable_kind")),
            visits=row.get("visit_count"),
        )
        if row.get("failure_classification") != classification:
            return False
        if row.get("observable_kind") == "occupancy":
            probability = row.get("exact_probability")
            count = row.get("sample_count")
            visits = row.get("visit_count")
            if (
                not isinstance(probability, (int, float))
                or not isinstance(count, int)
                or not isinstance(visits, int)
                or row.get("iid_reference_only") is not True
                or row.get("iid_calculation_is_mcmc_confidence_guarantee") is not False
                or not math.isclose(
                    row.get("iid_zero_hit_probability"),
                    iid_zero_hit_probability(float(probability), count),
                    rel_tol=1.0e-12,
                    abs_tol=0.0,
                )
            ):
                return False
            if (
                probability > 0.0
                and visits == 0
                and (
                    row.get("structurally_degenerate") is not False
                    or row.get("all_chain_ess_computable") is not False
                    or row.get("rhat_computable") is not False
                    or any(value is not None for value in row.get("chain_ess_values", []))
                    or "unobserved_rare_probe" not in row.get("failure_causes", [])
                )
            ):
                return False
    return True


def validate_artifact(payload: Mapping[str, Any], *, root: Path | None = None) -> list[str]:
    """Recompute coverage, classifications, gate preservation, hashes, and claims."""

    if not REQUIRED_ARTIFACT_FIELDS.issubset(payload):
        return ["missing_required_fields"]
    errors: list[str] = []
    if payload.get("field_principles") != FIELD_PRINCIPLES:
        errors.append("field_principles_invalid")
    if payload.get("run_date") != RUN_DATE:
        errors.append("run_date_invalid")
    if payload.get("execution_venue") != "host" or not payload.get("execution_host"):
        errors.append("execution_identity_invalid")
    duration = payload.get("duration_s")
    if not isinstance(duration, (int, float)) or not math.isfinite(duration) or duration < 0.0:
        errors.append("duration_invalid")
    if payload.get("reproducibility_checksum") != artifact_checksum(payload):
        errors.append("reproducibility_checksum_mismatch")
    if payload.get("MODEL_SPECS") != [] or payload.get("model_invoked") is not False:
        errors.append("model_declaration_invalid")

    if payload.get("verdict_class") == "blocked":
        gate = payload.get("gate_check_summary", {})
        if (
            payload.get("status") != "blocked_external_precondition"
            or payload.get("inference_substrate") != "blocked_no_run"
            or payload.get("inference_substrate_class") != "blocked_no_run"
            or payload.get("rare_event_audit_complete_score") != 0
            or payload.get("down_up_value_score") != 0
            or payload.get("rows")
            or payload.get("observable_rows")
            or not isinstance(gate, Mapping)
            or gate.get("passed") is not False
            or not all(
                gate.get(key) is not None
                for key in ("failed_check", "upstream", "field", "expected_value", "observed_value")
            )
        ):
            errors.append("blocked_contract_invalid")
        return list(dict.fromkeys(errors))

    rows = payload.get("observable_rows")
    if payload.get("rows") != rows or not _observable_rows_valid(rows):
        errors.append("observable_rows_invalid")
    budget = payload.get("sample_size_budget", {})
    if (
        not isinstance(budget, Mapping)
        or budget.get("planned_trace_records") != 720
        or budget.get("attempted_trace_records") != 720
        or budget.get("completed_trace_records") != 720
        or budget.get("censored_trace_records") != 0
        or budget.get("planned_observable_rows") != EXPECTED_OBSERVABLE_ROWS
        or budget.get("attempted_observable_rows") != EXPECTED_OBSERVABLE_ROWS
        or budget.get("completed_observable_rows") != EXPECTED_OBSERVABLE_ROWS
        or budget.get("censored_observable_rows") != 0
        or budget.get("independent_draw_assumption_made") is not False
    ):
        errors.append("sample_size_budget_invalid")
    original = payload.get("original_gate_preserved", {})
    if (
        payload.get("down_up_value_score") != 0
        or not isinstance(original, Mapping)
        or original.get("down_up_value_score") != 0
        or original.get("unchanged") is not True
        or not isinstance(original.get("primary_gate"), Mapping)
        or original["primary_gate"].get("passed") is not False
    ):
        errors.append("original_gate_not_preserved")
    if root is not None:
        upstream = _read_json_object(root / UPSTREAM_PATH)
        if (
            original.get("primary_gate") != upstream.get("primary_gate")
            or original.get("upstream_honest_verdict") != upstream.get("honest_verdict")
            or original.get("upstream_verdict_class") != upstream.get("verdict_class")
        ):
            errors.append("original_gate_not_preserved")
        recorded = payload.get("source_artifact_hashes", {})
        if (
            not isinstance(recorded, Mapping)
            or set(recorded) != {str(path) for path in REQUIRED_SOURCE_PATHS}
            or any(
                not (root / path).is_file() or recorded[str(path)] != sha256_file(root / path)
                for path in REQUIRED_SOURCE_PATHS
            )
        ):
            errors.append("source_artifact_hashes_invalid")
    if payload.get("iid_reference_only") is not True:
        errors.append("iid_reference_invalid")
    evidence = payload.get("evidence_checks", {})
    if not isinstance(evidence, Mapping) or evidence.get("all_passed") is not True:
        errors.append("evidence_checks_invalid")
    envelope = payload.get("next_measurement_envelope", {})
    if (
        not isinstance(envelope, Mapping)
        or envelope.get("observables_fixed_before_new_chain") is not True
        or envelope.get("iid_lower_bound_only") is not True
        or envelope.get("iid_lower_bound_is_mcmc_guarantee") is not False
        or envelope.get("feasible_at_budget") is not False
        or envelope.get("decision") != "retire_quality_claim_at_fixed_budget"
    ):
        errors.append("next_measurement_envelope_invalid")
    if (
        payload.get("status") != "complete"
        or payload.get("verdict_class") != "null"
        or not str(payload.get("honest_verdict", "")).startswith("complete_")
        or payload.get("rare_event_audit_complete_score") != 1
        or payload.get("inference_substrate") != INFERENCE_SUBSTRATE
        or payload.get("inference_substrate_class") != INFERENCE_SUBSTRATE_CLASS
    ):
        errors.append("terminal_verdict_invalid")
    if any(
        payload.get(field) is not False
        for field in (
            "sampler_rerun_performed",
            "throughput_sweep_performed",
            "paper_replication_claimed",
            "hardware_claimed",
        )
    ):
        errors.append("claim_limits_invalid")
    return list(dict.fromkeys(errors))


def atomic_write(path: Path, payload: Mapping[str, Any]) -> JsonDict:
    """Publish complete JSON with one same-directory atomic replacement."""

    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n"
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
        "atomic_replace": True,
    }


def run_experiment(root: Path, output: Path) -> JsonDict:
    """Build, validate, and atomically publish the terminal audit artifact."""

    artifact = build_artifact(root, output=output)
    _progress(7, "start", "final artifact validation")
    errors = validate_artifact(artifact, root=root)
    _progress(7, "end", f"final artifact validation errors={len(errors)}")
    if errors:
        raise ValueError(f"invalid Exp7229 artifact: {errors}")
    _progress(8, "start", "atomic terminal write")
    receipt = atomic_write(output, artifact)
    _progress(8, "end", f"atomic terminal write bytes={receipt['bytes']}")
    return artifact


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    """Parse the fixed date and optional read-only validation path."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=RUN_DATE)
    parser.add_argument("--output", type=Path, default=RESULT_PATH)
    parser.add_argument("--validate", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the aggregation or validate caller-selected durable bytes."""

    args = _parse_args(argv)
    root = Path(__file__).resolve().parents[2]
    if args.validate is not None:
        _progress(7, "start", f"read-only validation path={args.validate}")
        try:
            decoded = json.loads(args.validate.read_text(encoding="utf-8"))
            payload = decoded if isinstance(decoded, Mapping) else {}
            errors = validate_artifact(payload, root=root)
        except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
            print(f"validation_error: {exc}", flush=True)
            _progress(7, "end", "read-only validation errors=1")
            return 2
        print(canonical_json({"errors": errors, "valid": not errors}), flush=True)
        _progress(7, "end", f"read-only validation errors={len(errors)}")
        return 0 if not errors else 2
    if args.date != RUN_DATE:
        print(f"experiment_error: run date must be {RUN_DATE}", flush=True)
        return 2
    try:
        output = args.output if args.output.is_absolute() else root / args.output
        run_experiment(root, output)
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        print(f"experiment_error: {exc}", flush=True)
        return 2
    return 0
