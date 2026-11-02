"""Audit V637 mention extraction, exact execution, and decision value.

The task never invokes a model. It first authenticates the fixture, canary,
and capture chain. A quarantined capture produces a terminal explanation.

Spec refs: REQ-VERIFY-7239 and SCENARIO-VERIFY-7239-*.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from copy import deepcopy
from datetime import UTC, datetime
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import time
from typing import Any

import yaml

from carnot import experiment_7236_v637_mention_fixture as fixture_module
from carnot import experiment_7238_v637_mention_capture as capture_module
from carnot.experiment_artifacts import atomic_write_json
from carnot.paths import repo_root as find_repo_root


JsonDict = dict[str, Any]

RUN_DATE = "20260912"
MILESTONE = "2026.09.637"
EXPERIMENT_ID = "exp7239-semantic-audit"
RANDOM_SEED = 7_239_001
BOOTSTRAP_SEED = 7_239_010
BOOTSTRAP_DRAWS = 10_000
MODEL_SPECS: list[JsonDict] = []
ARMS = ("mention_pointer", "explicit_schema_offset_control", "direct_judge")
COMPARISONS = {
    "pointer_vs_offset": "explicit_schema_offset_control",
    "pointer_vs_direct": "direct_judge",
}

RESULT_PATH = Path("results/experiment_7239_v637_semantic_audit.json")
CHECKPOINT_PATH = Path("results/checkpoints/experiment_7239/running.json")
FIXTURE_PATH = Path("results/experiment_7236_v637_mention_fixture.json")
CANARY_PATH = Path("results/experiment_7237_v637_mention_canary.json")
CAPTURE_PATH = Path("results/experiment_7238_v637_mention_capture.json")
PUBLIC_PATH = Path("results/raw/experiment_7236/public_manifest.json")
AUTHORITY_PATH = Path("results/raw/experiment_7236/authority_manifest.json")
CANARY_MANIFEST_PATH = Path("results/raw/experiment_7237/raw_request_manifest.json")
CAPTURE_MANIFEST_PATH = Path("results/raw/experiment_7238/manifest.json")
EXCLUSION_PATH = Path("ops/exclusion_manifest.yaml")
SPEC_PATH = Path("openspec/capabilities/verification/spec.md")
MODULE_PATH = Path("python/carnot/experiment_7239_v637_semantic_audit.py")
WRAPPER_PATH = Path("scripts/experiments/experiment_7239_v637_semantic_audit.py")
TEST_PATH = Path("tests/python/test_experiment_7239_v637_semantic_audit.py")

PINNED_HASHES = {
    "fixture": "sha256:b5604654dcefe5755479137c930f0ff5266b4c064aa6701f5c72fd4dec8620e3",
    "canary": "sha256:e4944826d783d3bfc319899fd1bdd683e4fa36a6efe8aa053b22971bee7a5d75",
    "capture": "sha256:b91102c32462728266a43214b79bc07e77baf3ca73776cf96fc83ae47f855f3c",
    "public": "sha256:b6a11618b77431d6e6d80b449662ffe1845e6c0193097eae8bc95d388171019d",
    "authority": "sha256:7b19fdec3833f09805b210f51231365581269ec084dc09820e1b8b7f67c74edc",
    "canary_manifest": "sha256:50bcf999ccfb3b6d3c5bbfd60b09de68f860b83f6b70d1cd2ed23b38a9ad72d7",
}

SOURCE_PATHS = {
    "agents": Path("AGENTS.md"),
    "claude": Path("CLAUDE.md"),
    "codex": Path("CODEX.md"),
    "research_program": Path("research-program.md"),
    "research_references": Path("research-references.md"),
    "exclusion_manifest": EXCLUSION_PATH,
    "e2e_test_plan": Path("ops/e2e-test-plan.md"),
    "grounding_audit_module": Path("python/carnot/experiment_7197_v634_grounding_value_audit.py"),
    "span_fixture_module": Path("python/carnot/experiment_7222_v636_span_fixture.py"),
    "row_consistency_lint": Path("scripts/verdict_row_consistency_lint.py"),
    "adversarial_verifier": Path("scripts/adversarial_verify.py"),
    "verifier_gaps": Path("ops/verifier_gaps.md"),
    "verification_spec": SPEC_PATH,
    "fixture_module": Path("python/carnot/experiment_7236_v637_mention_fixture.py"),
    "canary_module": Path("python/carnot/experiment_7237_v637_mention_canary.py"),
    "capture_module": Path("python/carnot/experiment_7238_v637_mention_capture.py"),
    "fixture": FIXTURE_PATH,
    "canary": CANARY_PATH,
    "capture": CAPTURE_PATH,
    "public": PUBLIC_PATH,
    "authority": AUTHORITY_PATH,
    "canary_manifest": CANARY_MANIFEST_PATH,
    "capture_manifest": CAPTURE_MANIFEST_PATH,
    "module": MODULE_PATH,
    "entrypoint": WRAPPER_PATH,
    "focused_tests": TEST_PATH,
}

FIELD_PRINCIPLES: JsonDict = {
    "schema": "Version the artifact and bind experiment_id and milestone to this task.",
    "status": "Terminal complete or blocked only; unfinished work uses a separate checkpoint path.",
    "run_date": "Use 20260912; retain actual UTC start and end timestamps.",
    "field_principles": "Keep ordinary values at top level; put their explanations in this map.",
    "preconditions_checked": "Observed paths, resources, model identity and upstream checks before expensive work.",
    "inference_substrate": "Use the recognized literal for the operation actually executed.",
    "inference_substrate_class": "Actual class determines the duration floor; never pad duration or relabel to pass.",
    "execution_venue": "Top-level orchestration is host; board rows name kv260, gatemate or polarfire.",
    "execution_host": "Actual hostname, distinct from execution_venue.",
    "duration_s": "Measured monotonic elapsed work; record phase spans separately.",
    "MODEL_SPECS": "Models actually invoked; [] for tasks with no LLM.",
    "model_invoked": "Current task execution only; historical sources and injected fixtures are separate.",
    "source_artifact_hashes": "Hash source code, public inputs, private evaluator inputs and raw output files.",
    "rows": "Every comparison retains one row per independent unit and arm, with errors and abstentions.",
    "sample_size_budget": "Predeclared independent units, attempted/completed/censored units, and stopping rule.",
    "random_seed": "Freeze seeds and schedules before observing evaluation labels.",
    "reproducibility_checksum": "Hash the exact settings, inputs and raw rows supporting the result.",
    "gate_check_summary": "Every blocked_* verdict names check, upstream, artifact_field, expected and observed value.",
    "verifier_is_oracle": "True when the verification authority also defines correctness; separate code is insufficient independence.",
    "verdict_class": "Exactly positive | circular_positive | null | blocked | disqualified | partial. External incompleteness is blocked.",
    "honest_verdict": "Completed findings start complete_ or complete:. External absence starts blocked_. Failed acceptance forbids positive.",
    "acceptance_gate_results": "Preserve each frozen criterion, actual value and pass/fail independently of task completion.",
    "semantic_audit_complete_score": "All available authentic held-out rows reduced; blocked input remains a terminal blocked disposition.",
    "semantic_value_score": "All predeclared fidelity, coverage, paired error and false-accept criteria met.",
    "paired_comparison_rows": "Per base question and arm metrics; no hidden filtering of invalid outputs.",
    "authority_boundary": "Gold source graph, extracted graph and independent labels identified separately.",
    "positive_control_results": "Oracle headroom and effective interventions, including no-headroom diagnoses.",
    "timestamps": "Record actual UTC observations for task start and terminal completion.",
    "phase_spans": "Keep measured phase work separate from total monotonic duration.",
    "current_invocation_counts": "Current model loads, generations, and invocations stay zero.",
    "calibration_observations": "Keep quarantined calibration evidence as diagnosis with paths and hashes, never as promoted value evidence.",
    "upstream_disposition": "Name the exact capture-chain state that permits or blocks held-out replay.",
    "arm_metrics": "Keep fidelity, coverage, unconditional error, selective risk, and false accepts separate.",
    "cost_metrics": "Report decoding tokens, elapsed time, timeouts, and transport separately from accuracy.",
    "paired_interval_rows": "Use 64 base questions and 10000 frozen paired bootstrap draws for CI95.",
    "failed_layer": "Name extraction, execution, decision, or headroom before branch disposition.",
    "validation_command_rows": "Retain exact observed validation commands without changing scientific rows.",
}
REQUIRED_ARTIFACT_FIELDS = tuple(FIELD_PRINCIPLES)


def canonical_json(value: Any) -> str:
    """Use one stable Unicode JSON spelling for every content hash."""

    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def sha256_bytes(value: bytes) -> str:
    """Hash exact bytes so text decoding cannot change source identity."""

    return "sha256:" + hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    """Hash a file in bounded chunks without changing its bytes."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def unwrap_principle(value: Any) -> Any:
    """Unwrap only a mapping that supplies both annotation fields."""

    if isinstance(value, Mapping) and "principle" in value and "value" in value:
        return value["value"]
    return value


def artifact_checksum(artifact: Mapping[str, Any]) -> str:
    """Bind durable evidence while excluding process-local clock values."""

    stable = {
        key: value
        for key, value in artifact.items()
        if key not in {"duration_s", "timestamps", "phase_spans", "reproducibility_checksum"}
    }
    return sha256_bytes(canonical_json(stable).encode("utf-8"))


def gate_row(
    check: str,
    expected: Any,
    observed: Any,
    passed: bool,
    *,
    upstream: str | None,
    artifact_field: str,
) -> JsonDict:
    """Retain both sides of one gate so a block remains actionable."""

    return {
        "check": check,
        "upstream": upstream,
        "artifact_field": artifact_field,
        "expected_value": expected,
        "observed_value": observed,
        "passed": bool(passed),
    }


def gate_summary(failure: Mapping[str, Any] | None) -> JsonDict:
    """Project the first failure without dropping either observed side."""

    if failure is None:
        return {
            "passed": True,
            "failed_check": None,
            "upstream": None,
            "artifact_field": None,
            "expected_value": "all_required_checks_pass",
            "observed_value": "all_required_checks_pass",
        }
    return {
        "passed": False,
        "failed_check": failure.get("check"),
        "upstream": failure.get("upstream"),
        "artifact_field": failure.get("artifact_field"),
        "expected_value": deepcopy(failure.get("expected_value")),
        "observed_value": deepcopy(failure.get("observed_value")),
    }


def _utc_now() -> str:
    """Record a real UTC observation for process provenance."""

    return datetime.now(UTC).isoformat()


def _progress(phase: int, event: str, detail: str) -> None:
    """Flush each observed boundary so a slow task remains inspectable."""

    print(f"[exp7239] phase {phase} {event}: {detail}", flush=True)


def _resolved_paths(root: Path, overrides: Mapping[str, Path] | None = None) -> dict[str, Path]:
    """Resolve declared sources while allowing isolated missing-input tests."""

    paths = {name: root / path for name, path in SOURCE_PATHS.items()}
    for name, path in (overrides or {}).items():
        paths[name] = path if path.is_absolute() else root / path
    return paths


def _source_hashes(paths: Mapping[str, Path]) -> JsonDict:
    """Retain each source path and hash, including expected missing raw capture."""

    return {
        name: {
            "path": str(path),
            "sha256": sha256_file(path) if path.is_file() else "missing",
        }
        for name, path in sorted(paths.items())
    }


def load_upstream_bundle(root: Path, path_overrides: Mapping[str, Path] | None = None) -> JsonDict:
    """Read exact V637 artifacts and raw manifests before any scientific replay."""

    paths = _resolved_paths(root, path_overrides)
    bundle: JsonDict = {"paths": paths}
    for name in ("fixture", "canary", "capture"):
        raw = paths[name].read_bytes()
        bundle[f"{name}_bytes"] = raw
        value = json.loads(raw)
        if not isinstance(value, dict):
            raise ValueError(f"{name}_artifact_mapping")
        bundle[name] = value
    for name in ("public", "authority", "canary_manifest"):
        raw = paths[name].read_bytes()
        bundle[f"{name}_bytes"] = raw
        value = json.loads(raw)
        if not isinstance(value, dict):
            raise ValueError(f"{name}_mapping")
        bundle[name] = value
    bundle["capture_manifest_bytes"] = (
        paths["capture_manifest"].read_bytes() if paths["capture_manifest"].is_file() else None
    )
    bundle["exclusion_manifest"] = yaml.safe_load(
        paths["exclusion_manifest"].read_text(encoding="utf-8")
    )
    return bundle


def _quarantined(artifact: Mapping[str, Any]) -> bool:
    """Reject explicit quarantine markers even when a numeric score passes."""

    for field in ("flagged_adversarial", "quarantined", "fabricated"):
        observed = unwrap_principle(artifact.get(field))
        if observed is True or (
            isinstance(observed, str) and observed.lower() in {"true", "quarantined"}
        ):
            return True
    return False


def _manifest_hits(value: Any, wanted: set[str]) -> bool:
    """Find exact excluded identifiers without matching unrelated prose substrings."""

    if isinstance(value, Mapping):
        return any(_manifest_hits(item, wanted) for item in value.values())
    if isinstance(value, list):
        return any(_manifest_hits(item, wanted) for item in value)
    return isinstance(value, str) and value in wanted


def upstream_gate_rows(bundle: Mapping[str, Any]) -> list[JsonDict]:
    """Authenticate the capture chain and put quarantine before numeric gates."""

    observed_hashes = {
        name: sha256_bytes(bundle[f"{name}_bytes"])
        for name in ("fixture", "canary", "capture", "public", "authority", "canary_manifest")
    }
    rows = [
        gate_row(
            "exact_upstream_bytes",
            PINNED_HASHES,
            observed_hashes,
            observed_hashes == PINNED_HASHES,
            upstream="experiment_7236_through_7238",
            artifact_field="artifact_and_manifest_bytes",
        )
    ]
    fixture = bundle["fixture"]
    canary = bundle["canary"]
    capture = bundle["capture"]
    quarantine_sources = [
        name
        for name, artifact in (
            ("experiment_7236", fixture),
            ("experiment_7237", canary),
            ("experiment_7238", capture),
        )
        if _quarantined(artifact)
    ]
    rows.append(
        gate_row(
            "structured_quarantine",
            False,
            bool(quarantine_sources),
            not quarantine_sources,
            upstream=quarantine_sources[0]
            if quarantine_sources
            else "experiment_7236_through_7238",
            artifact_field="flagged_adversarial|quarantined|fabricated",
        )
    )
    excluded = _manifest_hits(
        bundle.get("exclusion_manifest"),
        {
            "Exp7236",
            "Exp7237",
            "Exp7238",
            EXPERIMENT_ID,
            FIXTURE_PATH.as_posix(),
            CANARY_PATH.as_posix(),
            CAPTURE_PATH.as_posix(),
        },
    )
    rows.append(
        gate_row(
            "exclusion_manifest",
            False,
            excluded,
            not excluded,
            upstream="ops/exclusion_manifest.yaml",
            artifact_field="experiment_ids",
        )
    )
    fixture_observed = {
        "status": unwrap_principle(fixture.get("status")),
        "mention_fixture_ready_score": unwrap_principle(fixture.get("mention_fixture_ready_score")),
    }
    fixture_expected = {"status": "complete", "mention_fixture_ready_score": 1}
    rows.append(
        gate_row(
            "fixture_gate_fields",
            fixture_expected,
            fixture_observed,
            fixture_observed == fixture_expected,
            upstream="experiment_7236",
            artifact_field="status|mention_fixture_ready_score",
        )
    )
    canary_observed = {
        "status": unwrap_principle(canary.get("status")),
        "mention_canary_ready_score": unwrap_principle(canary.get("mention_canary_ready_score")),
    }
    canary_expected = {"status": "complete", "mention_canary_ready_score": 1}
    rows.append(
        gate_row(
            "canary_gate_fields",
            canary_expected,
            canary_observed,
            canary_observed == canary_expected,
            upstream="experiment_7237",
            artifact_field="status|mention_canary_ready_score",
        )
    )
    capture_observed = {
        "status": unwrap_principle(capture.get("status")),
        "mention_capture_complete_score": unwrap_principle(
            capture.get("mention_capture_complete_score")
        ),
    }
    capture_expected = {"status": "complete", "mention_capture_complete_score": 1}
    rows.append(
        gate_row(
            "capture_gate_fields",
            capture_expected,
            capture_observed,
            capture_observed == capture_expected,
            upstream="experiment_7238",
            artifact_field="status|mention_capture_complete_score",
        )
    )
    blocked_capture_ok = bool(
        capture_observed["status"] == "blocked"
        and isinstance(capture.get("gate_check_summary"), Mapping)
        and capture["gate_check_summary"].get("failed_check") == "structured_quarantine"
        and bundle.get("capture_manifest_bytes") is None
    )
    rows.append(
        gate_row(
            "capture_manifest_or_block_receipt",
            "authentic_manifest_or_structured_quarantine_block",
            {
                "manifest_present": bundle.get("capture_manifest_bytes") is not None,
                "blocked_capture_receipt": blocked_capture_ok,
            },
            bundle.get("capture_manifest_bytes") is not None or blocked_capture_ok,
            upstream="experiment_7238",
            artifact_field="capture_manifest_path|gate_check_summary",
        )
    )
    return rows


def collect_preconditions(
    root: Path,
    run_date: str,
    output_root: Path,
    *,
    path_overrides: Mapping[str, Path] | None = None,
) -> tuple[list[JsonDict], JsonDict]:
    """Check date, sources, imports, and writable destinations before replay."""

    paths = _resolved_paths(root, path_overrides)
    checks = [
        gate_row(
            "run_date",
            RUN_DATE,
            run_date,
            run_date == RUN_DATE,
            upstream=None,
            artifact_field="run_date",
        )
    ]
    if checks[0]["passed"] is False:
        return checks, {}
    optional = {"capture_manifest"}
    for name, path in paths.items():
        if name in optional:
            continue
        readable = path.is_file() and os.access(path, os.R_OK)
        checks.append(
            gate_row(
                "required_source",
                "readable_file",
                str(path) if readable else "missing_or_unreadable",
                readable,
                upstream=name,
                artifact_field="path",
            )
        )
        if not readable:
            return checks, {}
    spec_text = paths["verification_spec"].read_text(encoding="utf-8")
    checks.append(
        gate_row(
            "driving_spec",
            True,
            "REQ-VERIFY-7239" in spec_text,
            "REQ-VERIFY-7239" in spec_text,
            upstream="openspec/capabilities/verification/spec.md",
            artifact_field="REQ-VERIFY-7239",
        )
    )
    imports_ready = all(
        callable(value)
        for value in (
            fixture_module.resolve_pointer,
            fixture_module._execute_compiled_pair,
            capture_module.replay_completion_rows,
            capture_module.score_semantics,
        )
    )
    checks.append(
        gate_row(
            "required_imports",
            True,
            imports_ready,
            imports_ready,
            upstream="python/carnot",
            artifact_field="mention_resolver|typed_executor|capture_reducer",
        )
    )
    for name, path in (
        ("checkpoint", output_root / CHECKPOINT_PATH),
        ("result", output_root / RESULT_PATH),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        writable = path.parent.is_dir() and os.access(path.parent, os.W_OK)
        checks.append(
            gate_row(
                "output_destination",
                "writable_directory",
                str(path.parent) if writable else "missing_or_unwritable",
                writable,
                upstream=name,
                artifact_field="parent",
            )
        )
    if any(row["passed"] is False for row in checks):
        return checks, {}
    try:
        bundle = load_upstream_bundle(root, path_overrides)
    except (OSError, ValueError, json.JSONDecodeError, yaml.YAMLError) as exc:
        checks.append(
            gate_row(
                "upstream_parse",
                "valid_artifacts_and_manifests",
                f"{type(exc).__name__}:{exc}",
                False,
                upstream="experiment_7236_through_7238",
                artifact_field="json_or_yaml",
            )
        )
        return checks, {}
    checks.extend(upstream_gate_rows(bundle))
    return checks, bundle


def summarize_calibration(bundle: Mapping[str, Any]) -> JsonDict:
    """Preserve calibration counts while refusing their quarantined promotion."""

    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in bundle["canary"].get("rows", []):
        grouped[str(row.get("arm"))].append(row)
    arms = {}
    for arm in ("original_offset", "explicit_schema_offset_control", "mention_pointer"):
        rows = grouped[arm]
        arms[arm] = {
            "units": len(rows),
            "source_fidelity_count": sum(row.get("source_fidelity") is True for row in rows),
            "claim_fidelity_count": sum(row.get("claim_fidelity") is True for row in rows),
            "covered_count": sum(row.get("abstention") is not True for row in rows),
            "decision_correct_count": sum(row.get("decision_correct") is True for row in rows),
            "false_accept_count": sum(row.get("false_accept") is True for row in rows),
            "fully_correct_count": sum(row.get("fully_correct") is True for row in rows),
        }
    paths = bundle["paths"]
    return {
        "source_artifact": CANARY_PATH.as_posix(),
        "source_artifact_sha256": sha256_bytes(bundle["canary_bytes"]),
        "raw_manifest": CANARY_MANIFEST_PATH.as_posix(),
        "raw_manifest_sha256": sha256_bytes(bundle["canary_manifest_bytes"]),
        "transport_completed_calls": bundle["canary"].get("transport_completed_calls"),
        "numeric_ready_score": unwrap_principle(bundle["canary"].get("mention_canary_ready_score")),
        "quarantine_observed": _quarantined(bundle["canary"]),
        "eligible_for_promotion": False,
        "arms": arms,
        "observed_paths_exist": paths["canary"].is_file() and paths["canary_manifest"].is_file(),
    }


def authority_boundary(bundle: Mapping[str, Any]) -> JsonDict:
    """Identify which bytes define truth and when the evaluator may open them."""

    return {
        "gold_source_graph": {
            "path": AUTHORITY_PATH.as_posix(),
            "sha256": sha256_bytes(bundle["authority_bytes"]),
            "model_visible": False,
        },
        "extracted_graph": {
            "source": "capture_source_and_claim_outputs",
            "authority_fields_used_during_prediction": 0,
        },
        "independent_labels": {
            "path": AUTHORITY_PATH.as_posix(),
            "joined_after_prediction": True,
            "defines_correctness": True,
        },
        "verifier_is_oracle": True,
    }


def paired_row_errors(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    """Detect hidden filtering or changed identity in the 64-by-3 roster."""

    errors: list[str] = []
    if len(rows) != 64 * len(ARMS):
        errors.append("row_count")
    pairs = [(str(row.get("unit_id")), str(row.get("arm"))) for row in rows]
    if len(pairs) != len(set(pairs)):
        errors.append("duplicate_pair")
    if {arm for _, arm in pairs} != set(ARMS):
        errors.append("arm_roster")
    counts = Counter(unit_id for unit_id, _ in pairs)
    if len(counts) != 64 or any(count != len(ARMS) for count in counts.values()):
        errors.append("base_roster")
    return list(dict.fromkeys(errors))


def summarize_arms(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Keep unconditional error and selective risk on distinct denominators."""

    metrics: JsonDict = {}
    for arm in ARMS:
        arm_rows = [row for row in rows if row.get("arm") == arm]
        denominator = len(arm_rows)
        covered = [
            row
            for row in arm_rows
            if row.get("abstention") is not True and row.get("representation_valid") is True
        ]
        source_values = [row.get("source_fidelity") is True for row in arm_rows]
        claim_values = [row.get("claim_fidelity") is True for row in arm_rows]
        metrics[arm] = {
            "denominator": denominator,
            "source_fidelity": (
                sum(source_values) / denominator if arm != "direct_judge" and denominator else None
            ),
            "claim_fidelity": (
                sum(claim_values) / denominator if arm != "direct_judge" and denominator else None
            ),
            "coverage": len(covered) / denominator if denominator else 0.0,
            "decision_error": (
                sum(row.get("decision_correct") is not True for row in arm_rows) / denominator
                if denominator
                else 1.0
            ),
            "false_accept_rate": (
                sum(row.get("false_accept") is True for row in arm_rows) / denominator
                if denominator
                else 1.0
            ),
            "selective_risk": (
                sum(row.get("decision_correct") is not True for row in covered) / len(covered)
                if covered
                else None
            ),
            "selective_denominator": len(covered),
            "abstention_count": sum(row.get("abstention") is True for row in arm_rows),
            "missing_output_count": sum(
                row.get("missing_output_penalty") is True for row in arm_rows
            ),
        }
    return metrics


def summarize_costs(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Aggregate cost without mixing tokens or time into decision quality."""

    result: JsonDict = {}
    for arm in ARMS:
        arm_rows = [row for row in rows if row.get("arm") == arm]
        result[arm] = {
            "call_count": len(arm_rows),
            "prompt_tokens": sum(int(row.get("prompt_tokens") or 0) for row in arm_rows),
            "completion_tokens": sum(int(row.get("completion_tokens") or 0) for row in arm_rows),
            "elapsed_s": sum(float(row.get("elapsed_s") or 0.0) for row in arm_rows),
            "timeout_count": sum(row.get("timeout") is True for row in arm_rows),
            "transport_completed_count": sum(
                row.get("transport_complete") is True for row in arm_rows
            ),
        }
    return result


def _percentile(values: Sequence[float], probability: float) -> float:
    """Select one deterministic empirical percentile from frozen draws."""

    ordered = sorted(values)
    index = int(probability * (len(ordered) - 1))
    return ordered[index]


def paired_bootstrap(
    rows: Sequence[Mapping[str, Any]], seed: int, draws: int = BOOTSTRAP_DRAWS
) -> list[JsonDict]:
    """Resample base questions while keeping all three arm outcomes paired."""

    errors = paired_row_errors(rows)
    if errors:
        raise ValueError("paired_rows:" + ",".join(errors))
    by_unit: dict[str, dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for row in rows:
        by_unit[str(row["unit_id"])][str(row["arm"])] = row
    units = sorted(by_unit)
    generator = random.Random(seed)
    stored: dict[tuple[str, str], list[float]] = defaultdict(list)
    for _ in range(draws):
        sampled = [units[generator.randrange(len(units))] for _ in units]
        for comparison, control in COMPARISONS.items():
            error_differences = [
                float(by_unit[unit]["mention_pointer"].get("decision_correct") is not True)
                - float(by_unit[unit][control].get("decision_correct") is not True)
                for unit in sampled
            ]
            false_accept_differences = [
                float(by_unit[unit]["mention_pointer"].get("false_accept") is True)
                - float(by_unit[unit][control].get("false_accept") is True)
                for unit in sampled
            ]
            stored[(comparison, "decision_error_difference")].append(
                sum(error_differences) / len(error_differences)
            )
            stored[(comparison, "false_accept_difference")].append(
                sum(false_accept_differences) / len(false_accept_differences)
            )
    result = []
    for comparison, control in COMPARISONS.items():
        for metric in ("decision_error_difference", "false_accept_difference"):
            values = stored[(comparison, metric)]
            pointer_key = "decision_correct" if metric.startswith("decision") else "false_accept"
            point_values = []
            for unit in units:
                pointer_value = by_unit[unit]["mention_pointer"].get(pointer_key)
                control_value = by_unit[unit][control].get(pointer_key)
                if pointer_key == "decision_correct":
                    point_values.append(
                        float(pointer_value is not True) - float(control_value is not True)
                    )
                else:
                    point_values.append(float(pointer_value is True) - float(control_value is True))
            result.append(
                {
                    "comparison": comparison,
                    "treatment_arm": "mention_pointer",
                    "control_arm": control,
                    "metric": metric,
                    "estimate": sum(point_values) / len(point_values),
                    "ci95": [_percentile(values, 0.025), _percentile(values, 0.975)],
                    "bootstrap_draws": draws,
                    "independent_base_questions": len(units),
                }
            )
    return result


def _interval(rows: Sequence[Mapping[str, Any]], comparison: str, metric: str) -> Mapping[str, Any]:
    """Select one unique preregistered paired interval."""

    matches = [
        row for row in rows if row.get("comparison") == comparison and row.get("metric") == metric
    ]
    if len(matches) != 1:
        raise ValueError(f"interval:{comparison}:{metric}")
    return matches[0]


def acceptance_results(
    metrics: Mapping[str, Mapping[str, Any]],
    intervals: Sequence[Mapping[str, Any]],
    controls: Mapping[str, Any],
) -> list[JsonDict]:
    """Apply every frozen criterion without collapsing failed layers."""

    pointer = metrics["mention_pointer"]
    values = [
        (
            "source_fidelity_at_least_0_80",
            "extraction",
            0.80,
            pointer["source_fidelity"],
            pointer["source_fidelity"] >= 0.80,
        ),
        (
            "coverage_at_least_0_75",
            "extraction",
            0.75,
            pointer["coverage"],
            pointer["coverage"] >= 0.75,
        ),
    ]
    for comparison in COMPARISONS:
        error_upper = _interval(intervals, comparison, "decision_error_difference")["ci95"][1]
        false_upper = _interval(intervals, comparison, "false_accept_difference")["ci95"][1]
        values.extend(
            [
                (
                    f"{comparison}_error_ci95_upper_below_zero",
                    "decision",
                    "<0",
                    error_upper,
                    error_upper < 0,
                ),
                (
                    f"{comparison}_false_accept_ci95_upper_at_most_zero",
                    "decision",
                    "<=0",
                    false_upper,
                    false_upper <= 0,
                ),
            ]
        )
    values.extend(
        [
            (
                "positive_control_nondegenerate",
                "headroom",
                True,
                controls.get("non_degenerate"),
                controls.get("non_degenerate") is True,
            ),
            (
                "authority_boundary_controls",
                "execution",
                True,
                controls.get("authority_boundary_passed"),
                controls.get("authority_boundary_passed") is True,
            ),
        ]
    )
    return [
        {
            "criterion": criterion,
            "layer": layer,
            "expected_value": expected,
            "actual_value": actual,
            "evaluated": True,
            "passed": bool(passed),
        }
        for criterion, layer, expected, actual, passed in values
    ]


def classify_value(criteria: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Choose the narrow circular result only when every fixed gate passes."""

    failure = next((row for row in criteria if row.get("passed") is not True), None)
    if failure is None:
        return {
            "semantic_value_score": 1,
            "verdict_class": "circular_positive",
            "honest_verdict": "complete_circular_positive_semantic_value_fixed_criteria_passed",
            "failed_layer": None,
        }
    layer = str(failure.get("layer"))
    suffix = "uninformative_no_headroom" if layer == "headroom" else f"{layer}_criteria_failed"
    return {
        "semantic_value_score": 0,
        "verdict_class": "null",
        "honest_verdict": f"complete_null_semantic_value_{suffix}",
        "failed_layer": layer,
    }


def recompute_controls(bundle: Mapping[str, Any]) -> JsonDict:
    """Recompute semantic attacks and a bounded circular positive control."""

    fixture = bundle["fixture"]
    canary = bundle["canary"]
    mutations = {str(row.get("mutation")): row for row in fixture.get("mutation_rows", [])}
    held_out_pointer = [
        row
        for row in fixture.get("rows", [])
        if row.get("split") == "held_out" and row.get("arm") == "mention_pointer"
    ]
    control_rows = [
        row
        for row in canary.get("rows", [])
        if row.get("arm") in {"original_offset", "explicit_schema_offset_control"}
    ]
    control_accuracy = (
        sum(row.get("decision_correct") is True for row in control_rows) / len(control_rows)
        if control_rows
        else 0.0
    )
    upper = (
        sum(row.get("metric") == 1 for row in held_out_pointer) / len(held_out_pointer)
        if held_out_pointer
        else 0.0
    )
    acceptance = {
        str(row.get("criterion")): row.get("passed") is True
        for row in fixture.get("acceptance_gate_results", [])
    }
    selection = canary.get("selection_receipt", {})
    interventions = [
        {
            "control": "mention_id_permutation",
            "passed": mutations.get("mention_permutation", {}).get("passed") is True,
        },
        {
            "control": "surface_renaming",
            "passed": acceptance.get("permutation_equivariance") is True,
        },
        {
            "control": "relation_reversal",
            "passed": mutations.get("wrong_direction", {}).get("passed") is True,
        },
        {
            "control": "joint_support_deletion",
            "passed": mutations.get("missing_support", {}).get("passed") is True,
        },
        {
            "control": "authority_file_access_denial",
            "passed": selection.get("authority_fields_in_model_schedule") == 0
            and selection.get("held_out_units_opened") == 0,
        },
    ]
    authority_ok = all(row["passed"] for row in interventions)
    non_degenerate = upper > control_accuracy
    return {
        "status": "passed" if authority_ok else "failed",
        "gold_relation_upper_bound": upper,
        "control_decision_accuracy": control_accuracy,
        "oracle_headroom": upper - control_accuracy,
        "non_degenerate": non_degenerate,
        "headroom_diagnosis": "headroom_present" if non_degenerate else "no_headroom",
        "authority_boundary_passed": authority_ok,
        "interventions": interventions,
        "circular_control_only": True,
        "eligible_as_learned_verifier_evidence": False,
    }


def cold_replay_capture(bundle: Mapping[str, Any]) -> tuple[list[JsonDict], list[JsonDict]]:
    """Replay retained outputs through the shipped reducer before joining labels."""

    capture = bundle["capture"]
    public_rows, authority_rows = capture_module.load_held_out_manifests(
        bundle["paths"]["public"], bundle["paths"]["authority"]
    )
    schedule = list(capture.get("schedule") or [])
    if capture_module.schedule_errors(schedule, public_rows, authority_rows):
        raise ValueError("capture_schedule")
    retained = list(capture.get("raw_rows") or [])
    replayed = capture_module.replay_completion_rows(schedule, retained)
    paired = capture_module.score_semantics(schedule, replayed, public_rows, authority_rows)
    if paired != capture.get("paired_unit_rows"):
        raise ValueError("capture_semantic_rows")
    costs = capture_module.decoding_cost_rows(replayed)
    if costs != capture.get("decoding_cost_rows"):
        raise ValueError("capture_cost_rows")
    return paired, costs


def _unevaluated_acceptance() -> list[JsonDict]:
    """Keep every frozen gate visible when an external block prevents scoring."""

    criteria = [
        ("source_fidelity_at_least_0_80", "extraction", 0.80),
        ("coverage_at_least_0_75", "extraction", 0.75),
        ("pointer_vs_offset_error_ci95_upper_below_zero", "decision", "<0"),
        ("pointer_vs_offset_false_accept_ci95_upper_at_most_zero", "decision", "<=0"),
        ("pointer_vs_direct_error_ci95_upper_below_zero", "decision", "<0"),
        ("pointer_vs_direct_false_accept_ci95_upper_at_most_zero", "decision", "<=0"),
        ("positive_control_nondegenerate", "headroom", True),
        ("authority_boundary_controls", "execution", True),
    ]
    return [
        {
            "criterion": name,
            "layer": layer,
            "expected_value": expected,
            "actual_value": None,
            "evaluated": False,
            "passed": False,
        }
        for name, layer, expected in criteria
    ]


def base_artifact(run_date: str) -> JsonDict:
    """Create a schema-complete checkpoint before fallible source reads."""

    return {
        "schema": {
            "name": "carnot.experiment_7239_v637_semantic_audit",
            "version": 1,
            "experiment_id": EXPERIMENT_ID,
            "milestone": MILESTONE,
        },
        "status": "running",
        "run_date": run_date,
        "field_principles": deepcopy(FIELD_PRINCIPLES),
        "preconditions_checked": [],
        "inference_substrate": "blocked_no_run",
        "inference_substrate_class": "blocked_no_run",
        "execution_venue": "host",
        "execution_host": platform.node() or "unknown",
        "duration_s": 0.0,
        "MODEL_SPECS": [],
        "model_invoked": False,
        "source_artifact_hashes": {},
        "rows": [],
        "sample_size_budget": {
            "planned_independent_units": 64,
            "planned_arms": 3,
            "planned_comparison_rows": 192,
            "attempted_independent_units": 0,
            "completed_independent_units": 0,
            "completed_comparison_rows": 0,
            "censored_independent_units": 64,
            "bootstrap_draws": BOOTSTRAP_DRAWS,
            "stopping_rule": "reduce all 64 authentic held-out bases once or stop at the first external capture-chain block",
        },
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": "pending",
        "gate_check_summary": gate_summary(None),
        "verifier_is_oracle": True,
        "verdict_class": "partial",
        "honest_verdict": "partial_exp7239_running_checkpoint_only",
        "acceptance_gate_results": _unevaluated_acceptance(),
        "semantic_audit_complete_score": 0,
        "semantic_value_score": 0,
        "paired_comparison_rows": [],
        "authority_boundary": {},
        "positive_control_results": {"status": "not_run"},
        "timestamps": {"started_at_utc": _utc_now(), "completed_at_utc": None},
        "phase_spans": [],
        "current_invocation_counts": {
            "model_loads": 0,
            "generations": 0,
            "model_invocations": 0,
        },
        "calibration_observations": {},
        "upstream_disposition": {},
        "arm_metrics": {},
        "cost_metrics": {},
        "paired_interval_rows": [],
        "failed_layer": None,
        "validation_command_rows": [],
    }


def _seal(artifact: JsonDict, path: Path, started: float, *, terminal: bool = False) -> None:
    """Refresh measured duration and checksum immediately before an atomic write."""

    artifact["duration_s"] = time.monotonic() - started
    if terminal:
        artifact["timestamps"]["completed_at_utc"] = _utc_now()
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    atomic_write_json(path, artifact, allow_override=False, sort_keys=True)


def validate_artifact(value: object, root: Path | None = None) -> list[str]:
    """Cold-check terminal fields, blocked evidence, rows, gates, and hashes."""

    if not isinstance(value, Mapping):
        return ["artifact_mapping"]
    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in value]
    if missing:
        return [f"missing_required_field:{missing[0]}"]
    errors: list[str] = []
    if value.get("field_principles") != FIELD_PRINCIPLES:
        errors.append("field_principles")
    if value.get("run_date") != RUN_DATE:
        errors.append("run_date")
    if value.get("execution_venue") != "host" or not value.get("execution_host"):
        errors.append("execution_identity")
    duration = value.get("duration_s")
    if isinstance(duration, bool) or not isinstance(duration, (int, float)) or duration < 0:
        errors.append("duration_s")
    if (
        value.get("MODEL_SPECS") != []
        or value.get("model_invoked") is not False
        or value.get("current_invocation_counts")
        != {"model_loads": 0, "generations": 0, "model_invocations": 0}
    ):
        errors.append("model_contract")
    if value.get("random_seed") != RANDOM_SEED:
        errors.append("random_seed")
    if value.get("verifier_is_oracle") is not True:
        errors.append("verifier_is_oracle")
    if value.get("reproducibility_checksum") != artifact_checksum(value):
        errors.append("reproducibility_checksum")
    repository = root or find_repo_root(start=__file__)
    paths = _resolved_paths(repository)
    if value.get("source_artifact_hashes") != _source_hashes(paths):
        errors.append("source_artifact_hashes")
    if value.get("status") == "blocked":
        summary = value.get("gate_check_summary")
        if (
            value.get("verdict_class") != "blocked"
            or value.get("semantic_audit_complete_score") != 0
            or value.get("semantic_value_score") != 0
            or value.get("rows") != []
            or value.get("paired_comparison_rows") != []
            or value.get("inference_substrate") != "blocked_no_run"
            or value.get("inference_substrate_class") != "blocked_no_run"
        ):
            errors.append("blocked_terminal_state")
        if not isinstance(summary, Mapping) or summary.get("passed") is not False:
            errors.append("gate_check_summary")
        try:
            bundle = load_upstream_bundle(repository)
            expected_calibration = summarize_calibration(bundle)
        except (OSError, ValueError, json.JSONDecodeError, yaml.YAMLError):
            expected_calibration = None
        if value.get("calibration_observations") != expected_calibration:
            errors.append("calibration_observations")
        if value.get("positive_control_results", {}).get("status") != "not_run_external_block":
            errors.append("positive_control_results")
        if any(row.get("evaluated") is not False for row in value["acceptance_gate_results"]):
            errors.append("acceptance_gate_results")
        return list(dict.fromkeys(errors))
    if value.get("status") != "complete":
        errors.append("status")
        return list(dict.fromkeys(errors))
    paired = value.get("paired_comparison_rows")
    if not isinstance(paired, list) or paired_row_errors(paired):
        errors.append("paired_comparison_rows")
        return list(dict.fromkeys(errors))
    metrics = summarize_arms(paired)
    intervals = paired_bootstrap(paired, BOOTSTRAP_SEED, BOOTSTRAP_DRAWS)
    controls = value.get("positive_control_results")
    if value.get("arm_metrics") != metrics:
        errors.append("arm_metrics")
    if value.get("paired_interval_rows") != intervals:
        errors.append("paired_interval_rows")
    if not isinstance(controls, Mapping):
        errors.append("positive_control_results")
    else:
        criteria = acceptance_results(metrics, intervals, controls)
        outcome = classify_value(criteria)
        if value.get("acceptance_gate_results") != criteria:
            errors.append("acceptance_gate_results")
        if any(value.get(field) != expected for field, expected in outcome.items()):
            errors.append("terminal_classification")
    if (
        value.get("semantic_audit_complete_score") != 1
        or value.get("rows") != paired
        or value.get("inference_substrate") != "cpu_exact_solver_or_simulator"
        or value.get("inference_substrate_class") != "cpu_exact_solver_or_simulator"
    ):
        errors.append("complete_terminal_state")
    return list(dict.fromkeys(errors))


def attach_validation_receipts(
    artifact: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    root: Path | None = None,
) -> JsonDict:
    """Attach observed command results only to a cold-valid artifact."""

    if validate_artifact(artifact, root):
        raise ValueError("validation_receipt_source_artifact")
    required = {"command", "exit_code", "classification", "summary"}
    if any(set(row) != required for row in rows):
        raise ValueError("validation_receipt_schema")
    value = deepcopy(dict(artifact))
    value["validation_command_rows"] = deepcopy(list(rows))
    value["reproducibility_checksum"] = artifact_checksum(value)
    return value


def run_experiment(
    root: Path | None = None,
    run_date: str = RUN_DATE,
    *,
    output_root: Path | None = None,
    path_overrides: Mapping[str, Path] | None = None,
) -> JsonDict:
    """Produce the terminal audit or an exact external block disposition."""

    started = time.monotonic()
    repository = root or find_repo_root(start=__file__)
    destination = output_root or repository
    checkpoint = destination / CHECKPOINT_PATH
    result = destination / RESULT_PATH

    _progress(0, "start", "write schema-complete provisional checkpoint")
    artifact = base_artifact(run_date)
    _seal(artifact, checkpoint, started)
    _progress(0, "end", str(checkpoint))

    phase_started = time.monotonic()
    _progress(1, "start", "authenticate paths, hashes, imports, quarantine, and outputs")
    checks, bundle = collect_preconditions(
        repository, run_date, destination, path_overrides=path_overrides
    )
    artifact["preconditions_checked"] = checks
    paths = _resolved_paths(repository, path_overrides)
    artifact["source_artifact_hashes"] = _source_hashes(paths)
    artifact["phase_spans"].append(
        {
            "phase": 1,
            "name": "preconditions_and_upstream_authentication",
            "duration_s": time.monotonic() - phase_started,
        }
    )
    failure = next((row for row in checks if row.get("passed") is not True), None)
    _progress(1, "end", f"first_failure={failure.get('check') if failure else None}")

    phase_started = time.monotonic()
    _progress(2, "start", "preserve authenticated calibration observations")
    if bundle:
        artifact["calibration_observations"] = summarize_calibration(bundle)
        artifact["authority_boundary"] = authority_boundary(bundle)
    artifact["phase_spans"].append(
        {
            "phase": 2,
            "name": "calibration_diagnosis",
            "duration_s": time.monotonic() - phase_started,
        }
    )
    _progress(2, "end", f"observations={bool(bundle)} promoted=false")

    if failure is not None:
        _progress(3, "start", "held-out replay blocked by upstream gate")
        _progress(3, "end", "completed_units=0")
        _progress(4, "start", "paired bootstrap blocked by absent authentic rows")
        _progress(4, "end", "draws=0")
        _progress(5, "start", "positive controls not run after external block")
        artifact["positive_control_results"] = {
            "status": "not_run_external_block",
            "reason": failure["check"],
            "non_degenerate": None,
            "authority_boundary_passed": None,
        }
        _progress(5, "end", "status=not_run_external_block")
        _progress(6, "start", "classify terminal upstream disposition")
        artifact.update(
            {
                "status": "blocked",
                "gate_check_summary": gate_summary(failure),
                "verdict_class": "blocked",
                "honest_verdict": f"blocked_exp7239_{failure['check']}",
                "upstream_disposition": {
                    "action": "retire_same_verdict_lineage",
                    "continue_or_retire": "retire",
                    "first_failed_layer": "capture_authentication",
                    "failure": gate_summary(failure),
                },
                "failed_layer": "capture_authentication",
            }
        )
        _progress(6, "end", f"verdict={artifact['honest_verdict']}")
    else:
        _progress(3, "benchmark_start", "cold replay source and claim outputs")
        paired, costs = cold_replay_capture(bundle)
        artifact["rows"] = paired
        artifact["paired_comparison_rows"] = paired
        artifact["arm_metrics"] = summarize_arms(paired)
        artifact["cost_metrics"] = summarize_costs(costs)
        artifact["sample_size_budget"].update(
            {
                "attempted_independent_units": 64,
                "completed_independent_units": 64,
                "completed_comparison_rows": 192,
                "censored_independent_units": 0,
            }
        )
        artifact["inference_substrate"] = "cpu_exact_solver_or_simulator"
        artifact["inference_substrate_class"] = "cpu_exact_solver_or_simulator"
        _progress(3, "benchmark_end", "completed_units=64 rows=192")
        _progress(4, "benchmark_start", f"paired bootstrap draws={BOOTSTRAP_DRAWS}")
        artifact["paired_interval_rows"] = paired_bootstrap(paired, BOOTSTRAP_SEED, BOOTSTRAP_DRAWS)
        _progress(4, "benchmark_end", "interval_rows=4")
        _progress(5, "benchmark_start", "semantic interventions and oracle headroom")
        artifact["positive_control_results"] = recompute_controls(bundle)
        _progress(
            5,
            "benchmark_end",
            f"headroom={artifact['positive_control_results']['headroom_diagnosis']}",
        )
        _progress(6, "start", "apply frozen semantic-value criteria")
        artifact["acceptance_gate_results"] = acceptance_results(
            artifact["arm_metrics"],
            artifact["paired_interval_rows"],
            artifact["positive_control_results"],
        )
        artifact.update(classify_value(artifact["acceptance_gate_results"]))
        artifact["status"] = "complete"
        artifact["semantic_audit_complete_score"] = 1
        artifact["gate_check_summary"] = gate_summary(None)
        artifact["upstream_disposition"] = {
            "action": "continue" if artifact["semantic_value_score"] else "retire",
            "continue_or_retire": "continue" if artifact["semantic_value_score"] else "retire",
            "first_failed_layer": artifact["failed_layer"],
        }
        _progress(6, "end", f"verdict={artifact['honest_verdict']}")

    _progress(7, "validation_start", "cold-check terminal artifact before atomic write")
    artifact["duration_s"] = time.monotonic() - started
    artifact["timestamps"]["completed_at_utc"] = _utc_now()
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    errors = validate_artifact(artifact, repository)
    _progress(7, "validation_end", f"errors={errors}")
    if errors:
        raise ValueError(f"invalid Exp7239 artifact: {errors}")
    _progress(8, "write_start", f"atomic terminal artifact={result}")
    _seal(artifact, checkpoint, started)
    _seal(artifact, result, started, terminal=True)
    _progress(8, "write_end", str(result))
    return artifact


def replay_terminal_artifact(
    root: Path | None = None, *, output_root: Path | None = None
) -> JsonDict:
    """Independently recheck the shipped terminal artifact without inference."""

    repository = root or find_repo_root(start=__file__)
    destination = output_root or repository
    _progress(9, "benchmark_start", "independent terminal reducer replay")
    artifact = json.loads((destination / RESULT_PATH).read_text(encoding="utf-8"))
    errors = validate_artifact(artifact, repository)
    if errors:
        raise ValueError(f"terminal_artifact_validation:{errors}")
    bundle = load_upstream_bundle(repository)
    checks = upstream_gate_rows(bundle)
    failure = next((row for row in checks if row.get("passed") is not True), None)
    if artifact.get("status") == "blocked":
        if gate_summary(failure) != artifact.get("gate_check_summary"):
            raise ValueError("blocked_upstream_replay")
    else:
        paired, costs = cold_replay_capture(bundle)
        if paired != artifact.get("paired_comparison_rows"):
            raise ValueError("paired_replay")
        if summarize_costs(costs) != artifact.get("cost_metrics"):
            raise ValueError("cost_replay")
    _progress(9, "benchmark_end", f"status={artifact['status']} regenerated_model_calls=0")
    return artifact


def _date_argument(value: str) -> str:
    """Accept only the execution date fixed by the task contract."""

    if value != RUN_DATE:
        raise argparse.ArgumentTypeError(f"run date must be {RUN_DATE}")
    return value


def main(argv: Sequence[str] | None = None) -> int:
    """Build and cold-check the fixed-date terminal audit."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", required=True, type=_date_argument)
    args = parser.parse_args(argv)
    root = find_repo_root(start=__file__)
    artifact = run_experiment(root, args.date)
    errors = validate_artifact(artifact, root)
    if errors:
        print(f"[exp7239] invalid artifact: {errors}", flush=True)
        return 1
    print(
        f"[exp7239] terminal verdict={artifact['honest_verdict']} "
        f"complete={artifact['semantic_audit_complete_score']} "
        f"value={artifact['semantic_value_score']}",
        flush=True,
    )
    return 0
