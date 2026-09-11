"""Requalify the V635 span fixture under the registered CPU substrate.

The semantic work remains in the shipped V635 compiler, independent source
interpreter, and typed executor. This module owns the fresh V636 paths and
receipts so the quarantined Exp7208 verdict is evidence about a past failure,
not evidence that this run passed.

Spec refs: REQ-VERIFY-7222 and SCENARIO-VERIFY-7222-*.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from copy import deepcopy
from datetime import UTC, datetime
import json
import os
from pathlib import Path
import platform
import sys
import time
from typing import Any

from carnot import experiment_7208_v635_span_fixture as v635
from carnot.experiment_artifacts import atomic_write_bytes, atomic_write_json
from carnot.paths import repo_root as find_repo_root


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:  # pragma: no cover - depends on how Python starts.
    sys.path.insert(0, str(REPO_ROOT))

from scripts import adversarial_verify  # noqa: E402


JsonDict = dict[str, Any]

RUN_DATE = "20260911"
RANDOM_SEED = v635.RANDOM_SEED
SURFACE_SEED = v635.SURFACE_SEED
MODEL_SPECS: list[JsonDict] = []
INFERENCE_SUBSTRATE = "cpu_exact_solver_or_simulator"
INFERENCE_SUBSTRATE_CLASS = "cpu_exact_solver_or_simulator"
EXECUTION_VENUE = "host"
AUTHORITY_IMPORTS_CANDIDATE = v635.AUTHORITY_IMPORTS_CANDIDATE

RESULT_PATH = Path("results/experiment_7222_v636_span_fixture.json")
CHECKPOINT_PATH = Path("results/checkpoints/experiment_7222_v636_span_fixture.json")
RAW_DIR = Path("results/raw/experiment_7222")
PUBLIC_VIEW_PATH = RAW_DIR / "public.jsonl"
AUTHORITY_SIDECAR_PATH = RAW_DIR / "authority.jsonl"
FIXTURE_MANIFEST_PATH = RAW_DIR / "manifest.json"

MODULE_PATH = Path("python/carnot/experiment_7222_v636_span_fixture.py")
WRAPPER_PATH = Path("scripts/experiments/experiment_7222_v636_span_fixture.py")
TEST_PATH = Path("tests/python/test_experiment_7222_v636_span_fixture.py")
SPEC_PATH = Path("openspec/capabilities/constraint-verification/spec.md")
EXP7208_PATH = Path("results/experiment_7208_v635_span_fixture.json")

SOURCE_PATHS = {
    "agents": Path("AGENTS.md"),
    "claude": Path("CLAUDE.md"),
    "codex": Path("CODEX.md"),
    "research_program": Path("research-program.md"),
    "research_references": Path("research-references.md"),
    "exclusion_manifest": Path("ops/exclusion_manifest.yaml"),
    "e2e_test_plan": Path("ops/e2e-test-plan.md"),
    "v635_span_module": Path("python/carnot/experiment_7208_v635_span_fixture.py"),
    "typed_executor": Path("python/carnot/verify/experiment_7195_source_relation_executor.py"),
    "exp7208_artifact": EXP7208_PATH,
    "exp7209_artifact": Path("results/experiment_7209_v635_span_canary.json"),
    "adversarial_verifier": Path("scripts/adversarial_verify.py"),
    "row_consistency_lint": Path("scripts/verdict_row_consistency_lint.py"),
    "constraint_spec": SPEC_PATH,
    "module": MODULE_PATH,
    "entrypoint": WRAPPER_PATH,
    "focused_tests": TEST_PATH,
    "exp7196_artifact": v635.EXP7196_PATH,
    "exp7197_artifact": v635.EXP7197_PATH,
    "exp7196_raw_manifest": v635.RAW_MANIFEST_PATH,
}

REQUIRED_FIELD_PRINCIPLES = {
    "field_principles": (
        "Annotate actual values in this map; do not wrap arbitrary dictionaries as "
        "principle/value records."
    ),
    "status": (
        "Write a terminal artifact only when done or externally blocked; running "
        "checkpoints use a different path."
    ),
    "run_date": "Use 20260911 and record actual UTC timestamps, never copy an upstream run date.",
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
        "Only models actually invoked; [] for CPU/aggregation, mandated Qwen3.8 for every "
        "model task."
    ),
    "model_invoked": (
        "True only for actual model execution; upstream model outputs are cached evidence."
    ),
    "span_fixture_ready_score": (
        "One only after authentic reconstruction and clean unchanged verifier checks."
    ),
    "public_view_path": "Frozen model-visible inputs without authority.",
    "authority_sidecar_path": "Private labels used only for independent evaluation.",
    "fixture_manifest_path": "Exact paths, content hashes, counts and base splits.",
    "grammar_contract": "Syntax and bounded reference semantics remain separate.",
    "mutation_rows": "Concrete independent failures under span/semantic attacks.",
    "substrate_classifier_receipt": "Actual classification of new and historical artifacts.",
    "timestamps": "Actual UTC start and completion observations for this execution.",
    "split_manifest": "Frozen, disjoint base identities keep related variants in one split.",
    "lexical_control_rows": "The frozen lexical rule retains all easy and difficult units.",
    "readiness_checks": "Every conjunct is explicit; one failure keeps readiness at zero.",
    "upstream_diagnosis_rows": "Cached raw Qwen bytes are authenticated and diagnosed, not rerun.",
    "historical_exp7208_diagnosis": "The old flag stays visible and supplies no positive proof.",
}
REQUIRED_ARTIFACT_FIELDS = frozenset(REQUIRED_FIELD_PRINCIPLES)


# These names expose the exact production path that the new receipt qualifies.
build_panel = v635.build_panel
execute_panel = v635.execute_panel
lexical_control = v635.lexical_control


def canonical_json(value: Any) -> str:
    """Use one stable JSON spelling so a third party can reproduce every hash."""

    return v635.canonical_json(value)


def sha256_bytes(value: bytes) -> str:
    """Name SHA-256 beside the digest so a receipt never leaves the algorithm implicit."""

    return v635.sha256_bytes(value)


def sha256_file(path: Path) -> str:
    """Hash source bytes without newline or text-decoding changes."""

    return v635.sha256_file(path)


def _unwrap(value: Any) -> Any:
    """Unwrap only the exact two-key annotation form required by the task."""

    return v635._unwrap(value)


def artifact_checksum(artifact: Mapping[str, Any]) -> str:
    """Bind reproducible evidence while excluding process-local clocks."""

    stable = {
        key: value
        for key, value in artifact.items()
        if key not in {"duration_s", "timestamps", "reproducibility_checksum"}
    }
    return sha256_bytes(canonical_json(stable).encode("utf-8"))


def _progress(phase: int, event: str, detail: str) -> None:
    """Flush observed state so a slow native call cannot look like a silent stall."""

    print(f"[exp7222] phase {phase} {event}: {detail}", flush=True)


def _utc_now() -> str:
    """Record a real UTC observation instead of copying an upstream timestamp."""

    return datetime.now(UTC).isoformat()


def _paths(output_root: Path) -> dict[str, Path]:
    """Keep provisional, raw, and terminal outputs in their declared locations."""

    return {
        "public": output_root / PUBLIC_VIEW_PATH,
        "authority": output_root / AUTHORITY_SIDECAR_PATH,
        "manifest": output_root / FIXTURE_MANIFEST_PATH,
        "checkpoint": output_root / CHECKPOINT_PATH,
        "result": output_root / RESULT_PATH,
    }


def _gate_summary(failure: Mapping[str, Any] | None) -> JsonDict:
    """Retain both sides of a failed precondition so a block is actionable."""

    return v635._gate_summary(failure)


def _source_paths(root: Path, overrides: Mapping[str, Path] | None) -> dict[str, Path]:
    """Resolve every cited source while permitting isolated missing-input tests."""

    paths = {name: root / path for name, path in SOURCE_PATHS.items()}
    for name, path in (overrides or {}).items():
        paths[name] = path if path.is_absolute() else root / path
    return paths


def _source_hashes(paths: Mapping[str, Path]) -> JsonDict:
    """Bind cited code, documents, producer files, and frozen tuple settings."""

    hashes = {
        name: sha256_file(path) if path.is_file() else "missing" for name, path in paths.items()
    }
    hashes["tuple_schema"] = sha256_bytes(canonical_json(v635.TUPLE_SCHEMA).encode("utf-8"))
    hashes["model_settings"] = sha256_bytes(canonical_json(v635.MODEL_SETTINGS).encode("utf-8"))
    return hashes


def _preconditions(
    root: Path,
    run_date: str,
    output_root: Path,
    overrides: Mapping[str, Path] | None,
) -> tuple[list[JsonDict], dict[str, Path], JsonDict]:
    """Authenticate real producers first, then check task-owned sources and outputs."""

    checks, _, upstream = v635._preconditions(root, run_date, output_root, overrides)
    paths = _source_paths(root, overrides)
    if any(row.get("passed") is not True for row in checks):
        return checks, paths, upstream

    def record(row: JsonDict) -> bool:
        _progress(1, "check_start", str(row["check"]))
        checks.append(row)
        _progress(1, "check_end", f"{row['check']} passed={row['passed']}")
        return bool(row["passed"])

    for name in (
        "exp7208_artifact",
        "exp7209_artifact",
        "module",
        "entrypoint",
        "focused_tests",
    ):
        path = paths[name]
        if not record(
            v635._gate(
                "source_exists",
                name,
                "path",
                "existing_file",
                str(path),
                path.is_file() and os.access(path, os.R_OK),
            )
        ):
            return checks, paths, upstream

    try:
        spec_text = paths["constraint_spec"].read_text(encoding="utf-8")
        historical = json.loads(paths["exp7208_artifact"].read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        record(
            v635._gate(
                "source_parse",
                "exp7208|constraint_spec",
                "json_or_utf8",
                "valid",
                f"{type(exc).__name__}:{exc}",
                False,
            )
        )
        return checks, paths, upstream

    if not record(
        v635._gate(
            "driving_spec",
            "constraint_spec",
            "REQ-VERIFY-7222",
            True,
            "REQ-VERIFY-7222" in spec_text,
            "REQ-VERIFY-7222" in spec_text,
        )
    ):
        return checks, paths, upstream
    historical_state = {
        "flagged_adversarial": _unwrap(historical.get("flagged_adversarial")),
        "duration_s": _unwrap(historical.get("duration_s")),
        "inference_substrate": _unwrap(historical.get("inference_substrate")),
        "inference_substrate_class": _unwrap(historical.get("inference_substrate_class")),
    }
    if not record(
        v635._gate(
            "historical_quarantine_preserved",
            "exp7208_historical_only",
            "flagged_adversarial",
            True,
            historical_state,
            historical_state["flagged_adversarial"] is True
            and historical_state["inference_substrate"] != INFERENCE_SUBSTRATE,
        )
    ):
        return checks, paths, upstream

    destinations = _paths(output_root)
    for name, path in destinations.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        writable = os.access(path.parent, os.W_OK)
        if not record(
            v635._gate(
                "output_destination",
                name,
                "parent",
                "writable_directory",
                str(path.parent),
                writable,
            )
        ):
            return checks, paths, upstream
    return checks, paths, {**upstream, "historical_exp7208": historical}


def mutation_rows() -> list[JsonDict]:
    """Replay V635 attacks and add an explicit inverse-relation qualification unit."""

    rows = deepcopy(v635.span_mutation_checks())
    for row in rows:
        row["check_failed_as_required"] = (
            row["mutation"]
            in {
                "out_of_range",
                "cross_document",
                "wrong_sentence",
                "type_mismatch",
                "unsupported_predicate",
                "invalid_polarity",
                "excess_source_relations",
                "direction_reversal",
                "negation",
                "support_removed",
            }
            and row["passed"] is True
        )
    source = "Aster precedes Brin."
    claim = "Brin follows Aster."
    candidate = v635._candidate_decision(source, claim)
    authority = v635.authority_decision(source, claim)
    rows.append(
        {
            "mutation": "inverse_relation",
            "expected": "supported",
            "observed": candidate["decision"],
            "authority_observed": authority,
            "passed": candidate["decision"] == authority == "supported",
            "check_failed_as_required": False,
        }
    )
    return rows


def _manifest(panel: Mapping[str, Any], public_bytes: bytes, authority_bytes: bytes) -> JsonDict:
    """Add task-owned paths and exact counts to the shipped grammar and row receipts."""

    manifest = v635._fixture_manifest(panel, public_bytes, authority_bytes)
    manifest["schema"] = "carnot.exp7222.span_fixture.v1"
    manifest["paths"] = {
        "public_view": PUBLIC_VIEW_PATH.as_posix(),
        "authority_sidecar": AUTHORITY_SIDECAR_PATH.as_posix(),
        "fixture_manifest": FIXTURE_MANIFEST_PATH.as_posix(),
    }
    manifest["counts"] = {
        "public_rows": len(panel["public_rows"]),
        "authority_rows": len(panel["authority_rows"]),
        "base_rows": len(panel["split_manifest"]["base_rows"]),
        "grammar_requests": manifest["grammar_manifest"]["request_count"],
        "unique_grammar_requests": manifest["grammar_manifest"]["unique_request_count"],
        "row_receipts": len(manifest["row_receipts"]),
    }
    manifest["producer"] = MODULE_PATH.as_posix()
    manifest["reused_v635_module"] = SOURCE_PATHS["v635_span_module"].as_posix()
    return manifest


def _grammar_contract(manifest: Mapping[str, Any], panel: Mapping[str, Any]) -> JsonDict:
    """Summarize the exact grammars without loading even tokenizer model metadata."""

    completions = [
        canonical_json(
            v635.extract_public_completion(str(row[field]).encode("utf-8"), call_type)
        ).encode("utf-8")
        for row in panel["public_rows"]
        for call_type, field in (("source", "source_text"), ("claim", "claim_text"))
    ]
    contract = v635._grammar_contract(manifest, completions, None)
    contract["embedded_tokenizer_receipt"] = {
        "embedded_tokenizer_available": False,
        "defer_reason": "no_model_load_per_exp7222_contract",
        "model_invoked": False,
    }
    return contract


def _historical_receipt(path: Path, historical: Mapping[str, Any]) -> JsonDict:
    """Run the unchanged verifier on Exp7208 while keeping its verdict non-authoritative."""

    return {
        "path": EXP7208_PATH.as_posix(),
        "artifact_sha256": sha256_file(path),
        "stored_flagged_adversarial": _unwrap(historical.get("flagged_adversarial")),
        "duration_s": _unwrap(historical.get("duration_s")),
        "inference_substrate": _unwrap(historical.get("inference_substrate")),
        "inference_substrate_class": _unwrap(historical.get("inference_substrate_class")),
        "duration_floor": adversarial_verify.duration_floor_for_artifact(dict(historical)),
        "verifier_report": adversarial_verify.verify_artifact(path, declared=False),
        "used_as_positive_proof": False,
    }


def _base_artifact(run_date: str) -> JsonDict:
    """Create a schema-complete checkpoint before any fallible source read."""

    return {
        "schema": "carnot.exp7222.v636_span_fixture.v1",
        "field_principles": deepcopy(REQUIRED_FIELD_PRINCIPLES),
        "status": "running",
        "run_date": run_date,
        "preconditions_checked": [],
        "inference_substrate": "blocked_no_run",
        "inference_substrate_class": "blocked_no_run",
        "execution_venue": EXECUTION_VENUE,
        "execution_host": platform.node() or "unknown",
        "duration_s": 0.0,
        "source_artifact_hashes": {},
        "rows": [],
        "sample_size_budget": {
            "planned_rows": 320,
            "attempted_rows": 0,
            "completed_rows": 0,
            "censored_rows": 320,
            "independent_base_cases": 80,
            "held_out_rows": 256,
            "variants_per_base": 4,
        },
        "random_seed": RANDOM_SEED,
        "surface_render_seed": SURFACE_SEED,
        "reproducibility_checksum": "pending",
        "gate_check_summary": _gate_summary(None),
        "verifier_is_oracle": True,
        "verdict_class": "partial",
        "honest_verdict": "partial_exp7222_running",
        "MODEL_SPECS": [],
        "model_invoked": False,
        "span_fixture_ready_score": 0,
        "public_view_path": PUBLIC_VIEW_PATH.as_posix(),
        "authority_sidecar_path": AUTHORITY_SIDECAR_PATH.as_posix(),
        "fixture_manifest_path": FIXTURE_MANIFEST_PATH.as_posix(),
        "grammar_contract": {},
        "mutation_rows": [],
        "substrate_classifier_receipt": {},
        "timestamps": {"started_at_utc": _utc_now(), "completed_at_utc": None},
        "split_manifest": {},
        "lexical_control_rows": [],
        "readiness_checks": {},
        "upstream_diagnosis_rows": [],
        "historical_exp7208_diagnosis": {},
    }


def _seal(artifact: JsonDict, path: Path, started: float) -> None:
    """Refresh measured duration and checksum immediately before an atomic write."""

    artifact["duration_s"] = time.monotonic() - started
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    atomic_write_json(path, artifact, allow_override=False, sort_keys=True)


def _blocked_artifact(
    artifact: JsonDict,
    checks: Sequence[Mapping[str, Any]],
    sources: Mapping[str, Path],
    started: float,
) -> JsonDict:
    """Finish an external block without claiming that CPU qualification ran."""

    failure = next((row for row in checks if row.get("passed") is not True), None)
    artifact["status"] = "blocked"
    artifact["preconditions_checked"] = list(checks)
    artifact["source_artifact_hashes"] = _source_hashes(sources)
    artifact["gate_check_summary"] = _gate_summary(failure)
    artifact["verdict_class"] = "blocked"
    artifact["honest_verdict"] = f"blocked_exp7222_{failure['check'] if failure else 'unknown'}"
    artifact["timestamps"]["completed_at_utc"] = _utc_now()
    artifact["duration_s"] = time.monotonic() - started
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    return artifact


def _read_jsonl(path: Path) -> list[JsonDict]:
    """Read exact object rows from one sealed JSONL file."""

    rows: list[JsonDict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError("JSONL row must be an object")
        rows.append(value)
    return rows


def _readiness(
    artifact: Mapping[str, Any], panel: Mapping[str, Any], manifest: Mapping[str, Any]
) -> JsonDict:
    """Compute every readiness conjunct from fresh bytes and complete row evidence."""

    rows = artifact["rows"]
    mutations = artifact["mutation_rows"]
    grammar = artifact["grammar_contract"]
    return {
        "all_rows": len(rows) == 320 and all(row.get("metric") == 1 for row in rows),
        "test_rows": sum(row.get("split") == "test" for row in rows) == 256,
        "split_hashes": panel["split_manifest"]["split_hashes_disjoint"] is True,
        "mutations": bool(mutations) and all(row.get("passed") is True for row in mutations),
        "grammar_serialization": grammar.get("serialization_checks_passed") is True,
        "sealed_counts": manifest["counts"]
        == {
            "public_rows": 320,
            "authority_rows": 320,
            "base_rows": 80,
            "grammar_requests": 640,
            "unique_grammar_requests": manifest["grammar_manifest"]["unique_request_count"],
            "row_receipts": 320,
        },
        "source_authentication": all(
            row.get("passed") is True for row in artifact["preconditions_checked"]
        ),
        "no_model_invocation": artifact.get("MODEL_SPECS") == []
        and artifact.get("model_invoked") is False,
        "independent_authority": AUTHORITY_IMPORTS_CANDIDATE is False,
    }


def validate_artifact(
    artifact: object,
    root: Path | None = None,
    *,
    output_root: Path | None = None,
) -> list[str]:
    """Cold-replay terminal fields, raw bytes, semantics, receipts, and checksum."""

    if not isinstance(artifact, Mapping):
        return ["artifact_mapping"]
    missing = sorted(REQUIRED_ARTIFACT_FIELDS - set(artifact))
    if missing:
        return [f"missing_required_field:{missing[0]}"]
    errors: list[str] = []
    if artifact.get("field_principles") != REQUIRED_FIELD_PRINCIPLES:
        errors.append("field_principles")
    if artifact.get("run_date") != RUN_DATE:
        errors.append("run_date")
    if artifact.get("execution_venue") != EXECUTION_VENUE or not artifact.get("execution_host"):
        errors.append("execution_identity")
    if artifact.get("MODEL_SPECS") != [] or artifact.get("model_invoked") is not False:
        errors.append("model_contract")
    if artifact.get("verifier_is_oracle") is not True:
        errors.append("verifier_is_oracle")
    duration = artifact.get("duration_s")
    if isinstance(duration, bool) or not isinstance(duration, (int, float)) or duration < 0:
        errors.append("duration_s")
    if artifact.get("random_seed") != RANDOM_SEED:
        errors.append("random_seed")
    if artifact.get("reproducibility_checksum") != artifact_checksum(artifact):
        errors.append("reproducibility_checksum")
    if artifact.get("status") == "blocked":
        if (
            artifact.get("inference_substrate") != "blocked_no_run"
            or artifact.get("inference_substrate_class") != "blocked_no_run"
            or artifact.get("verdict_class") != "blocked"
            or artifact.get("span_fixture_ready_score") != 0
        ):
            errors.append("blocked_terminal_state")
        summary = artifact.get("gate_check_summary")
        if not isinstance(summary, Mapping) or summary.get("passed") is not False:
            errors.append("gate_check_summary")
        return errors
    if artifact.get("status") != "complete":
        errors.append("status")
        return errors
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate")
    if artifact.get("inference_substrate_class") != INFERENCE_SUBSTRATE_CLASS:
        errors.append("inference_substrate_class")
    if (
        artifact.get("verdict_class") != "circular_positive"
        or artifact.get("span_fixture_ready_score") != 1
    ):
        errors.append("readiness_terminal_state")

    budget = artifact.get("sample_size_budget")
    if not isinstance(budget, Mapping) or budget != {
        "planned_rows": 320,
        "attempted_rows": 320,
        "completed_rows": 320,
        "censored_rows": 0,
        "independent_base_cases": 80,
        "held_out_rows": 256,
        "variants_per_base": 4,
    }:
        errors.append("sample_size_budget")

    repo = root or find_repo_root(start=__file__)
    destination = output_root or repo
    paths = _paths(destination)
    try:
        public_rows = _read_jsonl(paths["public"])
        authority_rows = _read_jsonl(paths["authority"])
        manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError):
        errors.append("sealed_fixture_files")
        return errors
    panel = build_panel()
    public_bytes = v635.jsonl_bytes(panel["public_rows"])
    authority_bytes = v635.jsonl_bytes(panel["authority_rows"])
    expected_manifest = _manifest(panel, public_bytes, authority_bytes)
    if public_rows != panel["public_rows"]:
        errors.append("public_view")
    if authority_rows != panel["authority_rows"]:
        errors.append("authority_sidecar")
    if manifest != expected_manifest:
        errors.append("fixture_manifest")
    if artifact.get("split_manifest") != panel["split_manifest"]:
        errors.append("split_manifest")
    expected_rows = execute_panel(public_rows, authority_rows)
    if artifact.get("rows") != expected_rows:
        errors.append("rows")
    if artifact.get("lexical_control_rows") != lexical_control(public_rows, authority_rows):
        errors.append("lexical_control_rows")
    if artifact.get("mutation_rows") != mutation_rows():
        errors.append("mutation_rows")
    expected_grammar = _grammar_contract(manifest, panel)
    if artifact.get("grammar_contract") != expected_grammar:
        errors.append("grammar_contract")
    hashes = artifact.get("source_artifact_hashes")
    expected_raw_hashes = {
        "public_view": sha256_file(paths["public"]),
        "authority_sidecar": sha256_file(paths["authority"]),
        "fixture_manifest": sha256_file(paths["manifest"]),
    }
    if not isinstance(hashes, Mapping) or any(
        hashes.get(name) != value for name, value in expected_raw_hashes.items()
    ):
        errors.append("source_artifact_hashes")
    receipt = artifact.get("substrate_classifier_receipt")
    if not isinstance(receipt, Mapping):
        errors.append("substrate_classifier_receipt")
    else:
        old = receipt.get("historical_exp7208")
        candidate = receipt.get("candidate_checkpoint")
        if (
            not isinstance(old, Mapping)
            or old.get("stored_flagged_adversarial") is not True
            or not isinstance(old.get("verifier_report"), Mapping)
            or old["verifier_report"].get("flag_count", 0) < 1
            or old.get("used_as_positive_proof") is not False
            or not isinstance(candidate, Mapping)
            or not isinstance(candidate.get("duration_floor"), Mapping)
            or candidate["duration_floor"].get("min_duration_s") != 0.0001
            or not isinstance(candidate.get("verifier_report"), Mapping)
            or candidate["verifier_report"].get("flag_count") != 0
        ):
            errors.append("substrate_classifier_receipt")
    expected_readiness = _readiness(artifact, panel, manifest)
    if artifact.get("readiness_checks") != expected_readiness or not all(
        expected_readiness.values()
    ):
        errors.append("readiness_checks")
    return errors


def build_artifact(
    root: Path,
    run_date: str,
    *,
    output_root: Path | None = None,
    path_overrides: Mapping[str, Path] | None = None,
) -> JsonDict:
    """Execute one new CPU qualification or write an exact external block."""

    started = time.monotonic()
    destination = output_root or root
    paths = _paths(destination)
    paths["checkpoint"].parent.mkdir(parents=True, exist_ok=True)
    paths["result"].parent.mkdir(parents=True, exist_ok=True)
    _progress(0, "start", "write schema-complete running checkpoint before checks")
    artifact = _base_artifact(run_date)
    _seal(artifact, paths["checkpoint"], started)
    _progress(0, "end", str(paths["checkpoint"]))

    _progress(1, "start", "authenticate producers, sources, quarantine state, and outputs")
    checks, sources, upstream = _preconditions(root, run_date, destination, path_overrides)
    artifact["preconditions_checked"] = checks
    failure = next((row for row in checks if row.get("passed") is not True), None)
    if failure is not None:
        artifact = _blocked_artifact(artifact, checks, sources, started)
        _progress(1, "end", f"blocked={failure['check']}")
        _progress(8, "validation_start", "cold-check terminal external block")
        errors = validate_artifact(artifact, root, output_root=destination)
        _progress(8, "validation_end", f"errors={errors}")
        if errors:
            raise ValueError(f"invalid blocked Exp7222 artifact: {errors}")
        _progress(9, "write_start", "atomically write blocked checkpoint and terminal artifact")
        _seal(artifact, paths["checkpoint"], started)
        _seal(artifact, paths["result"], started)
        _progress(9, "write_end", str(paths["result"]))
        return artifact
    artifact["source_artifact_hashes"] = _source_hashes(sources)
    _seal(artifact, paths["checkpoint"], started)
    _progress(1, "end", "all exact preconditions passed")

    _progress(2, "start", "diagnose historical flag and authenticate cached raw Qwen evidence")
    historical_path = sources["exp7208_artifact"]
    artifact["historical_exp7208_diagnosis"] = _historical_receipt(
        historical_path, upstream["historical_exp7208"]
    )
    artifact["upstream_diagnosis_rows"] = v635.diagnose_exp7196(root)
    _seal(artifact, paths["checkpoint"], started)
    _progress(2, "end", "Exp7208 flag retained and 576 cached raw calls authenticated")

    _progress(3, "start", "reconstruct public and private V635 views under Exp7222 raw path")
    panel = build_panel()
    public_bytes = v635.jsonl_bytes(panel["public_rows"])
    authority_bytes = v635.jsonl_bytes(panel["authority_rows"])
    atomic_write_bytes(paths["public"], public_bytes, allow_override=False)
    atomic_write_bytes(paths["authority"], authority_bytes, allow_override=False)
    artifact["split_manifest"] = deepcopy(panel["split_manifest"])
    _seal(artifact, paths["checkpoint"], started)
    _progress(3, "end", "public=320 authority=320 bases=80 test_rows=256")

    _progress(4, "benchmark_start", "run compiler, closure, typed executor, and independent labels")
    artifact["rows"] = execute_panel(panel["public_rows"], panel["authority_rows"])
    artifact["lexical_control_rows"] = lexical_control(
        panel["public_rows"], panel["authority_rows"]
    )
    artifact["sample_size_budget"].update(
        {"attempted_rows": 320, "completed_rows": 320, "censored_rows": 0}
    )
    artifact["inference_substrate"] = INFERENCE_SUBSTRATE
    artifact["inference_substrate_class"] = INFERENCE_SUBSTRATE_CLASS
    _seal(artifact, paths["checkpoint"], started)
    _progress(4, "benchmark_end", "completed=320/320 censored=0 model_invoked=false")

    _progress(5, "start", "run adverse mutations and serialize request-bound grammars")
    artifact["mutation_rows"] = mutation_rows()
    manifest = _manifest(panel, public_bytes, authority_bytes)
    atomic_write_json(paths["manifest"], manifest, allow_override=False, sort_keys=True)
    artifact["grammar_contract"] = _grammar_contract(manifest, panel)
    artifact["source_artifact_hashes"].update(
        {
            "public_view": sha256_file(paths["public"]),
            "authority_sidecar": sha256_file(paths["authority"]),
            "fixture_manifest": sha256_file(paths["manifest"]),
        }
    )
    _seal(artifact, paths["checkpoint"], started)
    _progress(5, "end", f"mutations={len(artifact['mutation_rows'])} grammar_requests=640")

    _progress(6, "start", "compute readiness from regenerated bytes and complete rows")
    artifact["readiness_checks"] = _readiness(artifact, panel, manifest)
    artifact["status"] = "complete"
    artifact["span_fixture_ready_score"] = int(all(artifact["readiness_checks"].values()))
    artifact["verdict_class"] = (
        "circular_positive" if artifact["span_fixture_ready_score"] else "disqualified"
    )
    artifact["honest_verdict"] = (
        "complete_circular_positive_span_fixture_ready_no_distinct_verifier_value"
        if artifact["span_fixture_ready_score"]
        else "complete_disqualified_span_fixture_not_ready"
    )
    artifact["gate_check_summary"] = _gate_summary(None)
    _seal(artifact, paths["checkpoint"], started)
    _progress(6, "end", f"internal_ready={artifact['span_fixture_ready_score']}")

    _progress(7, "classifier_start", "classify candidate and run full verifier on checkpoint")
    candidate_floor = adversarial_verify.duration_floor_for_artifact(artifact)
    candidate_report = adversarial_verify.verify_artifact(paths["checkpoint"], declared=False)
    artifact["substrate_classifier_receipt"] = {
        "historical_exp7208": artifact["historical_exp7208_diagnosis"],
        "candidate_checkpoint": {
            "path": CHECKPOINT_PATH.as_posix(),
            "duration_floor": candidate_floor,
            "verifier_report": candidate_report,
        },
    }
    _seal(artifact, paths["checkpoint"], started)
    candidate_report = adversarial_verify.verify_artifact(paths["checkpoint"], declared=False)
    artifact["substrate_classifier_receipt"]["candidate_checkpoint"]["verifier_report"] = (
        candidate_report
    )
    if candidate_report.get("flag_count") != 0:
        artifact["span_fixture_ready_score"] = 0
        artifact["verdict_class"] = "disqualified"
        artifact["honest_verdict"] = "complete_disqualified_candidate_verifier_flagged"
    _seal(artifact, paths["checkpoint"], started)
    _progress(
        7,
        "classifier_end",
        f"floor={candidate_floor} flags={candidate_report.get('flag_count')}",
    )

    _progress(8, "validation_start", "cold-replay fields, raw hashes, semantics, and receipts")
    artifact["timestamps"]["completed_at_utc"] = _utc_now()
    artifact["duration_s"] = time.monotonic() - started
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    errors = validate_artifact(artifact, root, output_root=destination)
    _progress(8, "validation_end", f"errors={errors}")
    if errors:
        raise ValueError(f"invalid Exp7222 artifact: {errors}")

    _progress(9, "write_start", "atomically write stable checkpoint and terminal deliverable")
    _seal(artifact, paths["checkpoint"], started)
    _seal(artifact, paths["result"], started)
    _progress(9, "write_end", str(paths["result"]))
    return artifact


def _date_argument(value: str) -> str:
    """Accept only the execution date fixed by the V636 task contract."""

    if value != RUN_DATE:
        raise argparse.ArgumentTypeError(f"run date must be {RUN_DATE}")
    return value


def main(argv: Sequence[str] | None = None) -> int:
    """Build the fresh fixture and exit as soon as its terminal artifact validates."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", required=True, type=_date_argument)
    args = parser.parse_args(argv)
    root = find_repo_root(start=__file__)
    artifact = build_artifact(root, args.date)
    errors = validate_artifact(artifact, root)
    if errors:
        print(f"[exp7222] invalid artifact: {errors}", file=sys.stderr, flush=True)
        return 1
    print(
        f"[exp7222] complete verdict={artifact['honest_verdict']} "
        f"score={artifact['span_fixture_ready_score']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through the thin wrapper.
    raise SystemExit(main())
