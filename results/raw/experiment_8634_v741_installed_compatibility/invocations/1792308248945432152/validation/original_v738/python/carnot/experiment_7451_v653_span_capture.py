"""Repeat the sealed span comparison after the capture lifecycle repair.

The V652 module owns model serving, raw persistence, pairing, and metrics. This
module changes only the V653 identity, prerequisite, callback disposition, and
completion contract. Keeping one live engine prevents the repair repeat from
silently changing the scientific question.

Spec refs: REQ-VERIFY-7451 and SCENARIO-VERIFY-7451-*.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from copy import deepcopy
import json
from pathlib import Path
import time
from typing import Any

from carnot import experiment_7442_v652_span_capture as shared
from carnot.experiment_7358_v646_validation_contract import AffectedManifest
from carnot.reporting.experiment_7303_validation_scope import CommandSpec


JsonDict = dict[str, Any]
RUN_DATE = "20260920"
MILESTONE = "2026.09.653"
PHASE = 2
EXPERIMENT_ID = "exp7451-v653-span-capture"
TASK_ID = "experiment_7451_v653_span_capture"
SCHEMA = "carnot.exp7451.v653.span_capture.v1"
REPO_ROOT = Path(__file__).resolve().parents[2]

RESULT_PATH = Path("results/experiment_7451_v653_span_capture.json")
RAW_DIR = Path("results/raw/experiment_7451_v653_span_capture")
MODULE_PATH = Path("python/carnot/experiment_7451_v653_span_capture.py")
WRAPPER_PATH = Path("scripts/experiments/experiment_7451_v653_span_capture.py")
TEST_PATH = Path("tests/python/test_experiment_7451_v653_span_capture.py")
SPEC_PATH = Path("openspec/capabilities/constraint-verification/spec.md")
LIFECYCLE_PATH = Path("results/experiment_7448_v653_capture_lifecycle.json")
LIFECYCLE_MODULE_PATH = Path("python/carnot/experiment_7448_v653_capture_lifecycle.py")
EXPECTED_LIFECYCLE_SHA256 = (
    "sha256:7cb66f9072a5b8906a93a2b043fe43b4b01a1b4d47cf00e4fc9d13d63a97f640"
)

MODEL_SPECS = ["unsloth/Qwen3.8-27B-GGUF"]
INFERENCE_SUBSTRATE = "owned_native_cuda_llama_cpp_repaired_span_generation"
INFERENCE_SUBSTRATE_CLASS = "model_bounded_generation"
EXECUTION_VENUE = "host"
RANDOM_SEED = 6_537_451

AFFECTED_CHECK_NAMES = shared.AFFECTED_CHECK_NAMES
TERMINAL_CHECK_NAMES = shared.TERMINAL_CHECK_NAMES
VALIDATION_MANIFEST = AffectedManifest(
    experiment_id=EXPERIMENT_ID,
    test_paths=(
        TEST_PATH.as_posix(),
        "tests/python/test_experiment_7442_v652_span_capture.py",
        "tests/python/test_experiment_7448_v653_capture_lifecycle.py",
    ),
    changed_modules=(MODULE_PATH.as_posix(),),
    static_paths=(WRAPPER_PATH.as_posix(),),
)

FIELD_PRINCIPLES = deepcopy(shared.FIELD_PRINCIPLES)
FIELD_PRINCIPLES.update(
    {
        "schema": "Use the exact V653 schema, experiment identity, milestone, phase, and terminal status.",
        "preconditions_checked": "Authenticate the sealed protocol and exact Exp7448 lifecycle operands before model work.",
        "duration_s": "Measure real current work and keep model, computation, validation, and cold replay separate.",
        "span_capture_complete_score": "Set one when every planned call is terminal or explicitly unstarted and runtime integrity is clean.",
        "development_gate": "Separate correct-empty transport from usable nonempty factual extraction for all eight canary calls.",
        "semantic_scope": "Keep real-paragraph mechanical metrics separate from constructed exact qualifier authority.",
        "constructed_qualifier_delta": "Report the synthetic delta only after development opens evaluation.",
    }
)
REQUIRED_FIELDS = frozenset(FIELD_PRINCIPLES)

SEMANTIC_SCOPE = {
    "real_paragraphs": "unannotated_extraction_metrics_only",
    "constructed_pairs": "synthetic_exact_qualifier_authority",
    "natural_language_truth_established": False,
}

artifact_checksum = shared.artifact_checksum
sha256_file = shared.sha256_file
atomic_json = shared.atomic_json

_SHARED_BUILD_CAPTURE_ROW = shared.build_capture_row
_SHARED_REDUCE_DEVELOPMENT_GATE = shared.reduce_development_gate
_SHARED_BASE_ARTIFACT = shared._base_artifact
_SHARED_BUILD_BLOCKED_ARTIFACT = shared.build_blocked_artifact
_SHARED_MEASURED_ARTIFACT = shared._measured_artifact
_SHARED_COLLECT_PRECONDITIONS = shared.collect_preconditions
_SHARED_BUILD_FIXTURE_ARTIFACT = shared.build_fixture_artifact
_SHARED_VALIDATE_ARTIFACT = shared.validate_artifact
_SHARED_RUN_EXPERIMENT = shared.run_experiment


def progress(started: float, phase: str, event: str, **details: Any) -> None:  # pragma: no cover
    """Flush each phase and long-operation boundary for the outer conductor."""

    suffix = " ".join(f"{key}={value}" for key, value in sorted(details.items()))
    print(
        f"[exp7451] phase={phase} event={event} elapsed_s={time.monotonic() - started:.3f}"
        + (f" {suffix}" if suffix else ""),
        flush=True,
    )


def load_object(path: Path) -> JsonDict:
    """Read one JSON object and return an empty mapping for unusable bytes."""

    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _gate(
    check: str,
    field: str,
    operator: str,
    expected: Any,
    observed: Any,
    passed: bool,
) -> JsonDict:
    """Keep one lifecycle prerequisite as ordinary independently readable data."""

    return {
        "check": check,
        "category": "precondition",
        "operator": operator,
        "expected": deepcopy(expected),
        "observed": deepcopy(observed),
        "passed": bool(passed),
        "principle": "Only the exact ready, eligible, and unflagged lifecycle repair can authorize this repeat.",
        "upstream": "exp7448-v653-capture-lifecycle",
        "path": LIFECYCLE_PATH.as_posix(),
        "field": field,
    }


def lifecycle_gate_rows(value: Mapping[str, Any]) -> list[JsonDict]:
    """Authenticate the exact Exp7448 identity and three structured prerequisites."""

    declarations = (
        (
            "capture_lifecycle_identity",
            "experiment_id",
            "==",
            "exp7448-v653-capture-lifecycle",
        ),
        ("capture_lifecycle_ready", "capture_lifecycle_ready_score", "==", 1),
        ("capture_lifecycle_verdict", "verdict_class", "in", ["null", "positive"]),
        ("capture_lifecycle_unflagged", "flagged_adversarial", "==", False),
    )
    rows: list[JsonDict] = []
    for check, field, operator, expected in declarations:
        observed = value.get(field)
        passed = observed == expected if operator == "==" else observed in expected
        rows.append(_gate(check, field, operator, expected, observed, passed))
    return rows


def _development_disposition(row: Mapping[str, Any]) -> str:
    """Classify transport and extraction without treating an empty reply as coverage."""

    if row.get("attempted") is not True:
        return "unstarted"
    if row.get("terminal_state") != "response":
        return "transport_failure"
    if row.get("finish_reason") in {"length", "max_tokens"}:
        return "truncated"
    if not str(row.get("raw_reply") or "").strip():
        return "missing_output"
    if row.get("parse_valid") is not True:
        return "malformed"
    if not list(row.get("claims") or []):
        return "correct_empty"
    if row.get("completed_valid_output") is True:
        return "usable_nonempty"
    return "malformed"


def build_capture_row(schedule: Mapping[str, Any], response: Mapping[str, Any]) -> JsonDict:
    """Reuse the repaired parser and add the V653 development disposition."""

    row = _SHARED_BUILD_CAPTURE_ROW(schedule, response)
    if row.get("capture_phase") == "development":
        disposition = _development_disposition(row)
        row.update(
            {
                "development_disposition": disposition,
                "development_usable": disposition == "usable_nonempty",
            }
        )
    return row


def reduce_development_gate(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Open evaluation only from nonempty usable outputs while reporting all cases."""

    classified = [
        build_capture_row(row, row) if "development_disposition" not in row else deepcopy(dict(row))
        for row in rows
    ]
    reduced = _SHARED_REDUCE_DEVELOPMENT_GATE(classified)
    reduced["usable_by_arm"] = {
        arm: sum(
            row.get("development_disposition") == "usable_nonempty"
            for row in classified
            if row.get("arm") == arm
        )
        for arm in shared.protocol.ARMS
    }
    reduced["capture_open"] = bool(
        len(classified) == shared.DEVELOPMENT_CALLS
        and all(
            reduced["usable_by_arm"][arm] >= shared.DEVELOPMENT_USABLE_MINIMUM
            for arm in shared.protocol.ARMS
        )
    )
    reduced["terminal_class"] = "ready" if reduced["capture_open"] else "null"
    reduced["disposition_counts"] = dict(
        sorted(Counter(str(row.get("development_disposition")) for row in classified).items())
    )
    reduced["correct_empty_counts_as_extraction_coverage"] = False
    return reduced


def reportable_constructed_delta(*, capture_open: bool, measured: float | None) -> float | None:
    """Expose a constructed-pair delta only after development opens evaluation."""

    return measured if capture_open else None


def capture_readiness(
    *,
    development_rows: Sequence[Mapping[str, Any]],
    evaluation_rows: Sequence[Mapping[str, Any]],
    require_terminal: bool,
    affected_ok: bool,
    receipt_errors: Sequence[str],
    runner: Mapping[str, Any],
    flagged_adversarial: bool,
    producer_runtime_error: str | None,
) -> int:
    """Separate accountable completion from whether the scientific gate opened."""

    development_accounted = len(development_rows) == shared.DEVELOPMENT_CALLS and all(
        row.get("development_disposition")
        in {
            "usable_nonempty",
            "correct_empty",
            "malformed",
            "truncated",
            "missing_output",
            "transport_failure",
            "unstarted",
        }
        for row in development_rows
    )
    gate = reduce_development_gate(development_rows) if development_accounted else {}
    evaluation_accounted = len(evaluation_rows) == shared.EVALUATION_CALLS and all(
        row.get("disposition") in {"completed", "malformed", "truncated", "failed", "cancelled"}
        for row in evaluation_rows
    )
    if gate.get("capture_open") is not True:
        evaluation_accounted = len(evaluation_rows) == shared.EVALUATION_CALLS and all(
            row.get("disposition") == "unstarted" for row in evaluation_rows
        )
    clean = bool(
        require_terminal
        and affected_ok
        and not receipt_errors
        and runner.get("all_layers_offloaded") is True
        and runner.get("lease_released") is True
        and not flagged_adversarial
        and producer_runtime_error is None
    )
    return int(development_accounted and evaluation_accounted and clean)


def base_artifact() -> JsonDict:
    """Add V653 fields to the shared complete artifact shape."""

    value = _SHARED_BASE_ARTIFACT()
    value.update({"phase": PHASE, "semantic_scope": deepcopy(SEMANTIC_SCOPE)})
    return value


def _with_lifecycle_checks(
    checks: Sequence[Mapping[str, Any]], context: Mapping[str, Any]
) -> tuple[list[JsonDict], JsonDict]:
    """Attach exact lifecycle gates and byte identity to measured candidates."""

    lifecycle = load_object(REPO_ROOT / LIFECYCLE_PATH)
    rows = [deepcopy(dict(row)) for row in checks]
    present = {str(row.get("check")) for row in rows}
    rows.extend(row for row in lifecycle_gate_rows(lifecycle) if row["check"] not in present)
    copied = deepcopy(dict(context))
    source_hashes = deepcopy(dict(copied.get("source_hashes") or {}))
    path = REPO_ROOT / LIFECYCLE_PATH
    if path.is_file():
        source_hashes[LIFECYCLE_PATH.as_posix()] = {
            "path": LIFECYCLE_PATH.as_posix(),
            "sha256": sha256_file(path),
            "original_verdict_class": lifecycle.get("verdict_class"),
            "original_flagged_adversarial": lifecycle.get("flagged_adversarial"),
        }
    copied["source_hashes"] = source_hashes
    return rows, copied


def measured_artifact(**kwargs: Any) -> JsonDict:
    """Compose a V653 candidate and recompute completion from accountable rows."""

    checks, context = _with_lifecycle_checks(kwargs["checks"], kwargs["context"])
    supplied = dict(kwargs)
    supplied.update({"checks": checks, "context": context})
    value = _SHARED_MEASURED_ARTIFACT(**supplied)
    capture = dict(kwargs["capture"])
    reduced = dict(kwargs["reduced"])
    receipts = list(kwargs["receipts"])
    receipt_errors = shared.required_receipt_errors(receipts) if kwargs["require_terminal"] else []
    readiness = capture_readiness(
        development_rows=capture.get("development_rows") or [],
        evaluation_rows=reduced.get("extraction_rows") or [],
        require_terminal=bool(kwargs["require_terminal"]),
        affected_ok=bool(kwargs["affected_ok"]),
        receipt_errors=receipt_errors,
        runner=kwargs["runner"],
        flagged_adversarial=bool(kwargs["flagged_adversarial"]),
        producer_runtime_error=kwargs.get("producer_runtime_error"),
    )
    for gate in value["acceptance_gate_results"]:
        if gate.get("check") == "all_evaluation_dispositions":
            gate.update(
                {
                    "observed": len(reduced.get("extraction_rows") or []),
                    "passed": len(reduced.get("extraction_rows") or []) == shared.EVALUATION_CALLS,
                    "principle": "Every evaluation call must have a terminal or explicit unstarted disposition.",
                }
            )
        elif gate.get("check") == "development_gate_open":
            gate["category"] = "benefit"
    value.update(
        {
            "phase": PHASE,
            "span_capture_complete_score": readiness,
            "constructed_qualifier_delta": reportable_constructed_delta(
                capture_open=dict(capture.get("development_gate") or {}).get("capture_open")
                is True,
                measured=reduced.get("constructed_qualifier_delta"),
            ),
            "semantic_scope": deepcopy(SEMANTIC_SCOPE),
            "gate_check_summary": shared._gate_summary(value["acceptance_gate_results"]),
        }
    )
    value["reproducibility_checksum"] = artifact_checksum(value)
    return value


def _collect_preconditions(root: Path) -> tuple[list[JsonDict], JsonDict]:  # pragma: no cover
    """Add the driving requirement and exact lifecycle artifact to shared checks."""

    checks, context = _SHARED_COLLECT_PRECONDITIONS(root)
    spec = (root / SPEC_PATH).read_text(encoding="utf-8") if (root / SPEC_PATH).is_file() else ""
    checks.append(
        shared._gate(
            "driving_requirement_v653",
            "precondition",
            "==",
            "REQ-VERIFY-7451",
            "REQ-VERIFY-7451" if "REQ-VERIFY-7451" in spec else None,
            "REQ-VERIFY-7451" in spec,
            "The V653 requirement must exist before model work.",
            upstream=SPEC_PATH.as_posix(),
            path=SPEC_PATH.as_posix(),
            field="REQ-*",
        )
    )
    lifecycle_path = root / LIFECYCLE_PATH
    lifecycle = load_object(lifecycle_path)
    observed_hash = sha256_file(lifecycle_path) if lifecycle_path.is_file() else None
    checks.append(
        shared._gate(
            "capture_lifecycle_artifact_hash",
            "precondition",
            "==",
            EXPECTED_LIFECYCLE_SHA256,
            observed_hash,
            observed_hash == EXPECTED_LIFECYCLE_SHA256,
            "A changed lifecycle producer cannot silently authorize model work.",
            upstream=LIFECYCLE_PATH.as_posix(),
            path=LIFECYCLE_PATH.as_posix(),
            field="sha256",
        )
    )
    checks.extend(lifecycle_gate_rows(lifecycle))
    source_hashes = deepcopy(dict(context.get("source_hashes") or {}))
    for relative in (LIFECYCLE_PATH, LIFECYCLE_MODULE_PATH):
        path = root / relative
        if path.is_file():
            source = load_object(path) if relative.suffix == ".json" else {}
            source_hashes[relative.as_posix()] = {
                "path": relative.as_posix(),
                "sha256": sha256_file(path),
                **(
                    {
                        "original_verdict_class": source.get("verdict_class"),
                        "original_flagged_adversarial": source.get("flagged_adversarial"),
                    }
                    if source
                    else {}
                ),
            }
    context["source_hashes"] = source_hashes
    return checks, context


def collect_preconditions(root: Path = REPO_ROOT) -> tuple[list[JsonDict], JsonDict]:
    """Authenticate V653 inputs under the isolated shared-engine contract."""

    with capture_contract():
        return _collect_preconditions(root)


def terminal_commands(root: Path, candidate: Path) -> list[CommandSpec]:
    """Build fresh V653 replay, independent reduction, and strict readers."""

    python = str(root / ".venv/bin/python")
    reducer = (
        "import json,pathlib,sys;"
        "from carnot.experiment_7451_v653_span_capture import independent_reduce_artifact;"
        "p=pathlib.Path(sys.argv[1]);v=json.loads(p.read_text());"
        "e=independent_reduce_artifact(v,root=pathlib.Path.cwd(),require_terminal=False);"
        "print(json.dumps({'errors':e},sort_keys=True),flush=True);raise SystemExit(bool(e))"
    )
    return [
        CommandSpec(
            "declared_entrypoint_cold_replay",
            (
                python,
                "-u",
                WRAPPER_PATH.as_posix(),
                "--date",
                RUN_DATE,
                "--validate",
                str(candidate),
            ),
            "measured_candidate",
        ),
        CommandSpec(
            "independent_cold_reducer",
            (python, "-u", "-c", reducer, str(candidate)),
            "measured_candidate",
        ),
        CommandSpec(
            "adversarial_verify",
            (python, "-u", "scripts/adversarial_verify.py", str(candidate)),
            "measured_candidate",
        ),
        CommandSpec(
            "verdict_row_consistency_strict",
            (
                python,
                "-u",
                "scripts/verdict_row_consistency_lint.py",
                "--strict",
                str(candidate),
            ),
            "measured_candidate",
        ),
    ]


@contextmanager
def capture_contract() -> Iterator[None]:
    """Apply V653 identity to the shared engine and restore every changed global."""

    values = {
        "MILESTONE": MILESTONE,
        "EXPERIMENT_ID": EXPERIMENT_ID,
        "TASK_ID": TASK_ID,
        "SCHEMA": SCHEMA,
        "RESULT_PATH": RESULT_PATH,
        "RAW_DIR": RAW_DIR,
        "MODULE_PATH": MODULE_PATH,
        "WRAPPER_PATH": WRAPPER_PATH,
        "TEST_PATH": TEST_PATH,
        "SPEC_PATH": SPEC_PATH,
        "MODEL_SPECS": MODEL_SPECS,
        "INFERENCE_SUBSTRATE": INFERENCE_SUBSTRATE,
        "INFERENCE_SUBSTRATE_CLASS": INFERENCE_SUBSTRATE_CLASS,
        "EXECUTION_VENUE": EXECUTION_VENUE,
        "RANDOM_SEED": RANDOM_SEED,
        "VALIDATION_MANIFEST": VALIDATION_MANIFEST,
        "FIELD_PRINCIPLES": FIELD_PRINCIPLES,
        "REQUIRED_FIELDS": REQUIRED_FIELDS,
        "progress": progress,
        "build_capture_row": build_capture_row,
        "reduce_development_gate": reduce_development_gate,
        "_base_artifact": base_artifact,
        "_measured_artifact": measured_artifact,
        "collect_preconditions": _collect_preconditions,
        "_terminal_commands": terminal_commands,
        "validate_artifact": validate_artifact,
        "independent_reduce_artifact": independent_reduce_artifact,
    }
    previous = {name: getattr(shared, name) for name in values}
    try:
        for name, value in values.items():
            setattr(shared, name, value)
        yield
    finally:
        for name, value in previous.items():
            setattr(shared, name, value)


def build_fixture_artifact() -> JsonDict:
    """Build a deterministic V653 artifact without loading a model."""

    with capture_contract():
        return _SHARED_BUILD_FIXTURE_ARTIFACT()


def build_blocked_artifact(checks: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Build a V653 external block with no simulated current execution."""

    with capture_contract():
        return _SHARED_BUILD_BLOCKED_ARTIFACT(checks)


def _shared_expected_completion(value: Mapping[str, Any], require_terminal: bool) -> int:
    """Compute the legacy engine score used only to reuse its other validators."""

    receipts = value.get("validation_receipts") or []
    receipt_errors = shared.required_receipt_errors(receipts) if require_terminal else []
    rows = value.get("extraction_rows") or []
    runner = value.get("runner_receipt") or {}
    return int(
        require_terminal
        and not receipt_errors
        and len(rows) == shared.EVALUATION_CALLS
        and all(isinstance(row, Mapping) and row.get("disposition") != "unstarted" for row in rows)
        and runner.get("all_layers_offloaded") is True
        and runner.get("lease_released") is True
        and value.get("flagged_adversarial") is False
    )


def validate_artifact(
    value: object, *, require_terminal: bool, root: Path = REPO_ROOT
) -> list[str]:
    """Cold-check the shared evidence plus V653 gates, dispositions, and readiness."""

    if not isinstance(value, Mapping):
        return ["artifact_not_object"]
    errors: list[str] = []
    blocked = value.get("verdict_class") == "blocked"
    gates = value.get("preconditions_checked") or []
    gate_names = {str(row.get("check")) for row in gates if isinstance(row, Mapping)}
    for name in (
        "capture_lifecycle_identity",
        "capture_lifecycle_ready",
        "capture_lifecycle_verdict",
        "capture_lifecycle_unflagged",
    ):
        if name not in gate_names:
            errors.append(f"lifecycle_gate_missing:{name}")
    if value.get("phase") != PHASE:
        errors.append("declaration_mismatch:phase")
    if value.get("semantic_scope") != SEMANTIC_SCOPE:
        errors.append("semantic_scope_mismatch")
    development = value.get("development_rows") or []
    development_gate = reduce_development_gate(development) if development else {}
    if not blocked and (
        len(development) != shared.DEVELOPMENT_CALLS
        or any(not row.get("development_disposition") for row in development)
    ):
        errors.append("development_dispositions_mismatch")
    if (
        not blocked
        and development_gate.get("capture_open") is not True
        and value.get("constructed_qualifier_delta") is not None
    ):
        errors.append("constructed_qualifier_delta_without_open_capture")
    receipts = value.get("validation_receipts") or []
    receipt_errors = shared.required_receipt_errors(receipts) if require_terminal else []
    expected = (
        0
        if blocked
        else capture_readiness(
            development_rows=development,
            evaluation_rows=value.get("extraction_rows") or [],
            require_terminal=require_terminal,
            affected_ok=not any(
                error.startswith("required_validation_receipt_failed:")
                and error.split(":", 1)[1] in AFFECTED_CHECK_NAMES
                for error in receipt_errors
            ),
            receipt_errors=receipt_errors,
            runner=value.get("runner_receipt") or {},
            flagged_adversarial=value.get("flagged_adversarial") is True,
            producer_runtime_error=value.get("producer_runtime_error"),
        )
    )
    if value.get("span_capture_complete_score") != expected:
        errors.append("span_capture_complete_score_mismatch")
    normalized = deepcopy(dict(value))
    normalized["span_capture_complete_score"] = _shared_expected_completion(
        normalized, require_terminal
    )
    if not blocked and development_gate.get("capture_open") is not True:
        normalized["constructed_qualifier_delta"] = 0.0
    normalized["reproducibility_checksum"] = artifact_checksum(normalized)
    with capture_contract():
        errors.extend(
            _SHARED_VALIDATE_ARTIFACT(
                normalized,
                require_terminal=require_terminal,
                root=root,
            )
        )
    if value.get("reproducibility_checksum") != artifact_checksum(value):
        errors.append("reproducibility_checksum_mismatch")
    return list(dict.fromkeys(errors))


def independent_reduce_artifact(
    value: object, *, root: Path = REPO_ROOT, require_terminal: bool = True
) -> list[str]:
    """Recompute the V653 artifact without trusting its summary scores."""

    return validate_artifact(value, root=root, require_terminal=require_terminal)


def date_argument(value: str) -> str:
    """Accept only the sealed V653 execution date."""

    if value != RUN_DATE:
        raise ValueError(f"date must be {RUN_DATE}")
    return value


def run_experiment(
    *, root: Path = REPO_ROOT, run_date: str = RUN_DATE, output_path: Path | None = None
) -> JsonDict:  # pragma: no cover - bounded live capability E2E.
    """Run the shared live engine once under the isolated V653 contract."""

    with capture_contract():
        return _SHARED_RUN_EXPERIMENT(root=root, run_date=run_date, output_path=output_path)


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - thin CLI boundary.
    """Run the repaired capture or cold-replay one unpublished candidate."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", type=date_argument, default=RUN_DATE)
    parser.add_argument("--validate", type=Path)
    args = parser.parse_args(argv)
    started = time.monotonic()
    progress(started, "entrypoint", "start", date=args.date)
    if args.validate is not None:
        value = load_object(args.validate)
        names = {
            row.get("name")
            for row in value.get("validation_receipts") or []
            if isinstance(row, Mapping)
        }
        errors = independent_reduce_artifact(
            value,
            root=REPO_ROOT,
            require_terminal=set(TERMINAL_CHECK_NAMES).issubset(names),
        )
        print(json.dumps({"errors": errors}, sort_keys=True), flush=True)
        return int(bool(errors))
    result = run_experiment(root=REPO_ROOT, run_date=args.date)
    print(
        json.dumps(
            {
                "artifact": str(REPO_ROOT / RESULT_PATH),
                "honest_verdict": result["honest_verdict"],
                "span_capture_complete_score": result["span_capture_complete_score"],
                "span_value_score": result["span_value_score"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
