"""Exp 5327 SMT hint validation corrigendum re-emit.

Spec refs: REQ-VERIFY-5327, SCENARIO-VERIFY-5327.

This module repairs the artifact surface from Exp 5318 without changing the
underlying scientific claim. The prior run was a fast deterministic SMT solver
protocol, but its artifact carried future runtime marker text and was therefore
audited as though it had claimed compute-bound execution. Exp 5327 re-runs the
same local solver fixture, records the real wall-clock duration, and emits a
clean solver/protocol substrate with no live-inference marker strings.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
from pathlib import Path
import time
from typing import Any

from carnot import experiment_5318_smt_hint_validation_protocol_v485 as exp5318


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_DATE = "20260707"
RANDOM_SEED = 5327
EXPERIMENT_ID = "exp5327-smt-hint-corrigendum-reemit-v486"
MILESTONE = "2026.07.486"
SCHEMA = "carnot.experiment_5327.smt_hint_corrigendum_reemit.v486"
RESULT_RELATIVE_PATH = Path(
    "results/experiment_5327_smt_hint_corrigendum_reemit_v486.json"
)
FIXTURE_RELATIVE_PATH = Path(
    "python/carnot/experiment_5318_smt_hint_validation_protocol_v485.py"
)
SOURCE_EXP5318_RELATIVE_PATH = Path(
    "results/experiment_5318_smt_hint_validation_protocol_v485.json"
)
INFERENCE_SUBSTRATE = "deterministic_smt_solver_protocol"
SPEC_REFS = ("REQ-VERIFY-5327", "SCENARIO-VERIFY-5327")
TERMINAL_PREFIXES = ("complete:", "blocked_")
MIN_TRUSTED_METHODOLOGY_DURATION_S = 0.0001
COMPUTE_BOUND_MARKERS = (
    "unsloth/",
    "Qwen3.6-",
    "Qwen3.5-",
    "Qwen1.5-",
    "gemma-4-",
    "GGUF",
    "DualGPURunner",
    "DualGPUHarness",
    "llama.cpp",
    "torch",
    "torch.cuda",
    ".cuda(",
)

FIELD_PRINCIPLES: dict[str, str] = {
    "experiment_id": "Traceability for the Exp5327 SMT hint validation corrigendum re-emit.",
    "milestone": "Milestone accountability for the V486 SMT hint methodology repair.",
    "status": (
        "Machine-readable terminal state; complete means the deterministic solver "
        "protocol is clean, blocked means timing or marker hygiene failed."
    ),
    "honest_verdict": (
        "Terminal verdict must start with complete: or blocked_ and state whether "
        "the Exp5318 methodology flag was repaired without broadening the claim."
    ),
    "inference_substrate": (
        "Declares deterministic_smt_solver_protocol so the artifact is read as "
        "solver/protocol validation rather than runtime execution."
    ),
    "exp5318_flag_reason": (
        "Explains why Exp5318 was quarantined so downstream work does not inherit "
        "the substrate/timing confusion."
    ),
    "valid_hint_acceptance_rate": (
        "Bare numeric fraction of solver-validated sound hints accepted by the "
        "deterministic protocol."
    ),
    "unsound_hint_rejection_rate": (
        "Bare numeric fraction of solver-refuted unsound hints rejected before "
        "they can influence final solving."
    ),
    "usefulness_rate": (
        "Bare numeric fraction of accepted valid hints that reduce proof burden, "
        "keeping useful hints distinct from merely safe hints."
    ),
    "solver_fallback_complete": (
        "Bare boolean proving rejected hints fall back to the classical SMT solver "
        "and preserve the baseline result."
    ),
    "methodology_duration_s": (
        "Bare numeric wall-clock duration for the deterministic protocol run; "
        "fast is acceptable only when honestly measured."
    ),
    "compute_bound_marker_present": (
        "Bare boolean showing whether runtime-marker text remains in the artifact "
        "and could trigger another false compute-bound reading."
    ),
    "smt_hint_protocol_clean": (
        "Bare boolean true only when the solver metrics, fallback behavior, "
        "substrate declaration, marker scan, and timing floor all pass."
    ),
    "tests_run": (
        "Commands run to validate the corrigendum fixture, artifact schema, "
        "new-code coverage, and repository tests."
    ),
}

REQUIRED_ARTIFACT_FIELDS = (
    "experiment_id",
    "milestone",
    "status",
    "honest_verdict",
    "inference_substrate",
    "exp5318_flag_reason",
    "valid_hint_acceptance_rate",
    "unsound_hint_rejection_rate",
    "usefulness_rate",
    "solver_fallback_complete",
    "methodology_duration_s",
    "compute_bound_marker_present",
    "smt_hint_protocol_clean",
    "tests_run",
)
WRAPPED_FIELDS = (
    "experiment_id",
    "milestone",
    "status",
    "honest_verdict",
    "inference_substrate",
    "exp5318_flag_reason",
    "tests_run",
)
BARE_BOOL_FIELDS = (
    "solver_fallback_complete",
    "compute_bound_marker_present",
    "smt_hint_protocol_clean",
)
BARE_NUMERIC_FIELDS = (
    "valid_hint_acceptance_rate",
    "unsound_hint_rejection_rate",
    "usefulness_rate",
    "methodology_duration_s",
)


def wrap_field(field: str, value: Any) -> JsonDict:
    """Attach the task-required principle so audits can read both value and why."""

    return {"principle": FIELD_PRINCIPLES[field], "value": value}


def compute_bound_marker_present(payload: Mapping[str, Any]) -> bool:
    """Return whether the serialized artifact still contains runtime marker text."""

    serialized = json.dumps(payload, sort_keys=True)
    return any(marker in serialized for marker in COMPUTE_BOUND_MARKERS)


def run_corrigendum_protocol() -> JsonDict:
    """Re-run the deterministic Exp 5318 SMT hint fixture without any proposer call."""

    benchmark = exp5318.run_benchmark()
    return {
        "fixture_examples": benchmark["fixture_examples"],
        "hint_validation_telemetry": benchmark["hint_validation_telemetry"],
        "valid_hint_acceptance_rate": benchmark["valid_hint_acceptance_rate"],
        "unsound_hint_rejection_rate": benchmark["unsound_hint_rejection_rate"],
        "usefulness_rate": benchmark["usefulness_rate"],
        "solver_fallback_complete": benchmark["solver_fallback_complete"],
        "completeness_preserved": benchmark["completeness_preserved"],
        "counts": benchmark["counts"],
        "llm_invoked": False,
    }


def exp5318_flag_reason(
    source_path: Path = REPO_ROOT / SOURCE_EXP5318_RELATIVE_PATH,
) -> str:
    """Summarize the prior audit without copying the raw marker-bearing detail."""

    source = json.loads(source_path.read_text(encoding="utf-8"))
    flag_kinds = ",".join(
        str(flag.get("kind", "unknown")) for flag in source.get("corrigendum_pending", [])
    )
    return (
        f"Exp5318 was quarantined because it reported {source.get('duration_s')}s "
        "actual deterministic solver duration while also carrying future runtime "
        f"marker text; audit kinds={flag_kinds}. Exp5327 removes those marker "
        "strings and re-emits only the solver protocol."
    )


def build_artifact(
    *,
    methodology_duration_s: float | None = None,
    tests_run: Sequence[Mapping[str, Any]] | None = None,
) -> JsonDict:
    """Build the Exp 5327 corrigendum artifact from the deterministic fixture."""

    started_at = time.perf_counter()
    benchmark = run_corrigendum_protocol()
    measured_duration = (
        round(time.perf_counter() - started_at, 6)
        if methodology_duration_s is None
        else float(methodology_duration_s)
    )
    duration_trusted = measured_duration >= MIN_TRUSTED_METHODOLOGY_DURATION_S
    metrics_clean = _protocol_metrics_clean(benchmark)
    source_audit = _source_exp5318_audit()
    artifact: JsonDict = {
        "schema": SCHEMA,
        "run_date": RUN_DATE,
        "random_seed": RANDOM_SEED,
        "experiment_id": wrap_field("experiment_id", EXPERIMENT_ID),
        "milestone": wrap_field("milestone", MILESTONE),
        "status": wrap_field("status", "blocked"),
        "honest_verdict": wrap_field("honest_verdict", "blocked_initializing"),
        "inference_substrate": wrap_field("inference_substrate", INFERENCE_SUBSTRATE),
        "exp5318_flag_reason": wrap_field("exp5318_flag_reason", exp5318_flag_reason()),
        "valid_hint_acceptance_rate": benchmark["valid_hint_acceptance_rate"],
        "unsound_hint_rejection_rate": benchmark["unsound_hint_rejection_rate"],
        "usefulness_rate": benchmark["usefulness_rate"],
        "solver_fallback_complete": benchmark["solver_fallback_complete"],
        "methodology_duration_s": measured_duration,
        "compute_bound_marker_present": False,
        "smt_hint_protocol_clean": False,
        "tests_run": wrap_field("tests_run", [dict(row) for row in tests_run or []]),
        "methodology_duration_floor_s": MIN_TRUSTED_METHODOLOGY_DURATION_S,
        "fixture_path": str(FIXTURE_RELATIVE_PATH),
        "source_exp5318_audit": source_audit,
        "fixture_examples": benchmark["fixture_examples"],
        "hint_validation_telemetry": benchmark["hint_validation_telemetry"],
        "counts": benchmark["counts"],
        "llm_invoked": benchmark["llm_invoked"],
        "completeness_preserved": benchmark["completeness_preserved"],
        "claim_limits": [
            "deterministic SMT solver/protocol validation only",
            "canned hints only; no LLM proposer invoked",
            "solver entailment validates hints before acceptance",
            "unsound hints fall back to the classical SMT result",
            "corrigendum repairs artifact substrate and timing fields only",
        ],
        "field_principles": FIELD_PRINCIPLES,
        "spec_refs": list(SPEC_REFS),
    }
    marker_present = compute_bound_marker_present(artifact)
    clean = metrics_clean and duration_trusted and not marker_present
    artifact["compute_bound_marker_present"] = marker_present
    artifact["smt_hint_protocol_clean"] = clean
    artifact["status"] = wrap_field("status", "complete" if clean else "blocked")
    artifact["honest_verdict"] = wrap_field(
        "honest_verdict",
        _honest_verdict(clean=clean, duration_trusted=duration_trusted, marker_present=marker_present),
    )
    artifact["reproducibility_checksum"] = _checksum_payload(benchmark)
    validate_artifact(artifact)
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Fail closed when the Exp 5327 artifact drifts from its clean contract."""

    for field in REQUIRED_ARTIFACT_FIELDS:
        _require(field in artifact, f"missing required field: {field}")
    for field in WRAPPED_FIELDS:
        wrapped = artifact[field]
        _require(isinstance(wrapped, Mapping), f"{field} must be principle-wrapped")
        _require("value" in wrapped, f"{field} missing value")
        _require(
            wrapped.get("principle") == FIELD_PRINCIPLES[field],
            f"{field} principle drift",
        )
    for field in BARE_BOOL_FIELDS:
        _require(isinstance(artifact[field], bool), f"{field} must be a bare bool")
    for field in BARE_NUMERIC_FIELDS:
        _require(
            isinstance(artifact[field], int | float) and not isinstance(artifact[field], bool),
            f"{field} must be a bare numeric value",
        )
    for field in (
        "valid_hint_acceptance_rate",
        "unsound_hint_rejection_rate",
        "usefulness_rate",
    ):
        _require(0.0 <= float(artifact[field]) <= 1.0, f"{field} rate out of range")

    verdict = artifact["honest_verdict"]["value"]
    status = artifact["status"]["value"]
    _require(
        isinstance(verdict, str) and verdict.startswith(TERMINAL_PREFIXES),
        "honest_verdict prefix",
    )
    _require(artifact["experiment_id"]["value"] == EXPERIMENT_ID, "experiment_id drift")
    _require(artifact["milestone"]["value"] == MILESTONE, "milestone drift")
    _require(
        artifact["inference_substrate"]["value"] == INFERENCE_SUBSTRATE,
        f"inference_substrate must be {INFERENCE_SUBSTRATE}",
    )
    _require("0.029515s" in artifact["exp5318_flag_reason"]["value"], "flag reason drift")
    _require(artifact["valid_hint_acceptance_rate"] == 1.0, "valid hint acceptance must be complete")
    _require(
        artifact["unsound_hint_rejection_rate"] == 1.0,
        "unsound hint rejection must be complete",
    )
    _require(artifact["solver_fallback_complete"] is True, "solver fallback must be complete")
    _require(float(artifact["methodology_duration_s"]) >= 0.0, "duration must be non-negative")
    _require(artifact["compute_bound_marker_present"] is False, "compute-bound marker must be false")
    _require(not compute_bound_marker_present(artifact), "marker scan found runtime marker")
    _require(isinstance(artifact["tests_run"]["value"], list), "tests_run must be list")
    _require("REQ-VERIFY-5327" in artifact["spec_refs"], "spec refs must include REQ-VERIFY-5327")
    _require(len(str(artifact["reproducibility_checksum"])) == 64, "checksum drift")

    if artifact["smt_hint_protocol_clean"]:
        _require(status == "complete", "clean artifact status must be complete")
        _require(str(verdict).startswith("complete:"), "clean artifact verdict must be complete")
        _require(
            float(artifact["methodology_duration_s"]) >= MIN_TRUSTED_METHODOLOGY_DURATION_S,
            "clean artifact duration below floor",
        )
    else:
        _require(status == "blocked", "unclean artifact status must be blocked")
        _require(str(verdict).startswith("blocked_"), "unclean artifact verdict must be blocked")


def write_outputs(
    *,
    artifact_path: str | Path = RESULT_RELATIVE_PATH,
    methodology_duration_s: float | None = None,
    tests_run: Sequence[Mapping[str, Any]] | None = None,
) -> JsonDict:
    """Write the Exp 5327 JSON artifact and return the validated payload."""

    artifact = build_artifact(
        methodology_duration_s=methodology_duration_s,
        tests_run=tests_run,
    )
    output_path = Path(artifact_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(artifact, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return artifact


def _source_exp5318_audit(
    source_path: Path = REPO_ROOT / SOURCE_EXP5318_RELATIVE_PATH,
) -> JsonDict:
    source = json.loads(source_path.read_text(encoding="utf-8"))
    return {
        "source_artifact": str(SOURCE_EXP5318_RELATIVE_PATH),
        "source_duration_s": source.get("duration_s"),
        "source_flagged_adversarial": bool(source.get("flagged_adversarial")),
        "source_flag_kinds": [
            str(flag.get("kind", "unknown"))
            for flag in source.get("corrigendum_pending", [])
        ],
        "conductor_status": "FLAGGED",
    }


def _protocol_metrics_clean(benchmark: Mapping[str, Any]) -> bool:
    return bool(
        benchmark["valid_hint_acceptance_rate"] == 1.0
        and benchmark["unsound_hint_rejection_rate"] == 1.0
        and benchmark["solver_fallback_complete"] is True
        and benchmark["completeness_preserved"] is True
        and benchmark["llm_invoked"] is False
    )


def _honest_verdict(
    *,
    clean: bool,
    duration_trusted: bool,
    marker_present: bool,
) -> str:
    if clean:
        return (
            "complete: Exp5318 SMT hint protocol re-emitted with clean deterministic "
            "solver substrate, actual methodology timing, and no runtime marker text"
        )
    if marker_present:
        return "blocked_runtime_marker_present: corrigendum artifact still contains runtime marker text"
    if not duration_trusted:
        return "blocked_methodology_duration_untrusted: deterministic run duration below audit floor"
    return "blocked_smt_hint_protocol_not_clean"


def _checksum_payload(benchmark: Mapping[str, Any]) -> str:
    payload = {
        "experiment_id": EXPERIMENT_ID,
        "spec_refs": SPEC_REFS,
        "source": str(SOURCE_EXP5318_RELATIVE_PATH),
        "fixture_examples": benchmark["fixture_examples"],
        "hint_validation_telemetry": [
            {
                "example_id": row["example_id"],
                "hint_id": row["hint_id"],
                "solver_valid": row["solver_valid"],
                "accepted": row["accepted"],
                "usefulness_class": row["usefulness_class"],
                "baseline_status": row["baseline_status"],
                "blindly_added_status": row["blindly_added_status"],
                "final_status": row["final_status"],
                "fallback_to_classical": row["fallback_to_classical"],
                "overwrite_clauses": row["overwrite_clauses"],
            }
            for row in benchmark["hint_validation_telemetry"]
        ],
        "rates": {
            "valid": benchmark["valid_hint_acceptance_rate"],
            "unsound": benchmark["unsound_hint_rejection_rate"],
            "usefulness": benchmark["usefulness_rate"],
        },
        "fallback": benchmark["solver_fallback_complete"],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> int:  # pragma: no cover - thin CLI wrapper.
    artifact = write_outputs()
    print(artifact["honest_verdict"]["value"])
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
