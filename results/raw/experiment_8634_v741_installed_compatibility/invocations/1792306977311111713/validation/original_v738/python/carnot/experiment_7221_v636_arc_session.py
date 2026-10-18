"""Run the V636 cumulative ARC session through the repaired V635 runtime.

The shared runtime owns model loading, GPU leasing, heartbeats, the scored E3
policy, and receipt reduction. This module supplies the current task identity
and authenticates retained V635 induction IDs without relabeling old rows.

Spec refs: REQ-ARC-WMTE-7221 and SCENARIO-ARC-WMTE-7221-*.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from carnot import experiment_7206_v635_arc_volume_a as base


JsonDict = dict[str, Any]
REPO_ROOT = Path(__file__).resolve().parents[2]

TASK_ID = "exp7221-arc-session"
EXPERIMENT_ID = 7221
MILESTONE = "2026.09.636"
RUN_DATE = "20260911"
GAME = "r11l"
RANDOM_SEED = 7_221_001
ACTION_BUDGET = 4000
SESSION_TIMEOUT_S = 3600
INDUCTION_TIMEOUT_S = 2400
HARD_CAP_S = 4800
N_CTX = 49152
COMPLETION_BUDGET = 4096
EVIDENCE_TARGET = 10
MODEL_ID = "unsloth/Qwen3.8-27B-GGUF"
QUANTIZATION = "Q4_K_M"
MODEL_SPECS = [{"hf_id": MODEL_ID, "quantization": QUANTIZATION}]
EXPECTED_PRIOR_VERDICT = "complete_null_no_observed_missing_tool_demand_cumulative_n_7"

SCHEMA = "carnot.experiment_7221.v636_arc_session.v1"
DRIVING_REQUIREMENT = "REQ-ARC-WMTE-7221"
MODULE_PATH = Path("python/carnot/experiment_7221_v636_arc_session.py")
WRAPPER_PATH = Path("scripts/experiments/experiment_7221_v636_arc_session.py")
TEST_PATH = Path("tests/python/test_experiment_7221_v636_arc_session.py")
RESULT_PATH = Path("results/experiment_7221_v636_arc_session.json")
CHECKPOINT_PATH = Path("results/checkpoints/experiment_7221_v636_arc_session/running.json")
CHECKPOINT_SCHEMA = "carnot.experiment_7221.checkpoint.v1"
RAW_DIR = Path("results/raw/experiment_7221")
V635_ARTIFACT_PATHS = (
    Path("results/experiment_7206_v635_arc_volume_a.json"),
    Path("results/experiment_7207_v635_arc_volume_b.json"),
)
PRIOR_ARTIFACT_PATH = V635_ARTIFACT_PATHS[-1]
SIBLING_PATH = V635_ARTIFACT_PATHS[-1]
SIBLING_TASK_ID = "authenticated-v635-cumulative"
BASE_MODULE_PATH = Path("python/carnot/experiment_7206_v635_arc_volume_a.py")
V635_B_MODULE_PATH = Path("python/carnot/experiment_7207_v635_arc_volume_b.py")
V635_CAPSTONE_PATH = Path("results/experiment_7218_v635_capstone.json")

EXPECTED_TASK_CONTRACT = {
    "id": TASK_ID,
    "milestone": MILESTONE,
    "deliverable": RESULT_PATH.as_posix(),
    "gated_on": None,
    "prior_failures": [
        {
            "experiment_id": "exp7207-arc-volume-b",
            "verdict": EXPECTED_PRIOR_VERDICT,
            "addressed_by": (
                "New separately capped session explicitly authorized on 2026-09-11; shipped "
                "reducer now supplies the current seed/session identity, preserving historical "
                "artifacts."
            ),
            "retire_if_same_verdict": True,
        }
    ],
    "operator_override": (
        "2026-09-11 ops/known-issues.md cumulative selfparse directive explicitly requests "
        "additional separately capped sessions toward ten unique inductions."
    ),
}

_REPLACED_SOURCES = {
    base.MODULE_PATH,
    base.WRAPPER_PATH,
    base.TEST_PATH,
    base.PRIOR_ARTIFACT_PATH,
}
REQUIRED_SOURCE_PATHS = tuple(
    path for path in base.REQUIRED_SOURCE_PATHS if path not in _REPLACED_SOURCES
) + (
    BASE_MODULE_PATH,
    V635_B_MODULE_PATH,
    V635_ARTIFACT_PATHS[0],
    V635_ARTIFACT_PATHS[1],
    V635_CAPSTONE_PATH,
    MODULE_PATH,
    WRAPPER_PATH,
    TEST_PATH,
)

FIELD_PRINCIPLES = deepcopy(base.FIELD_PRINCIPLES)
FIELD_PRINCIPLES.update(
    {
        "field_principles": (
            "Annotate actual values in this map; do not wrap arbitrary dictionaries as "
            "principle/value records."
        ),
        "status": (
            "Write a terminal artifact only when done or externally blocked; running "
            "checkpoints use a different path."
        ),
        "run_date": "Use 20260911 and record actual UTC timestamps, never copy an upstream run date.",
        "preconditions_checked": (
            "Actual code, resource, identity and gate observations before expensive work."
        ),
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
        "source_artifact_hashes": (
            "Bind code, source documents, manifests and raw evidence to claims."
        ),
        "rows": (
            "Per unit/arm/seed metric, error and abstention for every comparison; retain full "
            "denominators."
        ),
        "sample_size_budget": (
            "Historical, new, excluded and accepted counts; target 10 remains cumulative."
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
        "model_invoked": (
            "True only for actual model execution; upstream model outputs are cached evidence."
        ),
        "arc_session_complete_score": "Complete bounded receipt, independent of a win.",
        "solve_provenance": "live_agent_self_discovery; no new level-solve claim.",
        "per_game_results": "Raw banked progress and action counts.",
        "tool_induction_rows": "Unique identities, call events, validity and terminal outcomes.",
        "cumulative_induction_rows": "Deduplicate hashes without relabeling historical rows.",
        "adapter_isolation_receipt": "Actual live-policy access boundary.",
        "new_solve_claimed": "False; registry-prechecked public-game measurement.",
        "model_identity_receipt": (
            "GGUF revision/hash, exact loader and tokenizer, and actual CUDA execution."
        ),
        "gpu_receipts": "Task-owned lease and correlated CUDA observations.",
        "phase_spans": "Monotonic load/generate/score/cleanup spans.",
        "runner_receipt": "One model, runner choice, PID identity and cleanup.",
        "terminal_timestamp_utc": "Record when this process built the terminal receipt in UTC.",
        "historical_identity_receipt": (
            "Authenticate retained V635 artifacts and preserve their existing induction identities."
        ),
    }
)

_ORIGINAL_OPTIONAL_SIBLING = base._optional_sibling
_ORIGINAL_COLLECT_STATIC_PRECONDITIONS = base.collect_static_preconditions
_ORIGINAL_BUILD_TERMINAL_ARTIFACT = base.build_terminal_artifact
_ORIGINAL_VALIDATE_ARTIFACT = base.validate_artifact

_BASE_OVERRIDES: dict[str, Any] = {
    "TASK_ID": TASK_ID,
    "EXPERIMENT_ID": EXPERIMENT_ID,
    "MILESTONE": MILESTONE,
    "RUN_DATE": RUN_DATE,
    "GAME": GAME,
    "RANDOM_SEED": RANDOM_SEED,
    "ACTION_BUDGET": ACTION_BUDGET,
    "SESSION_TIMEOUT_S": SESSION_TIMEOUT_S,
    "INDUCTION_TIMEOUT_S": INDUCTION_TIMEOUT_S,
    "HARD_CAP_S": HARD_CAP_S,
    "N_CTX": N_CTX,
    "COMPLETION_BUDGET": COMPLETION_BUDGET,
    "EVIDENCE_TARGET": EVIDENCE_TARGET,
    "MODEL_ID": MODEL_ID,
    "QUANTIZATION": QUANTIZATION,
    "MODEL_SPECS": MODEL_SPECS,
    "EXPECTED_PRIOR_VERDICT": EXPECTED_PRIOR_VERDICT,
    "PRIOR_ARTIFACT_PATH": PRIOR_ARTIFACT_PATH,
    "SCHEMA": SCHEMA,
    "DRIVING_REQUIREMENT": DRIVING_REQUIREMENT,
    "MODULE_PATH": MODULE_PATH,
    "WRAPPER_PATH": WRAPPER_PATH,
    "TEST_PATH": TEST_PATH,
    "RESULT_PATH": RESULT_PATH,
    "CHECKPOINT_PATH": CHECKPOINT_PATH,
    "CHECKPOINT_SCHEMA": CHECKPOINT_SCHEMA,
    "RAW_DIR": RAW_DIR,
    "SIBLING_PATH": SIBLING_PATH,
    "SIBLING_TASK_ID": SIBLING_TASK_ID,
    "EXPECTED_TASK_CONTRACT": EXPECTED_TASK_CONTRACT,
    "REQUIRED_SOURCE_PATHS": REQUIRED_SOURCE_PATHS,
    "FIELD_PRINCIPLES": FIELD_PRINCIPLES,
}

_REUSED_FIELDS = (
    "TASK_ID",
    "MILESTONE",
    "RUN_DATE",
    "GAME",
    "RANDOM_SEED",
    "ACTION_BUDGET",
    "SESSION_TIMEOUT_S",
    "INDUCTION_TIMEOUT_S",
    "N_CTX",
    "COMPLETION_BUDGET",
    "RESULT_PATH",
    "CHECKPOINT_PATH",
    "RAW_DIR",
    "WRAPPER_PATH",
)


def authenticated_v635_cumulative_rows(
    root: Path, source_hashes: JsonDict
) -> tuple[JsonDict, list[JsonDict]]:
    """Authenticate both retained artifacts, then return their latest cumulative rows."""

    receipts: list[JsonDict] = []
    accepted_rows: list[list[JsonDict]] = []
    for expected_id, relative in zip((7206, 7207), V635_ARTIFACT_PATHS, strict=True):
        path = root / relative
        payload, read_error = base._load_json(path)
        content_hash = base.sha256_file(path) if read_error is None else "missing"
        source_hashes[relative.as_posix()] = content_hash
        quarantined = base.is_quarantined(payload) if read_error is None else None
        completion: Any = "not_consumed"
        checksum_valid: bool | None = None
        experiment_id: Any = "not_consumed"
        rows: Any = "not_consumed"
        invalid_ids: list[Any] = []
        duplicate_ids: list[str] = []
        consumed = False
        if read_error is None and quarantined is False:
            completion = base.unwrap_evidence_value(payload.get("arc_session_complete_score"))
            experiment_id = base.unwrap_evidence_value(payload.get("experiment_id"))
            checksum = payload.get("reproducibility_checksum")
            checksum_valid = bool(
                isinstance(checksum, str)
                and base.HASH_RE.fullmatch(checksum)
                and checksum == base.artifact_checksum(payload)
            )
            raw_rows = payload.get("cumulative_induction_rows")
            rows = raw_rows if isinstance(raw_rows, list) else []
            ids = [row.get("induction_id") for row in rows if isinstance(row, Mapping)]
            invalid_ids = [
                identity for identity in ids if not base.HASH_RE.fullmatch(str(identity))
            ]
            duplicate_ids = sorted({str(identity) for identity in ids if ids.count(identity) > 1})
            consumed = bool(
                completion == 1
                and experiment_id == expected_id
                and checksum_valid
                and isinstance(raw_rows, list)
                and len(ids) == len(raw_rows)
                and not invalid_ids
                and not duplicate_ids
                and all(
                    isinstance(row, Mapping)
                    and row.get("source_authenticated") is True
                    and row.get("engaged") is True
                    for row in raw_rows
                )
            )
        receipt = {
            "source": f"exp{expected_id}",
            "path": relative.as_posix(),
            "source_hash": content_hash,
            "read_error": read_error,
            "quarantined": quarantined,
            "completion_value": completion,
            "experiment_id": experiment_id,
            "artifact_checksum_valid": checksum_valid,
            "retained_inductions": len(rows) if isinstance(rows, list) else 0,
            "invalid_induction_ids": invalid_ids,
            "duplicate_induction_ids": duplicate_ids,
            "consumed": consumed,
        }
        receipts.append(receipt)
        if consumed:
            accepted_rows.append([deepcopy(dict(row)) for row in rows])

    all_accepted = len(accepted_rows) == len(V635_ARTIFACT_PATHS)
    latest_rows = accepted_rows[-1] if all_accepted else []
    older_ids = {row["induction_id"] for row in accepted_rows[0]} if all_accepted else set()
    latest_ids = {row["induction_id"] for row in latest_rows}
    lineage_preserved = bool(all_accepted and older_ids <= latest_ids)
    if not lineage_preserved:
        latest_rows = []
    return (
        {
            "source": SIBLING_TASK_ID,
            "state": "authenticated" if lineage_preserved else "rejected_required_source",
            "sources": receipts,
            "historical_rows_selected_from": V635_ARTIFACT_PATHS[-1].as_posix(),
            "older_ids_retained_by_latest": lineage_preserved,
            "historical_rows_relabelled": False,
            "accepted_unique_inductions": len(latest_rows),
            "excluded_inductions": sum(
                int(row["retained_inductions"]) for row in receipts if row["consumed"] is False
            ),
            "consumed": lineage_preserved,
        },
        latest_rows,
    )


def _collect_static_preconditions_configured(
    *, root: Path, result_path: Path, checkpoint_path: Path, raw_dir: Path
) -> tuple[list[JsonDict], JsonDict, JsonDict]:
    checks, source_hashes, context = _ORIGINAL_COLLECT_STATIC_PRECONDITIONS(
        root=root,
        result_path=result_path,
        checkpoint_path=checkpoint_path,
        raw_dir=raw_dir,
    )
    optional = context.get("optional_source_receipts", [])
    identity = next(
        (
            row
            for row in optional
            if isinstance(row, Mapping) and row.get("source") == SIBLING_TASK_ID
        ),
        {},
    )
    checks.extend(
        [
            base.gate_check(
                "current_task_identity",
                MODULE_PATH.as_posix(),
                "TASK_ID",
                TASK_ID,
                base.TASK_ID,
            ),
            base.gate_check(
                "current_seed_default",
                MODULE_PATH.as_posix(),
                "RANDOM_SEED",
                RANDOM_SEED,
                base.RANDOM_SEED,
            ),
            base.gate_check(
                "authenticated_v635_cumulative_identity",
                ",".join(path.as_posix() for path in V635_ARTIFACT_PATHS),
                "quarantine,complete,checksum,induction_ids",
                "authenticated",
                identity.get("state"),
            ),
        ]
    )
    return checks, source_hashes, context


def _build_terminal_artifact_configured(**kwargs: Any) -> JsonDict:
    artifact = _ORIGINAL_BUILD_TERMINAL_ARTIFACT(**kwargs)
    receipts = artifact.get("optional_source_receipts", [])
    historical = next(
        (
            row
            for row in receipts
            if isinstance(row, Mapping) and row.get("source") == SIBLING_TASK_ID
        ),
        {},
    )
    sample = artifact.get("sample_size_budget", {})
    if isinstance(sample, dict):
        sample.update(
            {
                "historical_reported_unique_inductions": int(
                    historical.get("accepted_unique_inductions", 0) or 0
                ),
                "historical_excluded_inductions": int(
                    historical.get("excluded_inductions", 0) or 0
                ),
                "accepted_cumulative_unique_inductions": len(
                    artifact.get("cumulative_induction_rows", [])
                ),
                "excluded_cumulative_inductions": int(
                    historical.get("excluded_inductions", 0) or 0
                ),
            }
        )
    current = artifact.get("tool_induction_rows", [])
    current_rows = current if isinstance(current, list) else []
    valid_world_models = sum(
        bool(row.get("terminal_result_returned"))
        and not row.get("model_validity_errors")
        and not row.get("error")
        for row in current_rows
        if isinstance(row, Mapping)
    )
    for row in artifact.get("per_game_results", []):
        if not isinstance(row, dict):
            continue
        engagements = int(row.get("new_tool_loop_inductions", 0) or 0)
        actions = int(row.get("actions", 0) or 0)
        row["tool_engagement"] = {
            "returned_tool_loop_inductions": engagements,
            "tool_calls": int(row.get("tool_calls", 0) or 0),
        }
        row["useful_world_model_validation"] = {
            "accepted_world_models": valid_world_models,
            "attempted_inductions": len(current_rows),
            "criterion": "terminal_result_without_model_validity_or_projection_error",
        }
        row["action_efficiency"] = {
            "actions": actions,
            "actions_per_returned_tool_loop": actions / engagements if engagements else None,
            "causal_comparator_available": False,
        }
        row["transition_errors"] = [
            error
            for induction in current_rows
            if isinstance(induction, Mapping)
            for error in induction.get("model_validity_errors", [])
            if "transition" in str(error)
        ]
    artifact["historical_identity_receipt"] = deepcopy(dict(historical))
    artifact["terminal_timestamp_utc"] = datetime.now(timezone.utc).isoformat()
    artifact["field_principles"]["historical_identity_receipt"] = FIELD_PRINCIPLES[
        "historical_identity_receipt"
    ]
    artifact["field_principles"]["terminal_timestamp_utc"] = FIELD_PRINCIPLES[
        "terminal_timestamp_utc"
    ]
    artifact["reproducibility_checksum"] = base.artifact_checksum(artifact)
    return artifact


def _validate_artifact_configured(value: Mapping[str, Any] | str | Path) -> list[str]:
    errors = _ORIGINAL_VALIDATE_ARTIFACT(value)
    if isinstance(value, (str, Path)):
        loaded, read_error = base._load_json(Path(value))
        artifact = loaded if read_error is None else {}
    else:
        artifact = dict(value)
    sample = artifact.get("sample_size_budget", {})
    required_counts = {
        "historical_reported_unique_inductions",
        "historical_excluded_inductions",
        "new_tool_loop_inductions",
        "accepted_cumulative_unique_inductions",
        "excluded_cumulative_inductions",
    }
    if not isinstance(sample, Mapping) or not required_counts <= set(sample):
        errors.append("sample_size_acceptance_counts_missing")
    elif sample.get("accepted_cumulative_unique_inductions") != len(
        artifact.get("cumulative_induction_rows", [])
    ):
        errors.append("accepted_cumulative_count_mismatch")
    receipt = artifact.get("historical_identity_receipt", {})
    historical_gate = next(
        (
            row
            for row in artifact.get("preconditions_checked", [])
            if isinstance(row, Mapping)
            and row.get("check") == "authenticated_v635_cumulative_identity"
        ),
        {},
    )
    if historical_gate.get("passed") is True and (
        not isinstance(receipt, Mapping)
        or receipt.get("state") != "authenticated"
        or receipt.get("historical_rows_relabelled") is not False
    ):
        errors.append("historical_identity_receipt_invalid")
    try:
        timestamp = datetime.fromisoformat(str(artifact.get("terminal_timestamp_utc")))
        if timestamp.tzinfo is None:
            raise ValueError("timezone missing")
    except ValueError:
        errors.append("terminal_timestamp_utc_invalid")
    if artifact.get("status") == "complete":
        for row in artifact.get("tool_induction_rows", []):
            if isinstance(row, Mapping) and (
                row.get("source_session_id") != TASK_ID or row.get("seed") != RANDOM_SEED
            ):
                errors.append("current_induction_identity_mismatch")
                break
    runner = artifact.get("runner_receipt", {})
    if artifact.get("model_invoked") is True and (
        not isinstance(runner, Mapping) or runner.get("model_count") != 1
    ):
        errors.append("model_invoked_without_single_model_runner")
    return list(dict.fromkeys(errors))


@contextmanager
def configured_runtime() -> Iterator[Any]:
    """Apply V636 identity and reducer seams only during this process scope."""

    scoped = {
        **_BASE_OVERRIDES,
        "_optional_sibling": authenticated_v635_cumulative_rows,
        "collect_static_preconditions": _collect_static_preconditions_configured,
        "build_terminal_artifact": _build_terminal_artifact_configured,
        "validate_artifact": _validate_artifact_configured,
    }
    base_before = {name: getattr(base, name) for name in scoped}
    reused_before = {name: getattr(base.reused, name) for name in _REUSED_FIELDS}
    try:
        for name, value in scoped.items():
            setattr(base, name, deepcopy(value))
        yield base
    finally:
        for name, value in base_before.items():
            setattr(base, name, value)
        for name, value in reused_before.items():
            setattr(base.reused, name, value)


def run_experiment(
    *,
    root: Path,
    run_date: str,
    result_path: Path,
    checkpoint_path: Path,
    raw_dir: Path,
    live_runner: Callable[..., Any] | None = None,
) -> JsonDict:
    """Run one bounded session through the repaired shared implementation."""

    with configured_runtime() as configured:
        return configured.run_experiment(
            root=root,
            run_date=run_date,
            result_path=result_path,
            checkpoint_path=checkpoint_path,
            raw_dir=raw_dir,
            live_runner=live_runner,
        )


def validate_artifact(value: Mapping[str, Any] | str | Path) -> list[str]:
    """Cold-validate the V636 receipt with shared and historical-identity checks."""

    with configured_runtime() as configured:
        return configured.validate_artifact(value)


def run_session_child(args: Any) -> int:
    """Run the isolated model child with the V636 task and seed defaults."""

    with configured_runtime() as configured:
        return int(configured.run_session_child(args))


def main(argv: Sequence[str] | None = None) -> int:
    """Execute the V636 driver, isolated child, or cold validator."""

    with configured_runtime() as configured:
        return int(configured.main(argv))


if __name__ == "__main__":
    raise SystemExit(main())
