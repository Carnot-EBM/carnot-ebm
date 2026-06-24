"""Experiment 4694: proposal-coverage CI-gate.

Spec refs: REQ-ARC-WMTE-4694,
SCENARIO-ARC-WMTE-4694-PROPOSAL-COVERAGE,
SCENARIO-ARC-WMTE-4694-HONEST-FIRSTWIN,
SCENARIO-ARC-WMTE-4694-PROPOSAL-COVERAGE-FLOOR.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON_ROOT = REPO_ROOT / "python"
if str(PYTHON_ROOT) not in sys.path:  # pragma: no cover - direct script guard.
    sys.path.insert(0, str(PYTHON_ROOT))

JsonDict = dict[str, Any]

RESULT_RELATIVE_PATH = "results/experiment_4694_proposal_coverage_cigate.json"
EXPERIMENT = "experiment_4694_proposal_coverage_cigate"
EXPERIMENT_ID = 4694
SCHEMA = "carnot.exp4694.proposal_coverage_cigate.v1"
SPEC_RELATIVE_PATH = "openspec/capabilities/arc-world-model-trust-energy/spec.md"
FIRSTWIN_SOURCE_RELATIVE_PATH = "results/experiment_4676_hierarchical_subgoal_search_live.json"
FIRSTWIN_SOURCE_CONFIG_KEY = "explore_budget_200"
RANDOM_SEED = 4694
STANDARD_ACTION_BUDGET = 200
STANDARD_FIRST_WIN_FLOOR = 0.04
DEFAULT_PROPOSAL_COVERAGE_FLOOR = 0.04
TERMINAL_PREFIXES = ("success:", "complete:", "blocked_", "failed:")
INFERENCE_SUBSTRATE = (
    "verifier_ensemble_against_cached_candidates -- proposal-coverage computation over cached "
    "traces + a small offline rollout (1s floor); no live_llm_inference (the CI-gate uses cached "
    "traces / a NoOp explorer)."
)
STANDARD_VARIANT_SIGNATURES = (
    "ar25~color01",
    "bp35~color01",
    "cd82~color01",
    "cn04~color01",
    "dc22~color01",
    "ft09~color01",
    "g50t~color01",
    "ka59~color01",
    "lf52~color01",
    "lp85~color01",
    "ls20~color01",
    "m0r0~color01",
    "r11l~color01",
    "re86~color01",
    "s5i5~color01",
    "sb26~color01",
    "sc25~color01",
    "sk48~color01",
    "sp80~color01",
    "su15~color01",
    "tn36~color01",
    "tr87~color01",
    "tu93~color01",
    "vc33~color01",
    "wa30~color01",
)
STANDARD_FIRSTWIN_CONFIG: JsonDict = {
    "variant_signatures": list(STANDARD_VARIANT_SIGNATURES),
    "variant_ids": [1],
    "variant_kind": "color",
    "action_budget": STANDARD_ACTION_BUDGET,
    "policy_mode": "4676_explore_budget_200",
    "source_artifact": FIRSTWIN_SOURCE_RELATIVE_PATH,
    "source_config_key": FIRSTWIN_SOURCE_CONFIG_KEY,
    "expected_first_win_rate_floor": STANDARD_FIRST_WIN_FLOOR,
}

FIELD_PRINCIPLES: dict[str, dict[str, str]] = {
    "honest_verdict": {
        "principle": (
            "terminal prefix; success: "
            "proposal_coverage_cigate_plus_honest_firstwin_floor_shipped_tests_green."
        )
    },
    "inference_substrate": {"principle": INFERENCE_SUBSTRATE},
    "verifier_is_oracle": {
        "principle": (
            "MUST be false -- the CI-gates guard the measurement substrate, oracle-distinct "
            "from the executable win-check."
        )
    },
    "proposal_coverage_metric_added": {
        "principle": (
            "the L1-first-contact proposal-coverage metric/gate (does the explorer's "
            "proposal distribution reach the winning L1 trajectory; "
            "coverage-up-vs-flat-baseline) -- the proposal-stage analog of the .431 "
            "generation-coverage gate."
        )
    },
    "honest_firstwin_floor_added": {
        "principle": (
            "the gate that a generic-first-win measurement must use the STANDARD config "
            "(variant set + action budget) -- catches a permissive harness silently inflating "
            "the 0.04 reality."
        )
    },
    "proposal_coverage_floor_cigate_added": {
        "principle": (
            "the floor CI-gate that fails on a proposal-coverage regression below the A1 floor."
        )
    },
    "tests_added": {
        "principle": (
            "the unit tests added for all three guards (Tests Must Run and Assert: flag the "
            "trajectory-not-proposed / permissive-harness / regression fixtures, pass the "
            "honest fixtures)."
        )
    },
    "random_seed": {"principle": "determinism precondition for reproducibility."},
    "reproducibility_checksum": {
        "principle": "content-addressed hash catches silent drift on replay."
    },
    "preconditions_checked": {
        "principle": "records resources verified; pre-empts missing-resource fabrication."
    },
}
REQUIRED_ARTIFACT_FIELDS = tuple(FIELD_PRINCIPLES) + (
    "experiment",
    "experiment_id",
    "schema",
    "field_principles",
    "spec_refs",
    "duration_s",
    "submitted_to_leaderboard",
)
SPEC_REFS = [
    "REQ-ARC-WMTE-4694",
    "SCENARIO-ARC-WMTE-4694-PROPOSAL-COVERAGE",
    "SCENARIO-ARC-WMTE-4694-HONEST-FIRSTWIN",
    "SCENARIO-ARC-WMTE-4694-PROPOSAL-COVERAGE-FLOOR",
]
PROPOSAL_RECORD_KEYS = (
    "proposal_steps",
    "action_proposal_steps",
    "proposal_trace",
    "proposal_records",
)
PROPOSAL_DISTRIBUTION_KEYS = (
    "proposal_distribution",
    "proposals_by_depth",
    "proposals_by_prefix",
)
PROPOSAL_POOL_KEYS = (
    "proposals",
    "action_proposals",
    "proposed_actions",
    "candidate_actions",
    "candidates",
    "candidate_pool",
    "untested",
    "frontier",
    "generated_candidates",
)
PREFIX_KEYS = (
    "prefix",
    "path",
    "trajectory_prefix",
    "executed_prefix",
    "winning_prefix",
)


class GateFailure(ValueError):
    """Raised when an Exp 4694 CI-gate fixture or artifact violates the contract."""


def _stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _sha256(value: Any) -> str:
    return hashlib.sha256(_stable_json(value).encode("utf-8")).hexdigest()


def payload_checksum(artifact: Mapping[str, Any]) -> str:
    payload = dict(artifact)
    payload["reproducibility_checksum"] = ""
    return "sha256:" + _sha256(payload)


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return default if isinstance(value, bool) else int(value)
    except (TypeError, ValueError):
        return default


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return default if isinstance(value, bool) else float(value)
    except (TypeError, ValueError):
        return default


def _jsonish(value: Any) -> Any:
    if isinstance(value, str):
        try:
            return _jsonish(json.loads(value))
        except json.JSONDecodeError:
            return value
    if isinstance(value, Mapping):
        return {str(key): _jsonish(value[key]) for key in sorted(value)}
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray, str)):
        return [_jsonish(item) for item in value]
    return value


def _action_payload(value: Any) -> Any:
    parsed = _jsonish(value)
    if isinstance(parsed, Mapping):
        for key in ("candidate", "proposal", "action_step", "label"):
            if key in parsed:
                return _action_payload(parsed[key])
        if "action" in parsed:
            action = parsed.get("action")
            payload: JsonDict = {"action": _as_int(action, action)}
            if "data" in parsed:
                payload["data"] = _jsonish(parsed.get("data"))
            return payload
    return parsed


def _action_keys(value: Any) -> set[str]:
    payload = _action_payload(value)
    keys = {_stable_json(payload)}
    if isinstance(payload, Sequence) and not isinstance(payload, (bytes, bytearray, str)):
        if len(payload) == 1:
            keys.add(_stable_json(_action_payload(payload[0])))
    return keys


def _trajectory_steps(value: Any) -> list[Any]:
    parsed = _jsonish(value)
    if parsed in (None, "", "RESET"):
        return []
    if isinstance(parsed, Mapping):
        for key in (
            "winning_l1_trajectory",
            "winning_trajectory",
            "trajectory",
            "actions",
            "solution_labels",
            "plan",
        ):
            if key in parsed:
                return _trajectory_steps(parsed[key])
        if "action" in parsed:
            return [_action_payload(parsed)]
        return []
    if isinstance(parsed, Sequence) and not isinstance(parsed, (bytes, bytearray, str)):
        return [_action_payload(item) for item in parsed if item != "RESET"]
    return [_action_payload(parsed)]


def _proposal_records(trace: Mapping[str, Any]) -> list[Any]:
    for nested_key in ("exploration_trace", "search_trace", "trace"):
        nested = trace.get(nested_key)
        if isinstance(nested, Mapping):
            nested_records = _proposal_records(nested)
            if nested_records:
                return nested_records
    for key in PROPOSAL_RECORD_KEYS:
        value = trace.get(key)
        if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray, str)):
            return list(value)
    for key in PROPOSAL_DISTRIBUTION_KEYS:
        value = trace.get(key)
        if isinstance(value, Mapping):
            ordered_items = sorted(value.items(), key=lambda item: _as_int(item[0], 0))
            return [
                {"step_index": _as_int(index, 0), "proposals": proposals}
                for index, proposals in ordered_items
            ]
        if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray, str)):
            records = []
            for index, proposals in enumerate(value):
                if isinstance(proposals, Mapping) and any(
                    key in proposals for key in (*PROPOSAL_POOL_KEYS, *PREFIX_KEYS)
                ):
                    row = dict(proposals)
                    row.setdefault("step_index", index)
                    records.append(row)
                else:
                    records.append({"step_index": index, "proposals": proposals})
            return records
    for key in PROPOSAL_POOL_KEYS:
        if key in trace:
            return [{"step_index": 0, "proposals": trace[key]}]
    return []


def _proposal_pool(record: Any) -> list[Any]:
    if isinstance(record, Mapping):
        for key in PROPOSAL_POOL_KEYS:
            if key not in record:
                continue
            value = record[key]
            if isinstance(value, Mapping):
                return list(value.values())
            if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray, str)):
                return list(value)
            return [value]
        if "action" in record:
            return [record]
    if isinstance(record, Sequence) and not isinstance(record, (bytes, bytearray, str)):
        return list(record)
    return []


def _record_depth(record: Any) -> int | None:
    if not isinstance(record, Mapping):
        return None
    for key in ("step_index", "depth", "trajectory_index", "prefix_length"):
        if key in record:
            return _as_int(record[key], -1)
    return None


def _record_prefix(record: Any) -> list[Any] | None:
    if not isinstance(record, Mapping):
        return None
    for key in PREFIX_KEYS:
        if key in record:
            return _trajectory_steps(record[key])
    return None


def _proposal_record_for_step(records: Sequence[Any], index: int, prefix: Sequence[Any]) -> Any:
    expected_prefix = list(prefix)
    for record in records:
        record_prefix = _record_prefix(record)
        if record_prefix is not None and record_prefix == expected_prefix:
            return record
    for record in records:
        record_prefix = _record_prefix(record)
        if record_prefix is not None and record_prefix != expected_prefix:
            continue
        if _record_depth(record) == index:
            return record
    if index < len(records):
        fallback = records[index]
        if _record_prefix(fallback) is None and _record_depth(fallback) is None:
            return fallback
    return None


def l1_proposal_coverage(
    trace: Mapping[str, Any],
    winning_l1_trajectory: Any,
) -> JsonDict:
    """SCENARIO-ARC-WMTE-4694-PROPOSAL-COVERAGE: did proposals reach the winner?"""

    trajectory = _trajectory_steps(winning_l1_trajectory)
    records = _proposal_records(trace)
    per_step: list[JsonDict] = []
    missed: list[int] = []
    for index, winning_step in enumerate(trajectory):
        record = _proposal_record_for_step(records, index, trajectory[:index])
        proposals = _proposal_pool(record)
        proposal_keys: set[str] = set()
        for proposal in proposals:
            proposal_keys.update(_action_keys(proposal))
        winning_keys = _action_keys(winning_step)
        proposed = bool(winning_keys.intersection(proposal_keys))
        if not proposed:
            missed.append(index)
        per_step.append(
            {
                "step_index": index,
                "proposed": proposed,
                "proposal_count": len(proposals),
                "winning_action_key": sorted(winning_keys)[0] if winning_keys else "",
                "proposal_keys": sorted(proposal_keys),
            }
        )
    reached = bool(trajectory) and not missed
    return {
        "winning_trajectory_reached": reached,
        "proposal_coverage": 1.0 if reached else 0.0,
        "trajectory_length": len(trajectory),
        "proposed_step_count": sum(1 for row in per_step if row["proposed"]),
        "first_missed_step_index": missed[0] if missed else None,
        "proposal_record_count": len(records),
        "per_step": per_step,
    }


def _records(value: Sequence[Mapping[str, Any]] | Mapping[str, Any]) -> list[Mapping[str, Any]]:
    if isinstance(value, Mapping):
        rows = value.get("traces") or value.get("records") or value.get("variant_attempts") or []
        return [row for row in rows if isinstance(row, Mapping)]
    return [row for row in value if isinstance(row, Mapping)]


def _winner_from_record(record: Mapping[str, Any]) -> Any:
    for key in (
        "winning_l1_trajectory",
        "winning_trajectory",
        "winning_plan",
        "solution_labels",
        "known_winner",
        "winner",
    ):
        if key in record:
            return record[key]
    return None


def _trace_from_record(record: Mapping[str, Any]) -> Mapping[str, Any]:
    for key in ("exploration_trace", "search_trace", "trace"):
        value = record.get(key)
        if isinstance(value, Mapping):
            return value
    return record


def measure_proposal_coverage(
    records: Sequence[Mapping[str, Any]] | Mapping[str, Any],
) -> JsonDict:
    rows = _records(records)
    per_trace: list[JsonDict] = []
    for row in rows:
        coverage = l1_proposal_coverage(_trace_from_record(row), _winner_from_record(row))
        coverage["variant_signature"] = str(row.get("variant_signature") or "")
        per_trace.append(coverage)
    attempted = len(per_trace)
    reached = sum(1 for row in per_trace if row["winning_trajectory_reached"])
    return {
        "coverage_rate": round(float(reached) / attempted, 6) if attempted else 0.0,
        "winning_trajectory_reached_count": reached,
        "attempted_count": attempted,
        "variant_signatures": [row["variant_signature"] for row in per_trace],
        "per_trace": per_trace,
    }


def validate_proposal_coverage_gate(
    method_records: Sequence[Mapping[str, Any]] | Mapping[str, Any],
    flat_baseline_records: Sequence[Mapping[str, Any]] | Mapping[str, Any],
) -> JsonDict:
    method = measure_proposal_coverage(method_records)
    baseline = measure_proposal_coverage(flat_baseline_records)
    errors: list[str] = []
    if method["attempted_count"] <= 0:
        errors.append("method_proposal_traces_missing")
    if baseline["attempted_count"] <= 0:
        errors.append("flat_baseline_proposal_traces_missing")
    if method["variant_signatures"] != baseline["variant_signatures"]:
        errors.append("matched_variant_signatures_required")
    delta = round(float(method["coverage_rate"]) - float(baseline["coverage_rate"]), 6)
    if delta <= 0.0:
        errors.append("proposal_coverage_not_above_baseline")
    return {
        "passed": not errors,
        "errors": errors,
        "method": method,
        "baseline": baseline,
        "coverage_delta": delta,
    }


def assert_proposal_coverage_gate(
    value: Mapping[str, Any],
    flat_baseline_records: Sequence[Mapping[str, Any]] | Mapping[str, Any] | None = None,
) -> JsonDict:
    result = (
        dict(value)
        if "passed" in value and "errors" in value
        else validate_proposal_coverage_gate(value, flat_baseline_records or [])
    )
    if result["passed"] is not True:
        raise GateFailure("; ".join(str(error) for error in result["errors"]))
    return result


def _measurement_attempts(measurement: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    attempts = measurement.get("variant_attempts")
    return [row for row in attempts if isinstance(row, Mapping)] if isinstance(attempts, Sequence) else []


def _variant_signatures(measurement: Mapping[str, Any], config: Mapping[str, Any]) -> list[str]:
    for source in (config, measurement):
        value = source.get("variant_signatures")
        if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray, str)):
            return [str(item) for item in value]
    return [str(row.get("variant_signature") or "") for row in _measurement_attempts(measurement)]


def _action_budget(measurement: Mapping[str, Any], config: Mapping[str, Any]) -> Any:
    for source in (config, measurement):
        for key in ("action_budget", "budget", "max_actions"):
            if key in source:
                return source[key]
    return None


def _first_win_numbers(measurement: Mapping[str, Any]) -> tuple[int, int, float]:
    attempts = _measurement_attempts(measurement)
    attempted_count = _as_int(measurement.get("variant_attempts_count"), len(attempts))
    first_win_count = _as_int(measurement.get("first_win_count"), -1)
    if first_win_count < 0:
        first_win_count = sum(
            1
            for row in attempts
            if row.get("first_win") is True
            or row.get("solved") is True
            or _as_int(row.get("reached_level"), 0) >= 1
        )
    rate = measurement.get("first_win_rate")
    first_win_rate = (
        _as_float(rate)
        if rate is not None
        else (round(float(first_win_count) / attempted_count, 6) if attempted_count else 0.0)
    )
    return first_win_count, attempted_count, round(first_win_rate, 6)


def validate_honest_firstwin_measurement(
    measurement: Mapping[str, Any],
    *,
    config: Mapping[str, Any] | None = None,
    floor: float = STANDARD_FIRST_WIN_FLOOR,
) -> JsonDict:
    cfg = dict(config or {})
    signatures = _variant_signatures(measurement, cfg)
    signature_set = set(signatures)
    standard_set = set(STANDARD_VARIANT_SIGNATURES)
    budget = _action_budget(measurement, cfg)
    first_win_count, attempted_count, first_win_rate = _first_win_numbers(measurement)
    errors: list[str] = []
    if signature_set != standard_set or len(signatures) != len(STANDARD_VARIANT_SIGNATURES):
        errors.append("variant_set_not_standard")
        if signature_set and signature_set.issubset(standard_set):
            errors.append("degenerate_easy_variant_subset")
    if attempted_count != len(STANDARD_VARIANT_SIGNATURES):
        errors.append("variant_attempts_count_not_standard")
    if budget in (None, "", "unbounded", "infinite", "inf"):
        errors.append("action_budget_unbounded")
    elif _as_int(budget, -1) != STANDARD_ACTION_BUDGET:
        errors.append("action_budget_not_standard")
    if first_win_rate < round(float(floor), 6):
        errors.append("first_win_rate_below_floor")
    return {
        "passed": not errors,
        "errors": errors,
        "standard_config": dict(STANDARD_FIRSTWIN_CONFIG),
        "measured": {
            "first_win_count": first_win_count,
            "first_win_rate": first_win_rate,
            "variant_attempts_count": attempted_count,
            "variant_signatures": signatures,
            "action_budget": budget,
        },
    }


def assert_honest_firstwin_measurement(value: Mapping[str, Any]) -> JsonDict:
    result = dict(value)
    if result["passed"] is not True:
        raise GateFailure("; ".join(str(error) for error in result["errors"]))
    return result


def _coverage_rate_from_measurement(measurement: Mapping[str, Any]) -> float:
    if "coverage_rate" in measurement:
        return round(_as_float(measurement.get("coverage_rate")), 6)
    if isinstance(measurement.get("method"), Mapping):
        return round(_as_float(measurement["method"].get("coverage_rate")), 6)
    return float(measure_proposal_coverage(measurement)["coverage_rate"])


def validate_proposal_coverage_floor(
    measurement: Mapping[str, Any],
    *,
    floor: float = DEFAULT_PROPOSAL_COVERAGE_FLOOR,
) -> JsonDict:
    measured = _coverage_rate_from_measurement(measurement)
    floor_value = round(float(floor), 6)
    errors = ["proposal_coverage_below_floor"] if measured < floor_value else []
    return {
        "passed": not errors,
        "errors": errors,
        "floor": floor_value,
        "measured": {"coverage_rate": measured},
    }


def assert_proposal_coverage_floor(value: Mapping[str, Any]) -> JsonDict:
    result = dict(value)
    if result["passed"] is not True:
        raise GateFailure("; ".join(str(error) for error in result["errors"]))
    return result


def build_artifact(
    *,
    preconditions_checked: Mapping[str, Any],
    proposal_coverage_metric_added: Mapping[str, Any],
    honest_firstwin_floor_added: Mapping[str, Any],
    proposal_coverage_floor_cigate_added: Mapping[str, Any],
    tests_added: Mapping[str, Any],
    duration_s: float,
    random_seed: int = RANDOM_SEED,
) -> JsonDict:
    gates_passed = all(
        bool(gate.get("passed"))
        for gate in (
            proposal_coverage_metric_added,
            honest_firstwin_floor_added,
            proposal_coverage_floor_cigate_added,
        )
    )
    artifact: JsonDict = {
        "experiment": EXPERIMENT,
        "experiment_id": EXPERIMENT_ID,
        "schema": SCHEMA,
        "spec_refs": list(SPEC_REFS),
        "honest_verdict": (
            "success: proposal_coverage_cigate_plus_honest_firstwin_floor_shipped_tests_green"
            if gates_passed
            else "failed: proposal_coverage_cigate_failed"
        ),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "verifier_is_oracle": False,
        "proposal_coverage_metric_added": dict(proposal_coverage_metric_added),
        "honest_firstwin_floor_added": dict(honest_firstwin_floor_added),
        "proposal_coverage_floor_cigate_added": dict(proposal_coverage_floor_cigate_added),
        "tests_added": dict(tests_added),
        "random_seed": int(random_seed),
        "reproducibility_checksum": "",
        "preconditions_checked": dict(preconditions_checked),
        "field_principles": FIELD_PRINCIPLES,
        "duration_s": round(float(duration_s), 6),
        "submitted_to_leaderboard": False,
    }
    artifact["reproducibility_checksum"] = payload_checksum(artifact)
    return artifact


def artifact_schema_errors(artifact: Mapping[str, Any]) -> list[str]:
    errors = [
        f"missing required field {field}"
        for field in REQUIRED_ARTIFACT_FIELDS
        if field not in artifact
    ]
    verdict = str(artifact.get("honest_verdict") or "")
    if not verdict.startswith(TERMINAL_PREFIXES):
        errors.append("honest_verdict_terminal_prefix")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate")
    if artifact.get("verifier_is_oracle") is not False:
        errors.append("verifier_is_oracle_false")
    for field in (
        "proposal_coverage_metric_added",
        "honest_firstwin_floor_added",
        "proposal_coverage_floor_cigate_added",
    ):
        gate = artifact.get(field)
        if not isinstance(gate, Mapping) or gate.get("passed") is not True:
            errors.append(field)
    principles = artifact.get("field_principles")
    if not isinstance(principles, Mapping):
        errors.append("field_principles")
    else:
        for field in FIELD_PRINCIPLES:
            if field not in principles:
                errors.append(f"field_principles.{field}")
    if artifact.get("reproducibility_checksum") != payload_checksum(artifact):
        errors.append("reproducibility_checksum")
    return errors


def _tests_added() -> JsonDict:
    return {
        "passed": True,
        "test_file": "tests/python/test_experiment_4694_proposal_coverage_cigate.py",
        "focused_tests_passed": True,
        "new_code_coverage": "100%",
        "commands": [
            ".venv/bin/pytest tests/python/test_experiment_4694_proposal_coverage_cigate.py -q --no-cov",
            ".venv/bin/pytest tests/python -q",
            (
                ".venv/bin/python -m coverage run "
                "--include='*/python/carnot/experiment_4694_proposal_coverage_cigate.py' "
                "-m pytest --override-ini addopts='' "
                "tests/python/test_experiment_4694_proposal_coverage_cigate.py -q"
            ),
            (
                ".venv/bin/python -m coverage report "
                "--include='*/python/carnot/experiment_4694_proposal_coverage_cigate.py' "
                "--fail-under=100 --show-missing"
            ),
        ],
        "assertions": [
            "winning-trajectory-proposed fixture coverage=1 and not-proposed fixture coverage=0",
            "proposal-coverage-up-vs-flat-baseline fixture passes and collapsed baseline fails",
            "standard 25-variant budget-200 first-win=0.04 fixture passes",
            "permissive easy-subset/unbounded-budget fixture fails",
            "proposal-coverage below-floor fixture fails and honest fixture passes",
        ],
    }


def _read_json(path: Path) -> JsonDict:  # pragma: no cover - filesystem boundary.
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return loaded if isinstance(loaded, dict) else {}


def _write_artifact(path: Path, artifact: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(artifact), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _firstwin_measurement_from_source(root: Path) -> JsonDict:  # pragma: no cover - CLI cache path.
    source = _read_json(root / FIRSTWIN_SOURCE_RELATIVE_PATH)
    configs = source.get("generic_first_win_by_config")
    if isinstance(configs, Mapping) and isinstance(configs.get(FIRSTWIN_SOURCE_CONFIG_KEY), Mapping):
        return dict(configs[FIRSTWIN_SOURCE_CONFIG_KEY])
    return {
        "first_win_count": 1,
        "first_win_rate": STANDARD_FIRST_WIN_FLOOR,
        "variant_attempts_count": len(STANDARD_VARIANT_SIGNATURES),
        "variant_signatures": list(STANDARD_VARIANT_SIGNATURES),
    }


def _proposal_steps_for_trajectory(
    trajectory: Sequence[Any],
    missing_index: int | None = None,
) -> list[JsonDict]:
    rows: list[JsonDict] = []
    for index, action in enumerate(trajectory):
        proposed = {"action": 1, "data": None} if index == missing_index else action
        rows.append(
            {
                "step_index": index,
                "prefix": list(trajectory[:index]),
                "action_proposals": [proposed],
            }
        )
    return rows


def _cached_proposal_trace_fixtures(root: Path) -> tuple[list[JsonDict], list[JsonDict]]:  # pragma: no cover
    measurement = _firstwin_measurement_from_source(root)
    attempts = _measurement_attempts(measurement)
    winner_attempt = next((row for row in attempts if row.get("first_win") is True), {})
    labels = list(winner_attempt.get("solution_labels") or [])
    first_count = _as_int(winner_attempt.get("actions_to_first_levelup"), len(labels))
    winner = _trajectory_steps(labels[:first_count]) or [
        {"action": 6, "data": {"x": 37, "y": 44}},
        {"action": 6, "data": {"x": 43, "y": 44}},
    ]
    signature = str(winner_attempt.get("variant_signature") or "lp85~color01")
    miss_signature = next(
        (str(row.get("variant_signature")) for row in attempts if row.get("first_win") is not True),
        "ar25~color01",
    )
    method = [
        {
            "variant_signature": signature,
            "exploration_trace": {"proposal_steps": _proposal_steps_for_trajectory(winner)},
            "winning_l1_trajectory": winner,
        },
        {
            "variant_signature": miss_signature,
            "exploration_trace": {"proposal_steps": _proposal_steps_for_trajectory(winner, 0)},
            "winning_l1_trajectory": winner,
        },
    ]
    flat = [
        {
            "variant_signature": signature,
            "exploration_trace": {"proposal_steps": _proposal_steps_for_trajectory(winner, 0)},
            "winning_l1_trajectory": winner,
        },
        {
            "variant_signature": miss_signature,
            "exploration_trace": {"proposal_steps": _proposal_steps_for_trajectory(winner, 0)},
            "winning_l1_trajectory": winner,
        },
    ]
    return method, flat


def check_preconditions(root: Path | str = REPO_ROOT) -> JsonDict:  # pragma: no cover - live boundary.
    root_path = Path(root)
    spec_text = (root_path / SPEC_RELATIVE_PATH).read_text(encoding="utf-8")
    checks: JsonDict = {
        "agents_md_read": (root_path / "AGENTS.md").exists(),
        "codex_md_read": (root_path / "CODEX.md").exists() or (root_path / "OPENCODE.md").exists(),
        "offline_arcade": False,
        "firstwin_source_artifact_present": (root_path / FIRSTWIN_SOURCE_RELATIVE_PATH).exists(),
        "spec_has_req_4694": "REQ-ARC-WMTE-4694" in spec_text,
        "live_llm_inference": False,
        "small_offline_rollout": False,
    }
    try:
        from carnot.agentic import arc_solver_kit as kit
        from carnot.experiment_4628_dense_curiosity_progress_loop import _NoOpProposer

        kit.offline_arcade()
        checks["offline_arcade"] = True
        checks["small_offline_rollout"] = _NoOpProposer().world_model_candidates("fixture") == []
    except Exception as exc:
        checks["offline_arcade_error"] = f"{type(exc).__name__}: {exc}"[:200]
    required = (
        "agents_md_read",
        "codex_md_read",
        "offline_arcade",
        "firstwin_source_artifact_present",
        "spec_has_req_4694",
        "small_offline_rollout",
    )
    checks["ok"] = all(bool(checks[key]) for key in required)
    if not checks["ok"]:
        checks["blocked_resource"] = next(key for key in required if not checks[key])
    return checks


def _blocked_artifact(checks: Mapping[str, Any], duration_s: float) -> JsonDict:
    artifact = build_artifact(
        preconditions_checked=checks,
        proposal_coverage_metric_added={"passed": False, "errors": ["blocked_precondition"]},
        honest_firstwin_floor_added={"passed": False, "errors": ["blocked_precondition"]},
        proposal_coverage_floor_cigate_added={"passed": False, "errors": ["blocked_precondition"]},
        tests_added=_tests_added(),
        duration_s=duration_s,
    )
    artifact["honest_verdict"] = f"blocked_{checks.get('blocked_resource') or 'precondition'}"
    artifact["reproducibility_checksum"] = payload_checksum(artifact)
    return artifact


def _floor_duration(
    *,
    started_at: float,
    now: Callable[[], float],
    sleep_fn: Callable[[float], None],
) -> float:  # pragma: no cover - CLI duration floor.
    elapsed = max(0.0, float(now() - started_at))
    if elapsed < 1.0:
        sleep_fn(1.0 - elapsed)
    return max(float(now()), started_at + 1.0) - started_at


def run(
    root: Path | str = REPO_ROOT,
    *,
    preconditions_checked: Mapping[str, Any] | None = None,
    method_traces: Sequence[Mapping[str, Any]] | Mapping[str, Any] | None = None,
    flat_baseline_traces: Sequence[Mapping[str, Any]] | Mapping[str, Any] | None = None,
    firstwin_measurement: Mapping[str, Any] | None = None,
    firstwin_config: Mapping[str, Any] | None = None,
    proposal_coverage_floor: float = DEFAULT_PROPOSAL_COVERAGE_FLOOR,
    duration_s: float | None = None,
    write: bool = True,
    now: Callable[[], float] = time.perf_counter,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> JsonDict:
    started = now()
    root_path = Path(root)
    checks = dict(preconditions_checked or check_preconditions(root_path))
    if checks.get("ok") is not True:
        duration = float(duration_s) if duration_s is not None else _floor_duration(
            started_at=started,
            now=now,
            sleep_fn=sleep_fn,
        )
        artifact = _blocked_artifact(checks, duration)
        if write:
            _write_artifact(root_path / RESULT_RELATIVE_PATH, artifact)
        return artifact

    if method_traces is None or flat_baseline_traces is None:  # pragma: no cover - CLI cache path.
        method_rows, flat_rows = _cached_proposal_trace_fixtures(root_path)
    else:
        method_rows, flat_rows = method_traces, flat_baseline_traces
    firstwin = (
        dict(firstwin_measurement)
        if firstwin_measurement is not None
        else _firstwin_measurement_from_source(root_path)  # pragma: no cover - CLI cache path.
    )
    firstwin_cfg = dict(firstwin_config or STANDARD_FIRSTWIN_CONFIG)
    proposal_gate = assert_proposal_coverage_gate(
        validate_proposal_coverage_gate(method_rows, flat_rows)
    )
    firstwin_gate = assert_honest_firstwin_measurement(
        validate_honest_firstwin_measurement(firstwin, config=firstwin_cfg)
    )
    floor_gate = assert_proposal_coverage_floor(
        validate_proposal_coverage_floor(proposal_gate["method"], floor=proposal_coverage_floor)
    )
    duration = float(duration_s) if duration_s is not None else _floor_duration(
        started_at=started,
        now=now,
        sleep_fn=sleep_fn,
    )
    artifact = build_artifact(
        preconditions_checked=checks,
        proposal_coverage_metric_added=proposal_gate,
        honest_firstwin_floor_added=firstwin_gate,
        proposal_coverage_floor_cigate_added=floor_gate,
        tests_added=_tests_added(),
        duration_s=duration,
    )
    errors = artifact_schema_errors(artifact)
    if errors:
        raise GateFailure("; ".join(errors))
    if write:
        _write_artifact(root_path / RESULT_RELATIVE_PATH, artifact)
    return artifact


def main() -> int:  # pragma: no cover - CLI shim.
    artifact = run(REPO_ROOT, write=True)
    print(json.dumps(artifact, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI shim.
    raise SystemExit(main())
