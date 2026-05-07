"""Exp 1449 LTLZinc-style temporal continual-learning adapter.

This module intentionally implements a tiny finite-trace checker instead of
calling MiniZinc or a full LTLZinc runtime. The purpose of Exp 1449 is to give
FR-11 and DVI later milestones a stable stream of verified temporal cases:
small traces, clear labels, and an auditable local verifier that can separate
SAT examples from repair-hint examples without any fresh model inference.

Spec: REQ-LEARN-1449, SCENARIO-LEARN-1449.
"""

from __future__ import annotations

import json
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RESULTS_DIR = REPO_ROOT / "results"
OUTPUT_FILE = "experiment_1449_ltlzinc_temporal_continual_learning_adapter.json"
DATASET_FILE = "experiment_1449_ltlzinc_temporal_cases.jsonl"
DEFAULT_OUTPUT_PATH = DEFAULT_RESULTS_DIR / OUTPUT_FILE
DEFAULT_DATASET_PATH = DEFAULT_RESULTS_DIR / DATASET_FILE

EXPERIMENT = "1449_ltlzinc_temporal_continual_learning_adapter"
SCHEMA = "ltlzinc_temporal_continual_learning_adapter_v1"
RUN_DATE = "20260507"
MIN_CASES = 20
SUPPORTED_OPERATORS = ("always", "eventually", "next", "until")
REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "ltlzinc_adapter_ready",
    "temporal_cases_generated",
    "verifier_available",
    "accepted_case_count",
    "rejected_case_count",
    "dataset_path",
    "commands_run",
    "honest_verdict",
)


def _write_json(path: Path | str, artifact: Mapping[str, Any]) -> dict[str, Any]:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(artifact)
    destination.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    return payload


def _metadata(project_root: str | Path, run_date: str) -> dict[str, str]:
    return {"project_root": str(project_root), "run_date": run_date}


def write_in_progress_artifact(
    out_path: Path | str = DEFAULT_OUTPUT_PATH,
    *,
    project_root: str | Path = REPO_ROOT,
    run_date: str = RUN_DATE,
) -> dict[str, Any]:
    """REQ-LEARN-1449-1: write the bootstrap artifact before case generation."""

    return _write_json(
        out_path,
        {
            "experiment": EXPERIMENT,
            "schema": SCHEMA,
            "artifact_metadata": _metadata(project_root, run_date),
            "run_date": run_date,
            "status": "in_progress",
            "ltlzinc_adapter_ready": False,
            "temporal_cases_generated": 0,
            "verifier_available": False,
            "accepted_case_count": 0,
            "rejected_case_count": 0,
            "dataset_path": None,
            "commands_run": [],
            "honest_verdict": "in_progress",
        },
    )


def _normalize_trace(trace: Sequence[Mapping[str, Any]]) -> list[dict[str, bool | int]]:
    normalized: list[dict[str, bool | int]] = []
    for index, step in enumerate(trace):
        normalized_step: dict[str, bool | int] = {"t": index}
        for key, value in step.items():
            if key != "t":
                normalized_step[str(key)] = bool(value)
        normalized.append(normalized_step)
    return normalized


def _ltl_formula(operator: str, signal: str, guard_signal: str | None) -> str:
    if operator == "always":
        return f"G {signal}"
    if operator == "eventually":
        return f"F {signal}"
    if operator == "next":
        return f"X {signal}"
    return f"{guard_signal} U {signal}"


def _minizinc_constraint(operator: str, signal: str, guard_signal: str | None) -> str:
    if operator == "always":
        return f"constraint forall(t in TIME)({signal}[t]);"
    if operator == "eventually":
        return f"constraint exists(t in TIME)({signal}[t]);"
    if operator == "next":
        return f"constraint {signal}[2];"
    return f"constraint exists(t in TIME)({signal}[t] /\\ forall(p in 1..t-1)({guard_signal}[p]));"


def make_case(
    case_id: str,
    operator: str,
    signal: str,
    trace: Sequence[Mapping[str, Any]],
    expected_satisfied: bool,
    *,
    guard_signal: str | None = None,
    split: str = "train",
) -> dict[str, Any]:
    """Build one Carnot-friendly temporal case row.

    The row keeps both human-readable LTL/MiniZinc-style strings and structured
    fields. Later learning code can train on the text fields, while the local
    verifier uses the structured fields to avoid brittle formula parsing.
    """

    if operator not in SUPPORTED_OPERATORS:
        raise ValueError(f"unsupported temporal operator: {operator}")  # pragma: no cover
    normalized_trace = _normalize_trace(trace)
    guard = guard_signal if operator == "until" else None
    certificate_state = "SAT" if expected_satisfied else "REPAIR_HINT"
    return {
        "case_id": case_id,
        "source": "exp1449_ltlzinc_style_synthetic",
        "split": split,
        "constraint_family": operator,
        "temporal_operator": operator,
        "signal": signal,
        "guard_signal": guard,
        "ltl_formula": _ltl_formula(operator, signal, guard),
        "minizinc_constraint": _minizinc_constraint(operator, signal, guard),
        "trace": normalized_trace,
        "expected_satisfied": bool(expected_satisfied),
        "label": "accepted" if expected_satisfied else "rejected",
        "certificate_state": certificate_state,
        "dvi_label": 0 if expected_satisfied else 1,
        "fr11_memory_hint": (
            "promote_temporal_constraint_success"
            if expected_satisfied
            else "promote_temporal_constraint_violation"
        ),
    }


def _step_value(step: Mapping[str, Any], signal: str) -> bool:
    return bool(step.get(signal, False))


def _verify_until(
    trace: Sequence[Mapping[str, Any]],
    *,
    signal: str,
    guard_signal: str,
) -> bool:
    for index, step in enumerate(trace):
        if _step_value(step, signal):
            return all(_step_value(trace[prior], guard_signal) for prior in range(index))
    return False


def verify_temporal_case(case: Mapping[str, Any]) -> bool:
    """REQ-LEARN-1449-3: evaluate the supported temporal operators locally."""

    operator = str(case["temporal_operator"])
    signal = str(case["signal"])
    trace = case["trace"]
    if not isinstance(trace, Sequence) or isinstance(trace, (str, bytes)):
        raise TypeError("trace must be a sequence of step mappings")  # pragma: no cover

    if operator == "always":
        return all(_step_value(step, signal) for step in trace)
    if operator == "eventually":
        return any(_step_value(step, signal) for step in trace)
    if operator == "next":
        return len(trace) > 1 and _step_value(trace[1], signal)
    if operator == "until":
        guard = str(case["guard_signal"])
        return _verify_until(trace, signal=signal, guard_signal=guard)
    raise ValueError(f"unsupported temporal operator: {operator}")  # pragma: no cover


def validate_case_schema(case: Mapping[str, Any]) -> None:
    """Assert that one row has the fields needed by FR-11 and DVI later."""

    required = {
        "case_id",
        "constraint_family",
        "temporal_operator",
        "signal",
        "ltl_formula",
        "minizinc_constraint",
        "trace",
        "expected_satisfied",
        "certificate_state",
        "dvi_label",
        "fr11_memory_hint",
    }
    missing = sorted(required.difference(case))
    if missing:
        raise AssertionError(f"missing temporal case fields: {missing}")  # pragma: no cover
    if case["temporal_operator"] not in SUPPORTED_OPERATORS:
        raise AssertionError("unsupported temporal operator")  # pragma: no cover
    if case["certificate_state"] not in {"SAT", "REPAIR_HINT"}:
        raise AssertionError("unsupported certificate_state")  # pragma: no cover


def _template_rows() -> list[
    tuple[str, str, str | None, list[dict[str, bool]], list[dict[str, bool]]]
]:
    return [
        (
            "always",
            "power_ok",
            None,
            [{"power_ok": True}, {"power_ok": True}],
            [{"power_ok": False}],
        ),
        (
            "always",
            "access_safe",
            None,
            [{"access_safe": True}],
            [{"access_safe": True}, {"access_safe": False}],
        ),
        (
            "always",
            "below_limit",
            None,
            [{"below_limit": True}] * 3,
            [{"below_limit": True}, {"below_limit": False}],
        ),
        ("eventually", "ready", None, [{"ready": False}, {"ready": True}], [{"ready": False}] * 2),
        ("eventually", "ack", None, [{"ack": False}, {"ack": True}], [{"ack": False}] * 3),
        (
            "eventually",
            "recovered",
            None,
            [{"recovered": False}, {"recovered": True}],
            [{"recovered": False}],
        ),
        (
            "next",
            "armed",
            None,
            [{"armed": False}, {"armed": True}],
            [{"armed": True}, {"armed": False}],
        ),
        (
            "next",
            "token_valid",
            None,
            [{"token_valid": False}, {"token_valid": True}],
            [{"token_valid": False}, {"token_valid": False}],
        ),
        (
            "next",
            "checkpoint",
            None,
            [{"checkpoint": False}, {"checkpoint": True}],
            [{"checkpoint": True}],
        ),
        (
            "until",
            "ready",
            "waiting",
            [{"waiting": True, "ready": False}, {"waiting": False, "ready": True}],
            [{"waiting": False, "ready": False}, {"waiting": True, "ready": True}],
        ),
        (
            "until",
            "success",
            "retrying",
            [
                {"retrying": True, "success": False},
                {"retrying": True, "success": False},
                {"retrying": False, "success": True},
            ],
            [{"retrying": True, "success": False}, {"retrying": False, "success": False}],
        ),
        (
            "until",
            "released",
            "locked",
            [{"locked": True, "released": False}, {"locked": False, "released": True}],
            [{"locked": False, "released": False}, {"locked": False, "released": True}],
        ),
    ]


def generate_temporal_cases() -> list[dict[str, Any]]:
    """REQ-LEARN-1449-2: produce a deterministic 24-row temporal dataset."""

    cases: list[dict[str, Any]] = []
    for index, (operator, signal, guard, accepted_trace, rejected_trace) in enumerate(
        _template_rows()
    ):
        split = "train" if index < 8 else "eval"
        cases.append(
            make_case(
                f"ltlzinc-{operator}-{signal}-sat-{index:02d}",
                operator,
                signal,
                accepted_trace,
                True,
                guard_signal=guard,
                split=split,
            )
        )
        cases.append(
            make_case(
                f"ltlzinc-{operator}-{signal}-repair-hint-{index:02d}",
                operator,
                signal,
                rejected_trace,
                False,
                guard_signal=guard,
                split=split,
            )
        )
    return cases


def write_jsonl(path: Path | str, rows: Sequence[Mapping[str, Any]]) -> str:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    content = "\n".join(json.dumps(dict(row), sort_keys=True, ensure_ascii=True) for row in rows)
    destination.write_text(content + "\n", encoding="utf-8")
    return str(destination)


def build_artifact(
    *,
    cases: Sequence[Mapping[str, Any]],
    dataset_path: Path | str,
    project_root: str | Path = REPO_ROOT,
    run_date: str = RUN_DATE,
    started_at: str | None = None,
    duration_s: float = 0.0,
    commands_run: Sequence[str] | None = None,
) -> dict[str, Any]:
    """REQ-LEARN-1449-4/5: summarize generated cases and future feed paths."""

    family_states: dict[str, set[str]] = {operator: set() for operator in SUPPORTED_OPERATORS}
    for case in cases:
        validate_case_schema(case)
        expected_satisfied = bool(case["expected_satisfied"])
        if verify_temporal_case(case) != expected_satisfied:
            raise AssertionError("temporal verifier disagrees with case label")  # pragma: no cover
        family_states[str(case["constraint_family"])].add(str(case["certificate_state"]))

    counts = Counter(str(case["certificate_state"]) for case in cases)
    accepted_count = int(counts.get("SAT", 0))
    rejected_count = int(counts.get("REPAIR_HINT", 0))
    balanced = all(states == {"SAT", "REPAIR_HINT"} for states in family_states.values())
    ready = len(cases) >= MIN_CASES and accepted_count > 0 and rejected_count > 0 and balanced
    artifact = {
        "experiment": EXPERIMENT,
        "schema": SCHEMA,
        "artifact_metadata": _metadata(project_root, run_date),
        "run_date": run_date,
        "started_at": started_at or datetime.now(tz=UTC).isoformat(),
        "finished_at": datetime.now(tz=UTC).isoformat(),
        "duration_s": round(float(duration_s), 3),
        "status": "complete" if ready else "blocked",
        "spec": ["REQ-LEARN-1449", "SCENARIO-LEARN-1449"],
        "ltlzinc_adapter_ready": ready,
        "temporal_cases_generated": len(cases),
        "verifier_available": True,
        "accepted_case_count": accepted_count,
        "rejected_case_count": rejected_count,
        "dataset_path": str(dataset_path),
        "supported_operators": list(SUPPORTED_OPERATORS),
        "operator_family_certificate_states": {
            family: sorted(states) for family, states in family_states.items()
        },
        "later_milestone_feed": {
            "fr11": (
                "FR-11 can ingest REPAIR_HINT rows as verified temporal "
                "violation memory and SAT rows as replay nonforgetting anchors."
            ),
            "dvi": (
                "DVI can train a contrastive verifier with dvi_label=0 for SAT "
                "rows and dvi_label=1 for REPAIR_HINT rows."
            ),
        },
        "scope_note": (
            "This run creates local finite-trace verified cases only; it does "
            "not train a DVI adapter and does not execute MiniZinc."
        ),
        "commands_run": list(commands_run or []),
        "honest_verdict": (
            "ltlzinc_temporal_adapter_ready_verified_cases_only_no_training"
            if ready
            else "ltlzinc_temporal_adapter_blocked_unbalanced_cases"
        ),
    }
    validate_artifact(artifact)
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """REQ-LEARN-1449-4: enforce the final artifact contract."""

    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    if missing:
        raise AssertionError(f"missing required fields: {missing}")  # pragma: no cover
    if artifact["status"] == "in_progress":
        return
    generated = int(artifact["temporal_cases_generated"])
    accepted = int(artifact["accepted_case_count"])
    rejected = int(artifact["rejected_case_count"])
    if accepted + rejected != generated:
        raise AssertionError("accepted/rejected counts must equal generated cases")
    if generated < MIN_CASES:
        raise AssertionError("temporal case count below minimum")  # pragma: no cover
    if artifact["status"] == "complete" and not artifact["ltlzinc_adapter_ready"]:
        raise AssertionError("complete artifact requires ready adapter")  # pragma: no cover
    if not artifact["verifier_available"]:
        raise AssertionError("terminal artifact requires verifier_available")  # pragma: no cover


def run(
    *,
    out_path: Path | str = DEFAULT_OUTPUT_PATH,
    dataset_path: Path | str = DEFAULT_DATASET_PATH,
    project_root: str | Path = REPO_ROOT,
    run_date: str = RUN_DATE,
    commands_run: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Run Exp 1449 end-to-end, writing JSONL dataset and terminal artifact."""

    started_at = datetime.now(tz=UTC).isoformat()
    t0 = time.perf_counter()
    write_in_progress_artifact(out_path, project_root=project_root, run_date=run_date)
    cases = generate_temporal_cases()
    write_jsonl(dataset_path, cases)
    artifact = build_artifact(
        cases=cases,
        dataset_path=dataset_path,
        project_root=project_root,
        run_date=run_date,
        started_at=started_at,
        duration_s=time.perf_counter() - t0,
        commands_run=commands_run,
    )
    return _write_json(out_path, artifact)


if __name__ == "__main__":  # pragma: no cover
    print(json.dumps(run(), indent=2, sort_keys=True))
